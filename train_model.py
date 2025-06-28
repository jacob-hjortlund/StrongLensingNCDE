import os
import jax
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)

import torch
import optax
import hydra

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import stronglensingncde.loss as loss
import stronglensingncde.utils as utils
import stronglensingncde.models as models
import stronglensingncde.datasets as datasets
import stronglensingncde.training as training

from pathlib import Path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./stronglensingncde/config", config_name="config")
def train(cfg: DictConfig) -> None:
    value = os.getenv("NVIDIA_TF32_OVERRIDE", "<not set>")
    print(f"NVIDIA_TF32_OVERRIDE is: {value}")

    value = os.getenv("XLA_FLAGS", "<not set>")
    print(f"XLA_FLAGS is: {value}")

    print(f"\nWorking directory : {os.getcwd()}")
    print(f"Hydra Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}\n")

    print("\n")
    print(cfg)
    print("\n")
    
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    rng_key = jr.PRNGKey(cfg['seed'])

    data_path = Path(cfg['training']['path'])
    save_path = Path(cfg['model']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    train_path = data_path / "train.h5"
    val_path = data_path / "val.h5"

    # ---------------------------- Create Dataloaders ---------------------------- #

    stats = np.load(data_path / "train_statistics.npz", allow_pickle=True)['arr_0'].tolist()

    flux_norm = datasets.create_normalization_func(stats, "TRANS_FLUX")
    flux_err_norm = datasets.create_normalization_func(stats, "TRANS_FLUX_ERR")
    photoz_norm = datasets.create_normalization_func(stats, "TRANS_PHOTOZ")
    photoz_err_norm = datasets.create_normalization_func(stats, "TRANS_PHOTOZ_ERR")
    specz_norm = datasets.create_normalization_func(stats, "TRANS_SPECZ")
    specz_err_norm = photoz_err_norm

    redshift_norm = datasets.create_redshift_transform(
        specz_norm, specz_err_norm, photoz_norm, photoz_err_norm
    )

    train_dataloader, train_dataset = datasets.make_dataloader(
        h5_path=train_path,
        flux_transform=flux_norm,
        flux_err_transform=flux_err_norm,
        redshift_transform=redshift_norm,
        **cfg['training']['data_settings']
    )

    val_dataloader, val_dataset = datasets.make_dataloader(
        h5_path=val_path,
        flux_transform=flux_norm,
        flux_err_transform=flux_err_norm,
        redshift_transform=redshift_norm,
        **cfg['training']['data_settings']
    )

    if isinstance(train_dataset, torch.utils.data.dataset.Subset):
        n_classes = len(train_dataset.dataset.class_counts_array)
    else:
        n_classes = len(train_dataset.class_counts_array)
    cfg['model']['hyperparams']['num_classes'] = n_classes
    steps_per_epoch = cfg['training']['data_settings']['steps_per_epoch']
    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataloader)
    num_full_passes = cfg['training']['data_settings']['num_full_passes']
    num_batches_in_dataset = len(train_dataloader)
    epochs_in_full_pass = num_batches_in_dataset / steps_per_epoch
    num_epochs = int(np.ceil(num_full_passes * epochs_in_full_pass))

    if cfg['training']['data_settings']['verbose']:
        print(f"\nBatches / Epoch: {steps_per_epoch}")
        print(f"Num. Full Passes: {num_full_passes}")
        print(f"Num. Batches in Train Dataset: {num_batches_in_dataset}")
        print(f"Epochs In One Full Pass: {epochs_in_full_pass}")
        print(f"Num. Epochs For {num_full_passes} Full Passes: {num_epochs}\n")

    # ---------------------------- Loss Function Setup --------------------------- #

    loss_fn = loss.make_loss_and_metric_fn(
        **cfg['training']['loss_settings']
    )

    # ------------------------------ Optimizer Setup ----------------------------- #

    num_warmup_epochs = cfg['training']['num_warmup_epochs']
    warmup_steps = num_warmup_epochs * steps_per_epoch
    total_steps = warmup_steps + num_epochs * steps_per_epoch

    ncde_lr_schedule = training.make_lr_schedule(
        cfg['training']['ncde_lr_schedule']['fn'],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lr_schedule_settings=cfg['training']['ncde_lr_schedule']['settings']
    )

    classifier_lr_schedule = training.make_lr_schedule(
        cfg['training']['classifier_lr_schedule']['fn'],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lr_schedule_settings=cfg['training']['classifier_lr_schedule']['settings']
    )

    optimizer_fn = getattr(
        optax,
        cfg['training']['optimizer_fn']
    )

    ncde_mask_fn = training.make_optimizer_mask_fn('ncde')
    ncde_optimizer = optax.masked(
        optax.inject_hyperparams(optimizer_fn)(ncde_lr_schedule),
        mask=ncde_mask_fn
    )
    
    classifier_mask_fn = training.make_optimizer_mask_fn('classifier')
    classifier_optimizer = optax.masked(
        optax.inject_hyperparams(optimizer_fn)(classifier_lr_schedule),
        mask=classifier_mask_fn
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        ncde_optimizer,
        classifier_optimizer,
    )

    # ------------------------- Model Setup and Training ------------------------- #

    model_class = getattr(
        models,
        cfg['model']['class']
    )
    model = utils.make_model(
        key=rng_key,
        model_class=model_class,
        hyperparams=cfg['model']['hyperparams']
    )
    utils.save_hyperparams(save_path / "hyperparams.eqx", cfg['model']['hyperparams'])

    if isinstance(
        cfg['training']['loss_settings']['loss_components'], str
    ):
        num_loss_components = 1
    else:
        num_loss_components = len(
            cfg['training']['loss_settings']['loss_components']
        )

    model, optimizer_state, train_log, val_log = training.training_loop(
        model=model,
        loss_fn=loss_fn,
        num_loss_components=num_loss_components,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        number_of_epochs=num_epochs,
        number_of_warmup_epochs=num_warmup_epochs,
        steps_per_epoch=steps_per_epoch,
        save_path=save_path,
        **cfg['training']['training_settings']
    )

if __name__ == "__main__":
    train()