import os
import jax
jax.config.update("jax_enable_x64", True)

import loss
import utils
import optax
import models
import datasets

import training

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import hydra

from pathlib import Path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./config", config_name="config")
def train(cfg: DictConfig) -> None:

    print(f"\nWorking directory : {os.getcwd()}")
    print(f"Hydra Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}\n")

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

    class_counts = train_dataset.class_counts_array
    beta = 0.999
    weights = (1-beta) / (1-beta**class_counts)
    
    print("Class counts:")
    print(class_counts)
    print(" ")

    print("Weights:")
    for count, weight in zip(class_counts, weights):
        print(f"{count}: {weight:.8f}")
    

    steps_per_epoch = cfg['training']['data_settings']['steps_per_epoch']
    num_full_passes = cfg['training']['data_settings']['num_full_passes']
    num_batches_in_dataset = len(train_dataloader)
    epochs_in_full_pass = int(np.ceil(num_batches_in_dataset / steps_per_epoch))
    num_epochs = num_full_passes * epochs_in_full_pass

    # ---------------------------- Loss Function Setup --------------------------- #

    loss_fn = loss.make_loss_and_metric_fn(
        #**OmegaConf.to_container(cfg['training']['loss_settings'])
        **cfg['training']['loss_settings']
    )

    # ------------------------------ Optimizer Setup ----------------------------- #

    num_warmup_epochs = cfg['training']['num_warmup_epochs']
    warmup_steps = num_warmup_epochs * steps_per_epoch
    total_steps = warmup_steps + num_epochs * steps_per_epoch

    lr_schedule_fn = getattr(
        optax.schedules,
        cfg['training']['lr_schedule_fn']
    )
    lr_schedule = lr_schedule_fn(
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        **cfg['training']['lr_schedule_settings']
    )

    optimizer_fn = getattr(
        optax,
        cfg['training']['optimizer_fn']
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(
            optimizer_fn
        )(lr_schedule)
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
        number_of_epochs=num_epochs+num_warmup_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose_steps=True,
        save_path=save_path,
    )

if __name__ == "__main__":
    train()