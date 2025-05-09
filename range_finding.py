import os
import jax
jax.config.update("jax_enable_x64", True)

import loss
import hydra
import utils
import models
import datasets


import numpy as np
import seaborn as sns
import jax.random as jr
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

colors = sns.color_palette("colorblind")

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


    # ---------------------------- Loss Function Setup --------------------------- #

    loss_fn = loss.make_loss_and_metric_fn(
        **cfg['training']['loss_settings']
    )

    # ------------------------- Model Setup and Training ------------------------- #

    model_class = getattr(
        models,
        cfg['model']['class']
    )

    lrs, losses = utils.lr_range_test(
        rng_key=rng_key,
        model_class=model_class,
        model_hyperparams=cfg['model']['hyperparams'],
        loss_fn=loss_fn,
        train_loader=train_dataloader,
        **cfg['range_finding']
    )


    median_losses = np.median(losses, axis=0)
    p25, p75 = np.percentile(losses, [25, 75], axis=0)

    np.save(save_path / "range_finding_losses.npy", losses)
    np.save(save_path / "range_finding_lrs.npy", lrs[0])

    fig, ax = plt.subplots(ncols=2, figsize=(20, 5))

    ax[0].plot(lrs[0], median_losses, c=colors[0], label='Loss')
    ax[0].fill_between(
        lrs[0],
        p25,
        p75,
        color=colors[0],
        alpha=0.5,
        label='IQR'
    )

    ax[1].plot(lrs[0], median_losses, c=colors[0], label='Loss')
    ax[1].fill_between(
        lrs[0],
        p25,
        p75,
        color=colors[0],
        alpha=0.5,
        label='IQR'
    )

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[0].legend()
    ax[1].legend()

    ax[0].set_xlabel("Learning rate")
    ax[1].set_xlabel("Learning rate")
    ax[0].set_ylabel("Training loss")
    fig.suptitle("LR range test")
    fig.tight_layout()

    fig.savefig(save_path / 'lr_range_test.pdf')

if __name__ == "__main__":
    train()