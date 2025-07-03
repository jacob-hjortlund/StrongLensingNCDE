import os
import jax
import hydra
import torch

import numpy as np
import seaborn as sns
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import stronglensingncde.loss as loss
import stronglensingncde.utils as utils
import stronglensingncde.models as models
import stronglensingncde.datasets as datasets

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

colors = sns.color_palette("colorblind")

@hydra.main(version_base=None, config_path="./stronglensingncde/config", config_name="config")
def train(cfg: DictConfig) -> None:

    print(f"\nWorking directory : {os.getcwd()}")
    print(f"Hydra Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}\n")

    cfg = OmegaConf.to_container(cfg, resolve=True)

    rng_key = jr.PRNGKey(cfg['seed'])

    data_path = Path(cfg['training']['path'])
    save_path = Path(cfg['model']['save_path']) / "range_finding"
    save_path.mkdir(parents=True, exist_ok=True)
    
    train_path = data_path / "train.h5"

    # ---------------------------- Create Dataloaders ---------------------------- #

    stats = np.load(data_path / "train_statistics.npz", allow_pickle=True)['arr_0'].tolist()

    #flux_norm = datasets.create_normalization_func(stats, "TRANS_FLUX")
    #flux_err_norm = datasets.create_normalization_func(stats, "TRANS_FLUX_ERR")
    photoz_norm = datasets.create_normalization_func(stats, "TRANS_PHOTOZ")
    photoz_err_norm = datasets.create_normalization_func(stats, "TRANS_PHOTOZ_ERR")
    specz_norm = datasets.create_normalization_func(stats, "TRANS_SPECZ")
    specz_err_norm = photoz_err_norm

    redshift_norm = datasets.create_redshift_norm(
        specz_norm, specz_err_norm, photoz_norm, photoz_err_norm
    )

    train_dataloader, train_dataset = datasets.make_dataloader(
        h5_path=train_path,
        flux_norm='mean',
        flux_err_norm=f'mean',
        redshift_norm=redshift_norm,
        **cfg['training']['data_settings']
    )

    if isinstance(train_dataset, torch.utils.data.dataset.Subset):
        train_dataset = train_dataset.dataset
    
    class_frequencies = train_dataset.class_frequencies_array
    class_weights = jnp.asarray(1./ class_frequencies) / len(class_frequencies)
    # ---------------------------- Loss Function Setup --------------------------- #

    loss_fn = loss.make_loss_and_metric_fn(
        class_weights=class_weights,
        **cfg['training']['loss_settings']
    )

    # ------------------------- Model Setup and Training ------------------------- #

    model_class = getattr(
        models,
        cfg['model']['class']
    )

    (
        lrs, losses,
        gradnorms,
        grad2weight_ratios,
        update2weight_ratios
    ) = utils.lr_range_test(
        rng_key=rng_key,
        model_class=model_class,
        model_hyperparams=cfg['model']['hyperparams'],
        loss_fn=loss_fn,
        train_loader=train_dataloader,
        **cfg['range_finding']
    )


    median_losses = np.median(losses, axis=0)
    median_gradnorms = np.median(gradnorms, axis=0)
    median_update2weight_ratios = np.median(update2weight_ratios, axis=0)

    if len(losses) > 1:
        lp25, lp75 = np.percentile(losses, [25, 75], axis=0)
        gp25, gp75 = np.percentile(gradnorms, [25, 75], axis=0)
        up25, up75 = np.percentile(update2weight_ratios, [25, 75], axis=0)

    np.save(save_path / "range_finding_losses.npy", losses)
    np.save(save_path / "range_finding_lrs.npy", lrs[0])
    np.save(save_path / "range_finding_grad_norms.npy", gradnorms)
    np.save(save_path / "range_finding_g2w_ratios.npy", grad2weight_ratios)
    np.save(save_path / "range_finding_u2w_ratios.npy", update2weight_ratios)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
    ax=ax.flatten()

    ax[0].plot(lrs[0], median_losses, c=colors[0], label='Loss')
    ax[1].plot(lrs[0], median_losses, c=colors[0], label='Loss')
    ax[2].plot(lrs[0], median_gradnorms, c=colors[0], label='Grad. Norm')
    ax[3].plot(lrs[0], median_update2weight_ratios, c=colors[0], label='Update / Weight')
    
    if len(losses) > 1:
        ax[0].fill_between(
            lrs[0],
            lp25,
            lp75,
            color=colors[0],
            alpha=0.5,
            label='IQR'
        )
        ax[1].fill_between(
            lrs[0],
            lp25,
            lp75,
            color=colors[0],
            alpha=0.5,
            label='IQR'
        )
        ax[2].fill_between(
            lrs[0],
            gp25,
            gp75,
            color=colors[0],
            alpha=0.5,
            label='IQR'
        )
        ax[3].fill_between(
            lrs[0],
            up25,
            up75,
            color=colors[0],
            alpha=0.5,
            label='IQR'
        )

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    ax[2].set_xlabel("Learning rate")
    ax[3].set_xlabel("Learning rate")
    ax[0].set_ylabel("Training loss")
    ax[2].set_ylabel("Grad. Norm")
    ax[3].set_ylabel("Update / Weight Ratio")
    fig.suptitle("LR range test")
    fig.tight_layout()

    fig.savefig(save_path / 'lr_range_test.pdf')

if __name__ == "__main__":
    train()