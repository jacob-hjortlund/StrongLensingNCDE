import jax
jax.config.update("jax_enable_x64", True)
import models
import diffrax
import datasets
import training
import loss

import equinox as eqx

import optax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from pathlib import Path

data_path = Path("/home/jacob/PhD/Projects/jolteon_model/Data")
train_path = data_path / "jolteon_train.h5"
val_path = data_path / "jolteon_val.h5"


stats = np.load(data_path / "jolteon_train_statistics.npz", allow_pickle=True)['arr_0'].tolist()

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
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    flux_transform=flux_norm,
    flux_err_transform=flux_norm,
    redshift_transform=redshift_norm,
    sample_redshift=True,
)

val_dataloader, val_dataset = datasets.make_dataloader(
    h5_path=val_path,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    flux_transform=flux_norm,
    flux_err_transform=flux_norm,
    redshift_transform=redshift_norm,
    sample_redshift=True,
)

input_feature_size = 33
representation_size = 8
ncde_width = 128
ncde_depth = 1
ncde_solver = diffrax.Reversible(diffrax.Tsit5())
ncde_adjoint = diffrax.ReversibleAdjoint()
ncde_max_steps = 4096
ncde_rtol = 1e-3
ncde_atol = 1e-3
classifier_width = 128
classifier_depth = 2
num_classes = 11

rng_key = jr.PRNGKey(42)

model = models.PoolingONCDEClassifier(
    input_feature_size=input_feature_size,
    representation_size=representation_size,
    ncde_width=ncde_width,
    ncde_depth=ncde_depth,
    ncde_adjoint=ncde_adjoint,
    ncde_solver=ncde_solver,
    ncde_max_steps=ncde_max_steps,
    ncde_rtol = 1e-3,
    ncde_atol = 1e-3,
    classifier_width=classifier_width,
    classifier_depth=classifier_depth,
    num_classes=num_classes,
    key=rng_key
)

batch = next(iter(val_dataloader))
batch = [output.numpy() for output in batch]

times = batch[0][:, :50]
flux = batch[1][:, :, :50, :]
partial_ts = batch[2][:, :, :50, :]
trigger_idx = batch[3]
length = jnp.minimum(batch[4], 50)
peak_times = batch[5]
max_time = jnp.minimum(batch[6], times[:, -1])
labels = batch[8]
valid_lightcurve_mask = batch[9]

s, interp_s, interp_ts = training.batch_mapped_interpolate_timeseries(times, flux, partial_ts)
s_max = max_time + (length - 1)/1000

loss_fn = loss.make_loss_and_metric_fn(
    base_loss_fns=loss.temporal_cross_entropy_loss,
    loss_fn_kwargs={
        'class_frequencies': train_dataloader.dataset.class_frequencies_array,
    },
    metric_fns=loss.temporal_predictions,
    metric_fn_kwargs={},
    temporal_weight_fns=loss.unit_weight_fn,
    temporal_weight_fn_kwargs={},
)

loss_fn(
    model,
    s[:,0,:],
    s_max,
    interp_s,
    interp_ts,
    trigger_idx,
    length,
    labels,
    peak_times,
    valid_lightcurve_mask
)