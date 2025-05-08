

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

from pathlib import Path

rng_key = jr.PRNGKey(42)

# data_path = Path("/pscratch/sd/h/hjrtlnd/elasticc_data")
# train_path = data_path / "jolteon_train.h5"
# val_path = data_path / "jolteon_val.h5"
# save_path = Path("/pscratch/sd/h/hjrtlnd/ncde_classifier/specz_xe_loss")
# save_path.mkdir(parents=True, exist_ok=True)
data_path = Path("/home/jacob/PhD/Projects/jolteon_model/Data")
train_path = data_path / "jolteon_train.h5"
val_path = data_path / "jolteon_val.h5"
save_path = Path("/home/jacob/PhD/Projects/jolteon_model/test/")

# ----------------------------- Data Hyperparams ----------------------------- #

steps_per_epoch = 60
num_full_passes = 10
batch_size = 128
shuffle = True
num_workers = 0
sample_redshift = True
sample_redshift_probs = jnp.array([1., 0., 0.])

# ----------------------------- Model Hyperparams ---------------------------- #

model_hyperparams = {
    'input_feature_size': 33,
    'representation_size': 8,
    'ncde_width': 128,
    'ncde_depth': 1,
    'ncde_solver': 'Tsit5',
    'ncde_adjoint': 'ReversibleAdjoint',
    'ncde_max_steps': int(2**20),
    'classifier_width': 128,
    'classifier_depth': 2,
    'num_classes': 11,
    'ncde_rtol': 1e-3,
    'ncde_atol': 1e-3
}

# ----------------------------- Loss Hyperparams ----------------------------- #

use_class_frequencies = False
loss_hyperparams = {
    'loss_components': [
        loss.temporal_cross_entropy_loss,
    ],
    'loss_fn_kwargs': {
        'class_frequencies': jnp.ones(model_hyperparams['num_classes']),
    },
    'metric_fns': loss.temporal_predictions,
    'metric_fn_kwargs': {},
    'temporal_weight_fns': loss.unit_weight_fn,
    'temporal_weight_fn_kwargs': {},
    
}


# ---------------------------- Create Dataloaders ---------------------------- #

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
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=False,
    flux_transform=flux_norm,
    flux_err_transform=flux_norm,
    redshift_transform=redshift_norm,
    sample_redshift=sample_redshift,
    sample_redshift_probs=sample_redshift_probs,
    max_obs=None,
)

val_dataloader, val_dataset = datasets.make_dataloader(
    h5_path=val_path,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=False,
    flux_transform=flux_norm,
    flux_err_transform=flux_norm,
    redshift_transform=redshift_norm,
    sample_redshift=sample_redshift,
    sample_redshift_probs=sample_redshift_probs,
    max_obs=None,
)

num_batches_in_dataset = len(train_dataloader)
epochs_in_full_pass = int(np.ceil(num_batches_in_dataset / steps_per_epoch))
num_epochs = num_full_passes * epochs_in_full_pass

# ---------------------------- Loss Function Setup --------------------------- #

if use_class_frequencies:
    class_frequencies = jnp.asarray(train_dataloader.dataset.class_frequencies_array)
    loss_hyperparams['class_frequencies'] = class_frequencies

loss_fn = loss.make_loss_and_metric_fn(
    **loss_hyperparams,
)

# ------------------------------ Optimizer Setup ----------------------------- #

init_lr = 1e-6
min_lr = 1e-6
max_lr = 2e-3
num_warmup_epochs = 5
decay_steps = num_epochs * steps_per_epoch
warmup_steps = num_warmup_epochs * steps_per_epoch
total_steps = warmup_steps + decay_steps

lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
    init_value=init_lr,
    peak_value=max_lr,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=min_lr
)

optimizer = optax.inject_hyperparams(optax.adamw)(lr_schedule)

# ------------------------- Model Setup and Training ------------------------- #

model = utils.make_model(
    key=rng_key,
    model_class=models.PoolingONCDEClassifier,
    hyperparams=model_hyperparams,
)
utils.save_hyperparams(save_path / "hyperparams.eqx", model_hyperparams)

model, optimizer_state, train_log, val_log = training.training_loop(
    model=model,
    loss_fn=loss_fn,
    num_loss_components=len(loss_hyperparams['loss_components']),
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    number_of_epochs=num_epochs+num_warmup_epochs,
    steps_per_epoch=steps_per_epoch,
    verbose_steps=True,
    save_path=save_path,
)
