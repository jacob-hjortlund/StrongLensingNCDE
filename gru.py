import os

import jax
import torch
import optax
import diffrax

import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import stronglensingncde.utils as utils
import stronglensingncde.training as training
import stronglensingncde.datasets as datasets

from time import time
from pathlib import Path
from jax.scipy.special import logsumexp
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from collections import defaultdict
from typing import Callable

import warnings
warnings.filterwarnings('ignore')

with initialize(
    version_base=None,
    config_path="./stronglensingncde/config",
):
    cfg = compose(
        config_name="config",
        overrides=[
            "training.path=/mimer/NOBACKUP/groups/naiss2025-22-731/jolteon_challenge/Data/JOLTEON_V1_PREPROCESSED_V3",
            #"training.path=/pscratch/sd/h/hjrtlnd/Data/JOLTEON_V1_PREPROCESSED_V3",
            #"training.path=/home/jacob/PhD/Projects/jolteon_model/Data/JOLTEON_V1_PREPROCESSED_V3",
            "model.save_dir=/pscratch/sd/h/hjrtlnd/StrongLensingNCDE_Results",
            "training.num_warmup_epochs=6",
            "training.data_settings.steps_per_epoch=null",
            "training.data_settings.num_full_passes=50",
            "training.data_settings.batch_size=1024",
            "training.accumulate_gradients=False",
            "training.accumulate_gradients_steps=4",
            "training.data_settings.classes=[Ia,II,AGN,91bg,Ib/c,TDE,Iax,SLSN]",
            #"training.data_settings.subsample=True",
            "training.data_settings.num_workers=12",
            "training.data_settings.min_num_detections=3",
            "training.data_settings.min_num_observations=3",
            "training.data_settings.t_delta=0.001",
            "training.data_settings.dtype=float32",
            "training.loss_settings.temporal_weight_fns=unit_weight_fn",
            "training.training_settings.val_steps_per_epoch=null",
            "training.training_settings.save_every_n_epochs=5",
            "training.training_settings.patience=50",
            "training.training_settings.only_use_first_column=True",
            "training.training_settings.except_on_failure=True",
            "training.loss_settings.loss_components=[temporal_cross_entropy_loss]",
            "training.loss_settings.loss_modifiers=[class_weight_loss_fn]",
            "training.loss_settings.loss_modifier_kwargs.gamma=2",
            "training.loss_settings.loss_modifier_kwargs.eps=1e-5",
            "training.loss_settings.loss_scales=[1.0]",
            "training.ncde_lr_schedule.settings.peak_value=1e-3",
            "training.classifier_lr_schedule.settings.peak_value=1e-2",
            "model.hyperparams.weight_init_fn=identity",
            "model.hyperparams.representation_size=64",
            "model.hyperparams.ncde_max_steps=4096",
            "model.hyperparams.ncde_weight_norm=False",
            "model.hyperparams.ncde_width=256",
            "model.hyperparams.ncde_depth=2",
            "model.hyperparams.ncde_num_stacks=1",
            "model.hyperparams.ncde_atol=1e-6",
            "model.hyperparams.ncde_use_jump_ts=False",
            "model.hyperparams.ncde_throw=False",
            "model.hyperparams.ncde_cast_f64=False",
            "model.hyperparams.ncde_gated=True",
            "model.hyperparams.ncde_use_noise=False",
            "model.hyperparams.ncde_additive_noise_scale=0.0",
            "model.hyperparams.ncde_multiplicative_noise_scale=1.0",
            "model.hyperparams.ncde_dtmin=1e-7",
            "model.hyperparams.ncde_dtype=float32",
            "model.hyperparams.ncde_solver=ReversibleHeun",
            "model.hyperparams.ncde_adjoint=RecursiveCheckpointAdjoint",
            "model.hyperparams.classifier_width=256",
            "model.hyperparams.classifier_depth=1",
            "model.hyperparams.classifier_dtype=float32",
            "model.hyperparams.checkpoint_ncde=False",
            "model.name=reg_gated_1024_Ia_Ibc_full",
            "range_finding.num_steps=100",
            "range_finding.repeats=5",
            "range_finding.lr_min=1e-4",
            "range_finding.lr_max=1e-3",
        ]
    )
    print(cfg)

cfg = OmegaConf.to_container(cfg, resolve=True)
rng_key = jr.PRNGKey(cfg['seed'])

data_path = Path(cfg['training']['path'])
save_path = Path(cfg['model']['save_path'])

train_path = data_path / "train.h5"
val_path = data_path / "val.h5"

# ---------------------------- Create Dataloaders ---------------------------- #

stats = np.load(data_path / "train_statistics.npz", allow_pickle=True)['arr_0'].tolist()

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
    flux_err_norm='mean',
    redshift_norm=redshift_norm,
    **cfg['training']['data_settings']
)

if isinstance(train_dataset, torch.utils.data.dataset.Subset):
    flux_norm = train_dataset.dataset.flux_norm
    flux_err_norm = train_dataset.dataset.flux_err_norm
    class_frequencies = train_dataset.dataset.class_frequencies_array
else:
    flux_norm = train_dataset.flux_norm
    flux_err_norm = train_dataset.flux_err_norm
    class_frequencies = train_dataset.class_frequencies_array
class_weights = jnp.asarray(1./ class_frequencies) / len(class_frequencies)
num_classes = len(class_weights)

val_dataloader, val_dataset = datasets.make_dataloader(
    h5_path=val_path,
    flux_norm=flux_norm,
    flux_err_norm=flux_err_norm,
    redshift_norm=redshift_norm,
    **cfg['training']['data_settings']
)

class GRU(eqx.Module):
    cell: eqx.nn.GRUCell
    init: eqx.nn.MLP | None
    init_dropout: eqx.nn.Dropout | eqx.nn.Identity
    cell_dropout: eqx.nn.Dropout | eqx.nn.Identity
    init_layernorm: eqx.nn.LayerNorm | eqx.nn.Identity
    cell_layernorm: eqx.nn.LayerNorm | eqx.nn.Identity
    online: bool = eqx.field(static=True)
    use_metadata: bool = eqx.field(static=True)
    use_layernorm: bool = eqx.field(static=True)
    use_dropout: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size,
        hidden_size,
        key,
        online=True,
        use_dropout=False,
        dropout_rate=None,
        use_layernorm=False,
        use_metadata=False,
        metadata_size=None,
        init_width=None,
        init_depth=None,
        **kwargs
    ):
        
        init_key, gru_key = jr.split(key, 2)

        self.online = online
        self.cell = eqx.nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            key=gru_key,
            **kwargs
        )

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.cell_layernorm = eqx.nn.LayerNorm(hidden_size)
        else:
            self.cell_layernorm = eqx.nn.Identity()

        self.use_dropout = use_dropout
        if use_dropout:
            self.cell_dropout = eqx.nn.Dropout(p=dropout_rate)
        else:
            self.cell_dropout = eqx.nn.Identity()

        self.use_metadata = use_metadata
        if use_metadata:
            self.init = eqx.nn.MLP(
                in_size=metadata_size,
                out_size=hidden_size,
                width_size=init_width,
                depth=init_depth,
                key=init_key
            )
            if use_layernorm:
                self.init_layernorm = eqx.nn.LayerNorm(hidden_size)
            else:
                self.init_layernorm = eqx.nn.Identity()
            if use_dropout:
                self.init_dropout = eqx.nn.Dropout(p=dropout_rate)
            else:
                self.init_dropout = eqx.nn.Identity()
        else:
            self.init = None

    def __call__(self, xs, metadata, key):

        n_states = xs.shape[0]
        states = jnp.zeros(
            (n_states, self.cell.hidden_size)
        )
        init_key, carry_key = jr.split(key, 2)

        if self.use_metadata:
            init_state = self.init(metadata)
            init_state = self.init_layernorm(init_state)
            init_state = self.init_dropout(init_state, key=init_key)
        else:
            init_state = jnp.zeros(self.cell.hidden_size)

        def scan_fn(carry, input):
            i, key, input_state, states = carry
            output_key, key = jr.split(key, 2)
            output_state = self.cell(input, input_state)
            output_state = self.cell_layernorm(output_state)
            saved_output_state = self.cell_dropout(output_state, key=output_key)
            states = states.at[i].set(saved_output_state)
            output_carry = (i+1, key, output_state, states)

            return (output_carry, None) 

        input_carry = (0, carry_key, init_state, states)

        outputs, _ = jax.lax.scan(scan_fn, input_carry, xs)
        states = outputs[-1]
        
        if self.online:
            return states
        else:
            return states[-1]
        
class GRUClassifier(eqx.Module):
    grus: tuple[GRU, ...]
    classifier: eqx.nn.MLP
    online: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_gru_layers=1,
        classifier_width=100,
        classifier_depth=1,
        online=True,
        use_layernorm=False,
        use_dropout=False,
        dropout_rate=None,
        use_metadata=False,
        metadata_size=None,
        init_width=None,
        init_depth=None,
        *,
        key,
        **kwargs
    ):

        self.online = online

        keys = jr.split(key, num_gru_layers + 1)
        gru_keys = keys[:-1]
        classifier_key = keys[-1]

        grus = [
            GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                online=online,
                use_layernorm=use_layernorm,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
                use_metadata=use_metadata,
                metadata_size=metadata_size,
                init_width=init_width,
                init_depth=init_depth,
                key=gru_keys[0],
                **kwargs
            )
        ]
        if num_gru_layers > 1:
            for i in range(num_gru_layers-1):
                grus.append(
                    GRU(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        online=online,
                        use_layernorm=use_layernorm,
                        use_dropout=use_dropout,
                        dropout_rate=dropout_rate,
                        use_metadata=use_metadata,
                        metadata_size=metadata_size,
                        init_width=init_width,
                        init_depth=init_depth,
                        key=gru_keys[i+1]
                    )
                )
        self.grus = tuple(grus)

        self.classifier = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_classes,
            width_size=classifier_width,
            depth=classifier_depth,
            key=classifier_key
        )
    
    def __call__(self, x, metadata, key):

        gru_input = x
        for gru in self.grus:
            gru_key, key = jr.split(key, 2)
            gru_input = gru(
                xs=gru_input,
                metadata=metadata,
                key=gru_key
            )
        gru_output = gru_input

        if self.online:
            logits = jax.vmap(self.classifier)(gru_output)
        else:
            logits = self.classifier(gru_output)

        return logits, gru_output

def count_params(model: eqx.Module):
  num_params = sum(
      x.size for x in jax.tree_util.tree_leaves(
          eqx.filter(model, eqx.is_array)
      )
  )
  return num_params

def cross_entropy_loss(logits, label):

    log_py = jax.nn.log_softmax(logits, axis=-1)
    mask = jnp.arange(log_py.shape[-1]) == label
    loss = -jnp.sum(mask * log_py)

    return loss

def make_mask(ts, trigger_idx, length):

    indeces = jnp.arange(ts.shape[0])
    length_mask = indeces < length
    trigger_mask = indeces >= trigger_idx
    mask = jnp.logical_and(length_mask, trigger_mask)

    return mask

def make_stable_mask(ts_logits, trigger_idx, length, label):

    mask = make_mask(ts_logits, trigger_idx, length)
    preds = np.argmax(ts_logits, axis=-1)
    preds = jnp.where(
        mask,
        preds,
        label
    )
    correct_mask = preds == label

    rev_cum = jnp.cumprod(correct_mask[::-1])
    consecutive_correct_mask = rev_cum[::-1].astype(bool)
    stable_mask = jnp.logical_and(mask, consecutive_correct_mask)

    return stable_mask

def temporal_cross_entropy_loss(ts_logits, trigger_idx, length, label):

    mask = make_mask(ts_logits, trigger_idx, length)
    loss_values = jax.vmap(cross_entropy_loss, in_axes=(0, None))(ts_logits, label)
    ts_loss = jnp.sum(mask * loss_values) / jnp.sum(mask)

    return ts_loss

def _stable_focal_weight(p, gamma=1.0, eps=1e-20):

    log_one_minus_p = jnp.log1p(-p)
    log_w = gamma * log_one_minus_p
    log_S = logsumexp(log_w)
    focal_weight = jnp.clip(
        jnp.exp(log_w - log_S),
        min=eps,
        max=1.0-eps
    )

    return focal_weight

def focal_weight_fn(logits, label, gamma=1.0, eps=1e-7,):

    probs = jax.nn.softmax(logits, axis=-1)
    p_true = jnp.clip(probs[:, label], min=eps, max=1.0-eps)
    focal_weight = _stable_focal_weight(p_true, gamma, eps)

    return focal_weight

def step_regularisation(rep_diff, cov):

    cov = cov + 1e-8
    inv_cov = jnp.diag(1./ cov)
    step_val = rep_diff @ inv_cov @ rep_diff

    return step_val

def temporal_path_reg(ts_logits, trigger_idx, length, reps, cov):

    mask = make_mask(ts_logits, trigger_idx, length)
    rep_diffs = jnp.diff(reps, axis=0)
    step_regs = jax.vmap(step_regularisation, in_axes=(0, None))(rep_diffs, cov)
    step_regs = jnp.concatenate([jnp.zeros(1), step_regs])
    path_reg = jnp.sum(mask * step_regs) / jnp.sum(mask)

    return path_reg
    
def temporal_focal_cross_entropy_loss(ts_logits, trigger_idx, length, label):

    mask = make_mask(ts_logits, trigger_idx, length)
    loss_values = jax.vmap(cross_entropy_loss, in_axes=(0, None))(ts_logits, label)
    focal_weights = focal_weight_fn(ts_logits, label)
    ts_loss = jnp.sum(mask * focal_weights * loss_values) / jnp.sum(mask * focal_weights)

    return ts_loss

def temporal_accuracy(ts_logits, trigger_idx, length, label):

    mask = make_mask(ts_logits, trigger_idx, length)
    preds = np.argmax(ts_logits, axis=-1)
    correct_mask = preds == label
    accuracy = jnp.sum(mask * correct_mask) / jnp.sum(mask)

    return accuracy

def temporal_stable_accuracy(ts_logits, trigger_idx, length, label):

    mask = make_mask(ts_logits, trigger_idx, length)
    stable_mask = make_stable_mask(ts_logits, trigger_idx, length, label)
    accuracy = jnp.sum(stable_mask) / jnp.sum(mask)

    return accuracy

def correct_stable_prediction(ts_logits, trigger_idx, length, label):
    
    stable_mask = make_stable_mask(ts_logits, trigger_idx, length, label)
    return jnp.any(stable_mask)

def earliest_stable_time(t, ts_logits, trigger_idx, length, label):

    stable_mask = make_stable_mask(
        ts_logits=ts_logits,
        trigger_idx=trigger_idx,
        length=length,
        label=label
    )
    stable_mask = stable_mask.at[length-1].set(True)
    t0 = jnp.nanmin(jnp.where(stable_mask, t, jnp.nan))
    t0 = (t0 - 1e-8) * 1000
    
    return t0

def obs_until_stable_time(t, ts_logits, trigger_idx, length, label):

    stable_mask = make_stable_mask(
        ts_logits=ts_logits,
        trigger_idx=trigger_idx,
        length=length,
        label=label
    )
    stable_mask = stable_mask.at[length-1].set(True)
    idx = jnp.nanargmin(jnp.where(stable_mask, t, jnp.nan))
    idx = idx - trigger_idx

    return idx

def stable_accuracy(batch_logits, trigger_indeces, lengths, labels):

    stable_correct_preds = jax.vmap(
        correct_stable_prediction
    )(batch_logits, trigger_indeces, lengths, labels)
    accuracy = jnp.sum(stable_correct_preds) / len(stable_correct_preds)

    return accuracy

def take(x, i):

    return x[i]

def onevsrest_metrics(
    label,
    batch_logits,
    trigger_indeces,
    lengths,
    labels,
):

    last_logits = jax.vmap(take)(batch_logits, lengths-1)
    pred_class = jnp.argmax(last_logits, axis=-1)

    idx_true = labels == label
    idx_pred = pred_class == label

    stable_fracs = jax.vmap(
        temporal_stable_accuracy,
        in_axes=(0,0,0,None)
    )(
        batch_logits,
        trigger_indeces,
        lengths,
        label
    )
    stable_fracs = jnp.where(
        idx_true,
        stable_fracs,
        jnp.nan
    )
    stable_frac_sum = jnp.nansum(stable_fracs)
    label_count = jnp.sum(idx_true)
    stable_frac = jnp.nanmean(stable_fracs)

    tp = jnp.sum(jnp.logical_and(idx_true, idx_pred))
    fn = jnp.sum(jnp.logical_and(idx_true, ~idx_pred))
    fp = jnp.sum(jnp.logical_and(~idx_true, idx_pred))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    stable_f1 = 3 * precision * recall * stable_frac / (
        precision * recall + precision * stable_frac + recall * stable_frac
    )

    f1 = jnp.where(jnp.isnan(f1), 0, f1)
    stable_f1 =jnp.where(jnp.isnan(stable_f1), 0, stable_f1)

    return (
        tp, fn, fp, stable_frac_sum,
        label_count, stable_frac, precision,
        recall, f1, stable_f1
    )

def make_loss_fn(weights, temporal_loss_fn, num_classes, use_path_reg=False, path_reg_lambda=0.01):
    def loss_fn(model, data, labels, keys):

        ts, trigger_indeces, lengths, metadata = data
        batch_logits, batch_reps = jax.vmap(model)(ts, metadata, keys)
        batch_logits = batch_logits[:, ::2]
        batch_reps = batch_reps[:, ::2]
        t = ts[:, ::2, 0]

        batch_cov = jnp.std(batch_reps, axis=(0,1))

        ts_losses = jax.vmap(
            temporal_loss_fn
        )(batch_logits, trigger_indeces, lengths, labels)
        batch_weights = jnp.take(weights, labels)
        batch_loss = jnp.mean(batch_weights * ts_losses)

        batch_path_reg = 0.
        if use_path_reg:
            path_regs = jax.vmap(
                temporal_path_reg,
                in_axes=(0, 0, 0, 0, None)
            )(batch_logits, trigger_indeces, lengths, batch_reps, batch_cov)
            batch_path_reg = path_reg_lambda * jnp.mean(batch_weights * path_regs)
        batch_total_loss = batch_loss + batch_path_reg

        ts_accuracy = jax.vmap(
            temporal_accuracy
        )(batch_logits, trigger_indeces, lengths, labels)
        batch_ts_accuracy = jnp.mean(ts_accuracy)

        ts_stable_accuracy = jax.vmap(
            temporal_stable_accuracy,
        )(batch_logits, trigger_indeces, lengths, labels)
        batch_ts_stable_accuracy = jnp.mean(ts_stable_accuracy)

        batch_stable_accuracy = stable_accuracy(
            batch_logits, trigger_indeces, lengths, labels
        )

        earliest_stable_t = jax.vmap(earliest_stable_time)(
            t, batch_logits, trigger_indeces,
            lengths, labels
        )
        batch_earliest_stable_t = jnp.mean(earliest_stable_t)

        num_before_stable = jax.vmap(obs_until_stable_time)(
            t, batch_logits, trigger_indeces,
            lengths, labels
        )
        batch_num_before_stable = jnp.mean(num_before_stable)

        (
            tp, fn, fp, stable_frac_sum,
            label_count, stable_frac, precision,
            recall, f1, stable_f1
        ) = jax.vmap(
                onevsrest_metrics,
                in_axes=(0, None, None, None, None)
        )(
            jnp.arange(num_classes),
            batch_logits,
            trigger_indeces,
            lengths,
            labels,
        )

        batch_macro_stable_frac = jnp.nanmean(stable_frac)
        batch_macro_f1 = jnp.nanmean(f1)
        batch_macro_stable_f1 = jnp.nanmean(stable_f1)

        aux = (
            batch_loss,
            batch_path_reg,
            batch_ts_accuracy,
            batch_ts_stable_accuracy,
            batch_stable_accuracy,
            batch_earliest_stable_t,
            batch_num_before_stable,
            batch_macro_stable_frac,
            batch_macro_f1,
            batch_macro_stable_f1,
            tp, fn, fp, stable_frac_sum, label_count
        )

        return batch_total_loss, aux
    return loss_fn

def scale_tree(scalar, tree):

    return jax.tree.map(lambda x: scalar * x, tree)

def train_step_factory(optimizer, loss_fn):
    
    @eqx.filter_jit(donate='all')
    def make_train_step(
        model,
        opt_state,
        transform_state,
        batch_data,
        batch_labels,
        key
    ):
        
        (
            times, flux, partial_ts, redshifts, 
            trigger_idx, lengths, peak_times, max_times, 
            binary_labels, multiclass_labels,
            valid_lightcurve_mask
        ) = batch_data

        flux = flux[:, 0:1]
        partial_ts = partial_ts[:, 0:1]
        redshifts = redshifts[:, 0:1]

        dt = jnp.diff(times, axis=-1)
        dt0 = jnp.zeros(dt.shape[0])[:,None]
        dt = jnp.concatenate([dt0,dt], axis=-1)
        dt = dt[:, None, :, None]
        partial_ts = jnp.concatenate([dt, partial_ts], axis=-1)
        
        _, _, interp_ts = training.batch_mapped_interpolate_timeseries(
            times, flux, partial_ts
        )

        ts = interp_ts[:, 0]
        metadata = redshifts[:, 0]
        batch_shape = ts.shape[0]
        batch_keys = jr.split(key, batch_shape)
        batch_data = (ts, trigger_idx, lengths, metadata)

        (loss_value, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, batch_data, batch_labels, batch_keys,
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        updates = scale_tree(transform_state.scale, updates)
        model = eqx.apply_updates(model, updates)
    
        return model, opt_state, transform_state, loss_value, aux

    return make_train_step
def val_step_factory(loss_fn):
    @eqx.filter_jit(donate="all-except-first")
    def make_val_step(
        model,
        opt_state,
        transform_state,
        batch_data,
        batch_labels,
        key,
    ):

        (
            times, flux, partial_ts, redshifts, 
            trigger_idx, lengths, peak_times, max_times, 
            binary_labels, multiclass_labels,
            valid_lightcurve_mask
        ) = batch_data

        flux = flux[:, 0:1]
        partial_ts = partial_ts[:, 0:1]
        redshifts = redshifts[:, 0:1]
        
        dt = jnp.diff(times, axis=-1)
        dt0 = jnp.zeros(dt.shape[0])[:,None]
        dt = jnp.concatenate([dt0,dt], axis=-1)
        dt = dt[:, None, :, None]
        partial_ts = jnp.concatenate([dt, partial_ts], axis=-1)

        _, _, interp_ts = training.batch_mapped_interpolate_timeseries(
            times, flux, partial_ts
        )

        ts = interp_ts[:, 0]
        metadata = redshifts[:, 0]
        batch_shape = ts.shape[0]
        batch_keys = jr.split(key, batch_shape)
        batch_data = (ts, trigger_idx, lengths, metadata)


        loss_value, aux = loss_fn(
            model, batch_data, batch_labels, batch_keys,
        )

        return model, opt_state, transform_state, loss_value, aux
    
    return make_val_step


def infinite_loader(loader):
    while True:
        yield from loader

def inner_loop(
    model,
    opt_state,
    transform_state,
    dataloader,
    step_fn,
    num_steps,
    key,
    num_classes,
    verbose=True
):
    
    epoch_total_loss = np.zeros(num_steps)
    epoch_temporal_loss = np.zeros(num_steps)
    epoch_path_reg = np.zeros(num_steps)
    epoch_ts_acc = np.zeros(num_steps)
    epoch_avg_stable_acc = np.zeros(num_steps)
    epoch_stable_acc = np.zeros(num_steps)
    epoch_stable_t = np.zeros(num_steps)
    epoch_num_obs = np.zeros(num_steps)
    epoch_macro_stable_acc = np.zeros(num_steps)
    epoch_macro_f1 = np.zeros(num_steps)
    epoch_macro_stable_f1 = np.zeros(num_steps)
    epoch_tp = np.zeros(num_classes)
    epoch_fn = np.zeros(num_classes)
    epoch_fp = np.zeros(num_classes)
    epoch_stable_frac_sum = np.zeros(num_classes)
    epoch_label_count = np.zeros(num_classes)

    epoch_data_time = np.zeros(num_steps)
    epoch_step_time = np.zeros(num_steps)

    t0 = time()
    for step in range(num_steps):

        step_key, key = jr.split(key, 2)
        data = next(dataloader)
        batch_data = [output.numpy() for output in data]

        (
            times, flux, partial_ts, redshifts, 
            trigger_idx, lengths, peak_times, max_times, 
            binary_labels, multiclass_labels,
            valid_lightcurve_mask
        ) = batch_data
        multiclass_labels = multiclass_labels[:, 0:1]
        batch_labels = multiclass_labels.squeeze()

        step_t0 = time()
        data_time = step_t0 - t0
        model, opt_state, transform_state, loss_value, aux = step_fn(
            model=model,
            opt_state=opt_state,
            transform_state=transform_state,
            batch_data=batch_data,
            batch_labels=batch_labels,
            key=step_key,
        )

        (
            temporal_loss, path_reg,
            avg_ts_acc, avg_stable_acc,
            stable_acc, stable_t, num_obs,
            macro_stable_acc, macro_f1,
            macro_stable_f1, tp, fn,
            fp, stable_frac_sum, label_count
        ) = aux
        
        epoch_total_loss[step] = loss_value
        epoch_temporal_loss[step] = temporal_loss
        epoch_path_reg[step] = path_reg
        epoch_ts_acc[step] = avg_ts_acc
        epoch_avg_stable_acc[step] = avg_stable_acc
        epoch_stable_t[step] = stable_t
        epoch_num_obs[step] = num_obs
        epoch_stable_acc[step] = stable_acc
        epoch_data_time[step] = data_time
        epoch_macro_stable_acc[step] = macro_stable_acc
        epoch_macro_f1[step] = macro_f1
        epoch_macro_stable_f1[step] = macro_stable_f1
        epoch_tp += tp
        epoch_fn += fn
        epoch_fp += fp
        epoch_stable_frac_sum += stable_frac_sum
        epoch_label_count += label_count

        step_t1 = time()
        step_time = step_t1 - step_t0
        epoch_step_time[step] = step_time

        if verbose:
            log_string = (
                f"Step: {step+1} | Loss: {loss_value:.2e} | " +
                f"Avg. TS Acc: {avg_ts_acc*100:.2f}% | " +
                f"Avg. Stable Acc: {avg_stable_acc*100:.2f}% | " +
                f"Stable Acc: {stable_acc*100:.2f}% | " +
                f"Stable T0: {stable_t:.2f} | " +
                f"Num. Obs.: {num_obs:.2f} | " +
                f"Data Time: {data_time / 60:.2e} min | " +
                f"Step Time: {step_time / 60:.2e} min" 
            )
            print(log_string)
    
    epoch_total_loss = np.mean(epoch_total_loss)
    epoch_temporal_loss = np.mean(epoch_temporal_loss)
    epoch_path_reg = np.mean(epoch_path_reg)
    epoch_ts_acc = np.mean(epoch_ts_acc)
    epoch_avg_stable_acc = np.mean(epoch_avg_stable_acc)
    epoch_stable_acc = np.mean(epoch_stable_acc)
    epoch_stable_t = np.mean(epoch_stable_t)
    epoch_num_obs = np.mean(epoch_num_obs)
    epoch_avg_macro_stable_acc = np.nanmean(epoch_macro_stable_acc)
    epoch_avg_macro_f1 = np.nanmean(epoch_macro_f1)
    epoch_avg_macro_stable_f1 = np.nanmean(macro_stable_f1)

    epoch_precision = epoch_tp / (epoch_tp + epoch_fp)
    epoch_recall = epoch_tp / (epoch_tp + epoch_fn)
    epoch_stable_frac = epoch_stable_frac_sum / epoch_label_count
    epoch_stable_frac = np.where(
        np.isfinite(epoch_stable_frac),
        epoch_stable_frac,
        0
    )
    epoch_f1 = 2 * epoch_precision * epoch_recall / (epoch_precision + epoch_recall)
    epoch_stable_f1 = 3 * epoch_precision * epoch_recall * epoch_stable_frac / (
        epoch_precision * epoch_recall + epoch_precision * epoch_stable_frac +
        epoch_recall * epoch_stable_frac
    )
    epoch_f1 = np.where(
        np.isfinite(epoch_f1),
        epoch_f1,
        0
    )
    epoch_stable_f1 = np.where(
        np.isfinite(epoch_stable_f1),
        epoch_stable_f1,
        0
    )
    epoch_macro_stable_frac = np.mean(epoch_stable_frac)
    epoch_macro_f1 = np.mean(epoch_f1)
    epoch_macro_stable_f1 = np.mean(epoch_stable_f1)

    epoch_data_time = np.mean(epoch_data_time)
    epoch_step_time = np.mean(epoch_step_time)
    
    metrics = [
        epoch_total_loss,
        epoch_temporal_loss,
        epoch_path_reg,
        epoch_ts_acc,
        epoch_avg_stable_acc,
        epoch_stable_acc,
        epoch_stable_t,
        epoch_num_obs,
        epoch_avg_macro_stable_acc,
        epoch_avg_macro_f1,
        epoch_avg_macro_stable_f1,
        epoch_macro_stable_frac,
        epoch_macro_f1,
        epoch_macro_stable_f1,
        epoch_data_time,
        epoch_step_time
    ]
    metrics = np.array(metrics)

    return model, opt_state, transform_state, dataloader, metrics

@eqx.filter_jit
def get_lr(opt_state, transform_state, lr_schedule):

        
    lr_schedule_step =optax.tree_utils.tree_get(opt_state, 'ScaleByScheduleState').count
    scale = optax.tree_utils.tree_get(transform_state, "scale")
    lr = lr_schedule(lr_schedule_step) * scale

    return lr

def make_epoch_string(
    name, metrics, epoch_time,
    init_lr, final_lr,
    print_times=False,
    use_path_reg=False,
    lr_scale=1.0
    
):

    (
        total_loss,
        temporal_loss,
        path_reg,
        ts_acc,
        avg_stable_acc,
        stable_acc,
        stable_t,
        num_obs,
        avg_macro_stable_acc,
        avg_macro_f1,
        avg_macro_stable_f1,
        macro_stable_acc,
        macro_f1,
        macro_stable_f1,
        data_time,
        step_time
    ) = metrics

    prefix = f"{name} - "
    if name == "Val":
        prefix = prefix + "  "
    epoch_string = (
        prefix +
        f"Total Loss: {total_loss:.2e} | "
    )
    if use_path_reg:
        epoch_string = (
            epoch_string +
            f"Temp. Loss: {temporal_loss:.2e} | " +
            f"Path Reg.: {path_reg:.2e} | "
        )

    epoch_string = (
        epoch_string +
        f"Avg. TS Acc: {ts_acc*100:.2f}% | " +
        f"Avg. Stable Acc: {avg_stable_acc*100:.2f}% | " +
        f"Stable Acc: {stable_acc*100:.2f}% | " +
        f"Stable T0: {stable_t:.2f} | " +
        f"Num. Obs.: {num_obs:.2f} | "
    )

    if name == "Val":
        epoch_string = (
            epoch_string +
            f"Macro Stable Acc: {macro_stable_acc*100:.2f}% | " +
            f"Macro F1: {macro_f1*100:.2f}% | " + 
            f"Macro S1: {macro_stable_f1*100:.2f}%"
        )
    else:
        epoch_string = (
            epoch_string +
            f"Macro Stable Acc: {avg_macro_stable_acc*100:.2f}% | " +
            f"Macro F1: {avg_macro_f1*100:.2f}% | " + 
            f"Macro S1: {avg_macro_stable_f1*100:.2f}%"
        )

    if print_times:
        epoch_string = (
            epoch_string +
            f" | Data Time: {data_time / 60:.0e} min | " +
            f"Step Time: {step_time / 60:.0e} min"
        )
    
    epoch_string = (
        epoch_string +
        f" | lr: {init_lr:.2e} - {final_lr:.2e} | " +
        f"Scale: {lr_scale:.2e} | " +
        f"Epoch Time: {epoch_time / 60:.2f} min"
    )

    return epoch_string

key = jr.PRNGKey(0)
gru_key, ncde_key, key = jr.split(key, 3)

model_config = {
    "input_size": 31,
    "hidden_size": 256,
    "num_classes": num_classes,
    "num_gru_layers": 2,
    "classifier_width": 256,
    "classifier_depth": 0,
    "online": True,
    "use_dropout": True,
    "dropout_rate": 0.2,
    "use_layernorm": True,
    "use_metadata": True,
    "metadata_size": 3,
    "init_width": 32,
    "init_depth": 0,
}

gru = GRUClassifier(
    key=gru_key,
    **model_config
)

model = gru
epochs = 1000
warmup_epochs = 50
total_epochs = epochs + warmup_epochs

early_stopping_patience = 50
use_s1_for_patience = True

reduce_lr_patience = 10
reduce_lr_cooldown = 0
reduce_lr_factor = 0.5
reduce_lr_rtol = 1e-4
reduce_lr_accum_size = 1

verbose = False
print_times = True
use_class_weights = True
use_path_reg = True
path_reg_lambda = 0.1
temporal_loss_fn = temporal_focal_cross_entropy_loss

run_name = "GRU_Path_Reg"
save_path = Path("/mimer/NOBACKUP/groups/naiss2025-22-731/jolteon_challenge/Results"+run_name)
save_path.mkdir(parents=True, exist_ok=True)

utils.save_hyperparams(save_path / "hyperparams.eqx", model_config)

gru_num_params = count_params(gru)

print(gru_num_params)

train_steps_per_epoch = len(train_dataloader)
val_steps_per_epoch = len(val_dataloader)

scaling_factor = len(train_dataloader)/train_steps_per_epoch

warmup_steps = int(scaling_factor * train_steps_per_epoch * warmup_epochs) 
total_steps = int(scaling_factor * train_steps_per_epoch * total_epochs)

print(f"Num. Steps: {total_steps} | Warmup Steps: {warmup_steps}")

learning_rate_schedule = optax.schedules.warmup_cosine_decay_schedule(
    init_value=1e-6,
    peak_value=2e-4,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=1e-6,
)
optimizer = optax.adamw(learning_rate=learning_rate_schedule)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

lr_transform = optax.contrib.reduce_on_plateau(
    patience=reduce_lr_patience,
    cooldown=reduce_lr_cooldown,
    factor=reduce_lr_factor,
    rtol=reduce_lr_rtol,
    accumulation_size=reduce_lr_accum_size
)
transform_state = lr_transform.init(eqx.filter(model, eqx.is_array))

if use_class_weights:
    loss_fn = make_loss_fn(
        weights=jnp.asarray(class_weights),
        temporal_loss_fn=temporal_loss_fn, num_classes=num_classes,
        use_path_reg=use_path_reg, path_reg_lambda=path_reg_lambda
    )
else:
    _class_weights = jnp.ones_like(class_weights)
    loss_fn = make_loss_fn(
        weights=_class_weights,
        temporal_loss_fn=temporal_loss_fn, num_classes=num_classes,
        use_path_reg=use_path_reg, path_reg_lambda=path_reg_lambda
    )

val_loss_fn = loss_fn
make_train_step = train_step_factory(optimizer, loss_fn)
make_val_step = val_step_factory(val_loss_fn)

train_loader = infinite_loader(train_dataloader)
val_loader = infinite_loader(val_dataloader)

num_metrics = 16
train_metrics = np.zeros((epochs, num_metrics))
val_metrics = np.zeros_like(train_metrics)

patience_counter = 0
best_epoch = 0
best_loss = np.inf
best_s1 = 0

for epoch in range(epochs):

    train_key, val_key, key = jr.split(key, 3)
    train_t0 = time()
    if verbose:
        print("\nTraining\n")

    lr_scale = optax.tree_utils.tree_get(transform_state, "scale")
    lr_scale = np.asarray(lr_scale)
    init_lr = get_lr(opt_state, transform_state, learning_rate_schedule)
    init_lr = np.asarray(init_lr)
    model, opt_state, transform_state, train_loader, epoch_train_metrics = inner_loop(
        model=model,
        opt_state=opt_state,
        transform_state=transform_state,
        dataloader=train_loader,
        step_fn=make_train_step,
        num_steps=train_steps_per_epoch,
        verbose=verbose,
        num_classes=num_classes,
        key=train_key,
    )
    final_lr = get_lr(opt_state, transform_state, learning_rate_schedule)
    final_lr = np.asarray(final_lr)
    train_metrics[epoch] = epoch_train_metrics
    train_time = time() - train_t0
    
    val_t0 = time()
    if verbose:
        print("\nValidation\n")

    model = eqx.nn.inference_mode(model, value=True)
    model, opt_state, transform_state, val_loader, epoch_val_metrics = inner_loop(
        model=model,
        opt_state=opt_state,
        transform_state=transform_state,
        dataloader=val_loader,
        step_fn=make_val_step,
        num_steps=val_steps_per_epoch,
        verbose=verbose,
        num_classes=num_classes,
        key=val_key
    )
    model = eqx.nn.inference_mode(model, value=False)
    val_metrics[epoch] = epoch_val_metrics
    epoch_val_loss = epoch_val_metrics[0]
    epoch_val_s1 = epoch_val_metrics[-3]


    if use_s1_for_patience:
        lr_stat = epoch_val_s1
    else:
        lr_stat = epoch_val_loss
    _, transform_state = lr_transform.update(
        updates=eqx.filter(model, eqx.is_array), state=transform_state, value=lr_stat
    )

    if verbose:
        print("\n")
    val_time = time() - val_t0

    
    new_best_loss = (
        (epoch_val_loss < best_loss) & ~use_s1_for_patience
    )
    new_best_s1 = (
        (epoch_val_s1 > best_s1) & use_s1_for_patience
    )
    new_best = new_best_loss | new_best_s1

    if new_best:
        best_loss = epoch_val_loss
        best_s1 = epoch_val_s1
        best_epoch = epoch
        patience_counter = 0
        utils.save_model(save_path / "best_model_weights.eqx", model)
        np.save(save_path / "train_metrics.npy", train_metrics)
        np.save(save_path / "val_metrics.npy", val_metrics)

    train_string = make_epoch_string(
        name="Train", metrics=epoch_train_metrics,
        epoch_time=train_time, print_times=print_times,
        use_path_reg=use_path_reg, init_lr=init_lr,
        final_lr=final_lr, lr_scale=lr_scale
    )
    val_string = make_epoch_string(
        name="Val", metrics=epoch_val_metrics,
        epoch_time=train_time, print_times=print_times,
        use_path_reg=use_path_reg, init_lr=final_lr,
        final_lr=final_lr, lr_scale=lr_scale
    )

    print(f"\n--------------- EPOCH {epoch+1} ---------------")
    print(train_string)
    print(val_string)

    patience_counter += 1
    if patience_counter >= early_stopping_patience:
        best_train_metrics = train_metrics[best_epoch]
        best_val_metrics = val_metrics[best_epoch]

        train_string = make_epoch_string("Train", best_train_metrics, train_time, use_path_reg=use_path_reg)
        val_string = make_epoch_string("Val", best_val_metrics, val_time, use_path_reg=use_path_reg)        

        if use_s1_for_patience:
            basis = "S1"
        else:
            basis = "Val Loss"

        print(f"\n-----------------------------------------------")
        print(f"{patience_counter} epochs since last improvement, stopping training.")
        print(f"Best Epoch based on {basis}: {best_epoch}")
        print(train_string)
        print(val_string)
        break
    
np.save(save_path / "train_metrics.npy", train_metrics)
np.save(save_path / "val_metrics.npy", val_metrics)
utils.save_model(save_path / "last_model_weights.eqx", model)