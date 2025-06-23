import jax
import json
import optax

import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import stronglensingncde.models as models
import stronglensingncde.training as training

from typing import Any
from tqdm import trange
from jaxtyping import Array
from collections.abc import Callable

@jax.jit
def tree_contains_inf(tree):

    isInf = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isinf(x)), tree)
    isInf = jax.tree_util.tree_reduce(jnp.add, isInf, initializer=0.)
    isInf = isInf.astype(jnp.bool) 
    
    return isInf

@jax.jit
def tree_contains_nan(tree):

    isNaN = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isinf(x)), tree)
    isNaN = jax.tree_util.tree_reduce(jnp.add, isNaN, initializer=0.)
    isNaN = isNaN.astype(jnp.bool)

    return isNaN

def inspect_gradients(
    model: eqx.Module,
    loss_fn: Callable[[eqx.Module, Any], Array],
    data: Any
) -> None:
    """
    Inspects and prints gradient statistics for each parameter of an Equinox model.

    Parameters:
    ----------
    model : eqx.Module
        The Equinox model whose gradients you want to inspect.
    loss_fn : Callable
        Loss function taking (model, data) and returning a scalar loss.
    data : Any
        Data passed to the loss function (e.g., batch input/target).

    Prints:
    ------
    For each parameter, prints:
        - Parameter name
        - Gradient mean
        - Gradient std deviation
        - Gradient norm
        - Presence of NaNs or infinite values
    """
    @eqx.filter_jit
    def func(model, data):
        (
            times, flux, partial_ts, trigger_idx,
            lengths, peak_times, max_times, 
            binary_labels, multiclass_labels,
            valid_lightcurve_mask
        ) = data

        s, interp_s, interp_ts = training.batch_mapped_interpolate_timeseries(
            times, flux, partial_ts
        )
        s = s[:,0,:]
        
        max_s = max_times #+ (lengths-1) / 1000

        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(
            model,
            times,
            s, 
            max_s,
            interp_s,
            interp_ts,
            trigger_idx,
            lengths,
            multiclass_labels,
            peak_times,
            valid_lightcurve_mask
        )

        return (loss, aux), grads
    
    (loss, aux), grads = func(model, data)

    flat_grads, grad_tree = jax.tree_util.tree_flatten(grads)
    flat_names = eqx.tree_pformat(grads).splitlines()

    print(" ")
    global_norm = optax.global_norm(grads)
    print(f"Global Gradient Norm: {global_norm:.3e}\n")
    print("Gradient inspection:")
    for grad, name in zip(flat_grads, flat_names):
        if grad is not None:
            grad_mean = jnp.mean(grad)
            grad_std = jnp.std(grad)
            grad_norm = jnp.linalg.norm(grad)
            grad_has_nan = jnp.isnan(grad).any()
            grad_has_inf = jnp.isinf(grad).any()

            print(f"\n{name.strip()}:")
            print(f"    Mean       : {grad_mean:.3e}")
            print(f"    Std        : {grad_std:.3e}")
            print(f"    Norm       : {grad_norm:.3e}")
            print(f"    Contains NaN: {bool(grad_has_nan)}")
            print(f"    Contains Inf: {bool(grad_has_inf)}")
        else:
            print(f"\n{name.strip()}: No gradient (None)")
    
    print(" ")

    return loss, aux, grads

def identity(x, *args, **kwargs):
    
    return x

orthogonal_initializer = jax.nn.initializers.orthogonal()
def orthogonal_init(weights: Array, key: jr.PRNGKey) -> Array:

    shape = weights.shape
    dtype = weights.dtype

    new_weights = orthogonal_initializer(key, shape, dtype)

    return new_weights

def init_linear_weight(model, init_fn, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [x.weight
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  weights = get_weights(model)
  new_weights = [init_fn(weight, subkey)
                 for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  return new_model

def make_model(*, key, model_class, hyperparams):
    
    model_key, init_key = jr.split(key)
    model = model_class(key=model_key, **hyperparams)

    init_fn = globals()[
        hyperparams.get(
            'weight_init_fn', 'identity'
        )
    ]
    custom_init_ncde = init_linear_weight(model.ncde, init_fn, init_key)
    custom_init_model = eqx.tree_at(lambda m: m.ncde, model, custom_init_ncde)

    return custom_init_model

def save_hyperparams(filename, hyperparams):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str).encode())

def save_model(filename, model):
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)

def load_model(model_class, hyperparams_path, weights_path):
    with open(hyperparams_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(key=jr.PRNGKey(0), model_class=model_class, hyperparams=hyperparams)
    
    with open(weights_path, "rb") as f:
        
        return eqx.tree_deserialise_leaves(f, model)

def make_lr_finder_schedule(lr_min: float, lr_max: float, num_steps: int):
    return optax.exponential_decay(
        init_value=lr_min,
        transition_steps=num_steps,
        decay_rate=lr_max / lr_min,
        staircase=False,
    )

def infinite_dataloader(dataloader):
    while True:
        for data in dataloader:
            yield data

def _lr_range_test(
    model,
    train_step,
    optimizer,
    lr_schedule,
    train_loader,
    num_steps=200,
):

    optimizer_state = optimizer.init(
        eqx.filter(model, eqx.is_inexact_array)
    )

    lrs = []
    losses = []
    for step, batch in zip(trange(num_steps), train_loader):
        

        data = [output.numpy() for output in batch]
        loss, aux, model, optimizer_state = train_step(
            model, data, optimizer_state
        )

        lr = lr_schedule(step)
        lrs.append(lr)
        losses.append(float(loss))
    return np.array(lrs), np.array(losses)

def lr_range_test(
    rng_key,
    model_hyperparams,
    loss_fn,
    train_loader,
    model_class=models.PoolingONCDEClassifier,
    num_steps=200,
    lr_min=1e-7,
    lr_max=1.0,
    repeats=1,
    optimizer="adamw"
):

    optimizer = getattr(optax, optimizer)
    train_loader = infinite_dataloader(train_loader)

    # rebuild schedule & optimizer inside in case you want different ranges
    lr_schedule = make_lr_finder_schedule(lr_min, lr_max, num_steps)
    optimizer = optimizer(learning_rate=lr_schedule)

    train_step = training.make_train_step(optimizer, loss_fn)
    
    all_lrs = np.zeros((repeats, num_steps))
    all_losses = np.zeros((repeats, num_steps))
    
    for repeat in range(repeats):
        print(f"Repeat {repeat}")

        model_key, rng_key = jr.split(rng_key)
        model = make_model(
            key=model_key,
            model_class=model_class,
            hyperparams=model_hyperparams,
        )

        lrs, losses = _lr_range_test(
            model,
            train_step,
            optimizer,
            lr_schedule,
            train_loader,
            num_steps,
        )

        all_lrs[repeat] = lrs
        all_losses[repeat] = losses

    #lrs = np.mean(lrs, axis=0)
    #losses = np.mean(losses, axis=0)

    return all_lrs, all_losses

def compute_day_snapshots(times: np.ndarray, trigger_idx: int) -> np.ndarray:
    """
    Compute the step-by-step time “snapshots” around a trigger.

    Parameters
    ----------
    times : (T,) float
        Array of sample times (in days, can be negative).
    trigger_idx : int
        Index into `times` where the trigger (e.g. time == 0) occurs.

    Returns
    -------
    snapshots : (N, D) float
        A 2D array where:
          - D = number of unique calendar‐day bins = len(np.unique(floor(times)))
          - N = total number of samples on/after the trigger day
        Each row is a copy of the running‐mean‐vector of day-times
        after including one more sample from the trigger day onward.
    """
    # -- helper to compute initial pre-trigger day-means --
    def initial_average_days(x: np.ndarray, day_trigger: int) -> np.ndarray:
        # (a) compute calendar‐day indices of x
        day_idx_local = np.floor(x).astype(int)
        # (b) list all days
        all_days = np.unique(day_idx_local)
        # (c) init output array
        x_day = np.zeros(len(all_days), dtype=float)
        # (d) fill only for days *before* the trigger
        for i, d in enumerate(all_days[all_days < day_trigger]):
            mask = (day_idx_local == d)
            x_day[i] = x[mask].mean()
        return x_day

    # 1) floor to get each sample’s calendar-day index
    day_idx = np.floor(times).astype(int)
    # 2) find which day the trigger sits in
    day_trigger = day_idx[trigger_idx]
    # 3) find all unique days and how many samples in each
    days, day_counts = np.unique(day_idx, return_counts=True)

    # 4) start history with the pre-trigger means
    history = [initial_average_days(times, day_trigger)]

    # 5) now step forward: for each day ≥ trigger, for each sample in that day
    for day, count in zip(days[days >= day_trigger],
                          day_counts[days >= day_trigger]):
        # index of this day in the full `days` array
        i_day = np.where(days == day)[0][0]
        # which sample-positions belong to this day
        idxs = np.where(day_idx == day)[0]

        for j in range(count):
            snapshot = np.copy(history[-1])
            # running‐mean of the day-times up to the j-th sample in this day
            running_mean = np.mean(times[idxs[: j + 1]])
            snapshot[i_day] = running_mean
            history.append(snapshot)

    # drop the very first “pre-trigger only” entry
    return np.asarray(history[1:])