import jax

import equinox as eqx
import jax.numpy as jnp

from jaxtyping import ArrayLike
from collections.abc import Callable
from jax.scipy.special import logsumexp

def temporal_cross_entropy_loss(logits, label, scale=1.0, **kwargs):

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_prob_true = log_probs[:, label]
    loss = -scale * log_prob_true

    return loss

def focalize_loss_fn(loss_fn: Callable, gamma: float = 1.0, **kwargs) -> Callable:

    def new_loss_fn(logits, label, scale=1.0, **kwargs):

        probs = jax.nn.softmax(logits, axis=-1)
        prob_true = probs[:, label]
        scaling_factor = (1 - prob_true)**gamma
        unscaled_loss = loss_fn(logits, label, scale, **kwargs)
        loss = scaling_factor * unscaled_loss

        return loss
    
    return new_loss_fn

def dual_focalize_loss_fn(loss_fn: Callable, gamma: float = 1.0, **kwargs) -> Callable:

    def new_loss_fn(logits, label, scale=1.0, **kwargs):

        probs = jax.nn.softmax(logits, axis=-1)
        masked_probs = probs.at[:, label].set(-1.0)
        
        p_true = probs[:, label]
        p_next = jnp.max(masked_probs, axis=-1)
        scaling_factor = (1 - p_true + p_next)**gamma
        
        unscaled_loss = loss_fn(logits, label, scale, **kwargs)
        loss = scaling_factor * unscaled_loss

        return loss
    
    return new_loss_fn

def class_weight_loss_fn(loss_fn: Callable, class_weights: ArrayLike, **kwargs) -> Callable:

    def new_loss_fn(logits, label, scale=1.0, **kwargs):

        class_weight = class_weights[label]
        loss = class_weight * loss_fn(logits, label, scale, **kwargs)

        return loss
    
    return new_loss_fn

def temporal_hinge_loss(
    logits, label,
    hinge_class_frequencies,
    hinge_margin=0.0, 
    scale=1.0, **kwargs
):

    # logits: array of shape (T, N)
    T, N = logits.shape

    # 1) find the hard prediction at each epoch
    #    (argsort of -logits gives descending order)
    sorted_preds = jnp.argsort(-logits, axis=-1)   # shape (T, N)
    preds        = sorted_preds[:, 0]              # shape (T,)

    # 2) build a time index [0,1,…,T-1]
    t = jnp.arange(T)

    # 3) logits_prev: for t=1..T-1, grab logits[t, preds[t-1]]
    logits_prev = logits[t[1:], preds[:-1]]        # shape (T-1,)

    # 4) to get max_excl_prev, mask out preds[t-1] at each t
    prev_preds   = preds[:-1]                      # shape (T-1,)
    logits_next  = logits[1:]                      # shape (T-1, N)
    mask         = jax.nn.one_hot(prev_preds, N, dtype=bool)  # True where j == preds[t-1]

    # 5) compute per‐row minima
    row_min = jnp.min(logits_next, axis=1, keepdims=True)   # (T-1, 1)

    # 6) replacement value = one less than the row minimum
    replacement = row_min - 1.0                             # (T-1, 1)

    # 7) mask‐and‐replace
    #    broadcasting replacement across the N columns
    masked_logits = jnp.where(mask, replacement, logits_next)  # (T-1, N)

    # then take the max over axis=1
    max_excl_prev = jnp.max(masked_logits, axis=1)

    # 8) compute the hinge loss

    # Update hinge_scale based on the frequency of the label
    alpha = scale / hinge_class_frequencies[label]
    

    loss = alpha * jnp.maximum(0., max_excl_prev - logits_prev + hinge_margin)
    loss = jnp.concatenate([jnp.zeros((1,)), loss])  # shape (T,)

    return loss

def temporal_cumulative_cross_entropy_loss(logits, label, scale, **kwargs):

    counts = jnp.arange(1, logits.shape[0]+1)[:, None]
    cumulative_avg_logits = jnp.cumsum(logits, axis=0) / counts
    loss = temporal_cross_entropy_loss(cumulative_avg_logits, label, scale, **kwargs)

    return loss

def temporal_predictions(
    logits, label, **kwargs
):
    
    pred_check = jnp.argmax(logits, axis=-1) == label

    return pred_check

def earliest_correct_prediction_time(logits, label, times, length, trigger_idx):

    pred_class = jnp.argmax(logits, axis=-1)

    indeces = jnp.arange(times.shape[0])
    length_mask = indeces < length
    trigger_mask = indeces >= trigger_idx
    valid_mask = length_mask & trigger_mask
    correct_mask = pred_class == label
    correct_label_mask = valid_mask & correct_mask
    t_max = times[length - 1]

    pred_times = jnp.where(correct_label_mask, times, t_max)
    earliest_pred_time = jnp.nanmin(pred_times)

    return earliest_pred_time * 1000.0

def earliest_stable_correct_prediction_time(logits, label, times, length, trigger_idx):

    pred_class = jnp.argmax(logits, axis=-1)

    indeces = jnp.arange(times.shape[0])
    length_mask = indeces < length
    trigger_mask = indeces >= trigger_idx
    valid_mask = length_mask & trigger_mask
    correct_mask = pred_class == label
    t_max = times[length - 1]

    rev_cum = jnp.cumprod(correct_mask[::-1])               # (T,)
    consecutive_correct_mask = rev_cum[::-1].astype(bool)

    stable_mask = jnp.logical_and(valid_mask, consecutive_correct_mask)
    stable_pred_times = jnp.where(stable_mask, times, t_max)
    earliest_stable_pred_time = jnp.nanmin(stable_pred_times)

    return earliest_stable_pred_time * 1000.0

def stable_correct_prediction(logits, label, length, trigger_idx):

    pred_class = jnp.argmax(logits, axis=-1)

    indeces = jnp.arange(logits.shape[0])
    length_mask = indeces < length
    trigger_mask = indeces >= trigger_idx
    valid_mask = length_mask & trigger_mask
    correct_mask = pred_class == label

    rev_cum = jnp.cumprod(correct_mask[::-1])               # (T,)
    consecutive_correct_mask = rev_cum[::-1].astype(bool)

    stable_mask = jnp.logical_and(valid_mask, consecutive_correct_mask)
    
    return jnp.any(stable_mask)

def number_of_transitions(logits, length, trigger_idx):
    transitions = jnp.abs(jnp.diff(jnp.argmax(logits, axis=-1)))
    transitions = jnp.concatenate([transitions, jnp.array([0])])
    transitions = transitions > 0

    indeces = jnp.arange(logits.shape[0])
    length_mask = indeces < length
    trigger_mask = indeces >= trigger_idx
    valid_mask = length_mask & trigger_mask

    num_transitions = jnp.nansum(
        jnp.where(valid_mask, transitions, jnp.nan)
    )

    return num_transitions

def transition_rate(logits, length, trigger_idx):

    transitions = jnp.abs(jnp.diff(jnp.argmax(logits, axis=-1)))
    transitions = jnp.concatenate([transitions, jnp.array([0])])
    transitions = transitions > 0

    indeces = jnp.arange(logits.shape[0])
    length_mask = indeces < length
    trigger_mask = indeces >= trigger_idx
    valid_mask = length_mask & trigger_mask

    rate = jnp.nanmean(
        jnp.where(valid_mask, transitions, jnp.nan)
    )

    return rate

image_mapped_earliest_correct_prediction_time = jax.vmap(
    earliest_correct_prediction_time,
    in_axes=(0, 0, None, None, None)
)
batch_mapped_earliest_correct_prediction_time = jax.vmap(
    image_mapped_earliest_correct_prediction_time,
    in_axes=(0, 0, 0, 0, 0)
)

image_mapped_earliest_stable_correct_prediction_time = jax.vmap(
    earliest_stable_correct_prediction_time,
    in_axes=(0, 0, None, None, None)
)
batch_mapped_earliest_stable_correct_prediction_time = jax.vmap(
    image_mapped_earliest_stable_correct_prediction_time,
    in_axes=(0, 0, 0, 0, 0)
)

image_mapped_stable_correct_prediction = jax.vmap(
    stable_correct_prediction,
    in_axes=(0, 0, None, None)
)
batch_mapped_stable_correct_prediction = jax.vmap(
    image_mapped_stable_correct_prediction,
    in_axes=(0, 0, 0, 0)
)

image_mapped_transition_rate = jax.vmap(transition_rate, in_axes=(0, None, None))
batch_mapped_transition_rate = jax.vmap(image_mapped_transition_rate, in_axes=(0, 0, 0))

image_mapped_number_of_transitions = jax.vmap(
    number_of_transitions,
    in_axes=(0, None, None)
)
batch_mapped_number_of_transitions = jax.vmap(
    image_mapped_number_of_transitions,
    in_axes=(0, 0, 0)
)

@jax.jit
def batch_median_earliest_correct_prediction_time(
    logits, label, times, length, trigger_idx, valid_lightcurve_mask
):
    
    pred_times = batch_mapped_earliest_correct_prediction_time(
        logits,
        label,
        times,
        length,
        trigger_idx
    )

    batch_earliest_time = jnp.nanmedian(
        jnp.where(valid_lightcurve_mask, pred_times, jnp.nan),
    )

    return batch_earliest_time

@jax.jit
def batch_median_earliest_stable_correct_prediction_time(
    logits, label, times, length, trigger_idx, valid_lightcurve_mask
):
    pred_times = batch_mapped_earliest_stable_correct_prediction_time(
        logits,
        label,
        times,
        length,
        trigger_idx
    )

    batch_earliest_time = jnp.nanmedian(
        jnp.where(valid_lightcurve_mask, pred_times, jnp.nan),
    )

    return batch_earliest_time

@jax.jit
def batch_stable_accuracy(
    logits, label, length, trigger_idx, valid_lightcurve_mask
):
    stable_preds = batch_mapped_stable_correct_prediction(
        logits,
        label,
        length,
        trigger_idx
    )

    batch_stable_accuracy = jnp.nanmean(
        jnp.where(valid_lightcurve_mask, stable_preds, jnp.nan),
    )

    return batch_stable_accuracy

@jax.jit
def batch_transition_rate(logits, lengths, trigger_idx, valid_lightcurve_mask):

    rates = batch_mapped_transition_rate(
        logits,
        lengths,
        trigger_idx
    )
    masked_rates = jnp.where(valid_lightcurve_mask, rates, jnp.nan)
    return jnp.nanmean(masked_rates)

@jax.jit
def batch_number_of_transitions(logits, lengths, trigger_idx, valid_lightcurve_mask):

    num_transitions = batch_mapped_number_of_transitions(
        logits,
        lengths,
        trigger_idx
    )
    masked_num_transitions = jnp.where(valid_lightcurve_mask, num_transitions, jnp.nan)

    return jnp.nanmedian(masked_num_transitions)

def unit_weight_fn(times, t_peak, logits, label, **kwargs):
    return jnp.ones_like(times)

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

def focal_weight_fn(times, t_peak, logits, label, gamma=1.0, eps=1e-20, **kwargs):

    
    probs = jax.nn.softmax(logits, axis=-1)
    p_true = jnp.clip(probs[:, label], min=eps, max=1.0-eps)
    focal_weight = _stable_focal_weight(p_true, gamma, eps)

    return focal_weight

def dual_focal_weight_fn(times, t_peak, logits, label, gamma=1.0, **kwargs):

    probs = jax.nn.softmax(logits, axis=-1)
    masked_probs = probs.at[:, label].set(-1.0)
    
    p_true = probs[:, label]
    p_next = jnp.max(masked_probs, axis=-1)
    p_diff = jnp.clip(p_true - p_next, min=0., max=1.0)
    focal_weight = _stable_focal_weight(p_diff, gamma)

    return focal_weight

def weight_warmup_wrapper(weight_fn, num_warmup):

    _unit_weight_fn = lambda operand: unit_weight_fn(
        operand[0], operand[1], operand[2],
        operand[3], **operand[4]
    )

    _weight_fn = lambda operand: weight_fn(
        operand[0], operand[1], operand[2],
        operand[3], **operand[4]
    )

    def weight_fn(step, operands):

        weights = jax.lax.cond(
            step < num_warmup,
            _unit_weight_fn,
            _weight_fn,
            operands
        )

        return weights

def make_masked_timeseries_loss_fn(
    loss_fn: callable,
    loss_fn_kwargs: dict,
    temporal_weight_fn: callable,
    temporal_weight_kwargs: dict,
    
):
    """
    Args:
        loss_fn (callable): Loss function to use, takes logits and label of
                            a single epoch and returns a float.
        loss_fn_kwargs (dict): Additional arguments for the loss function.
        temporal_weight_fn (callable): Function to compute temporal weights, must
                                       take times, t_peak and kwargs.
        temporal_weight_kwargs (dict): Additional arguments for the temporal

    Returns:
        float: Computed loss.
    """

    def masked_loss_fn(
        logits: jnp.ndarray,
        label: int,
        trigger_idx: int,
        length: int,
        times: jnp.ndarray,
        t_peak: float,
    ):
        """
        Args:
            logits (jnp.ndarray): Model logits of shape (max_length, num_classes).
            label (int): Timeseries label.
            trigger_idx (int): Index of the trigger time.
            length (int): Length of the unpadded time series.
            times (jnp.ndarray): Times of the time series, input to temporal_weight_fn.
            t_peak (float): Peak time of the time series, input to temporal_weight_fn.
        """

        _loss = loss_fn(logits, label, **loss_fn_kwargs)
        _weights = temporal_weight_fn(times, t_peak, logits, label, **temporal_weight_kwargs)

        indeces = jnp.arange(_loss.shape[-1])
        length_mask = indeces < length
        trigger_mask = indeces >= trigger_idx
        mask = jnp.logical_and(length_mask, trigger_mask)

        loss = jnp.where(
            length_mask,
            _loss,
            jax.lax.stop_gradient(_loss)
        )
        weights = jnp.where(
            length_mask,
            _weights,
            jax.lax.stop_gradient(_weights)
        )
        
        weighted_loss = loss * weights
        ts_loss = jnp.sum(weighted_loss, axis=-1, where=mask) / jnp.sum(weights, axis=-1, where=mask)

        return ts_loss

    return masked_loss_fn

def make_batch_loss_fn(
    base_loss_fn: callable,
    loss_fn_kwargs: dict,
    temporal_weight_fn: callable,
    temporal_weight_kwargs: dict,
):
    
    ts_loss_fn = make_masked_timeseries_loss_fn(
        base_loss_fn,
        loss_fn_kwargs,
        temporal_weight_fn,
        temporal_weight_kwargs
    )

    def batch_loss_fn(
        logits,
        labels,
        trigger_indices,
        lengths,
        times,
        peak_times,
        valid_lightcurve_mask,
    ):
        
        image_mapped_ts_loss_fn = jax.vmap(
            ts_loss_fn,
            in_axes=(0, 0, None, None, None, 0)
        )

        batch_mapped_ts_loss_fn = jax.vmap(
            image_mapped_ts_loss_fn,
            in_axes=(0, 0, 0, 0, 0, 0)
        )

        batch_image_ts_loss = batch_mapped_ts_loss_fn(
            logits,
            labels,
            trigger_indices,
            lengths,
            times,
            peak_times
        )   # (N_batch, N_max_img)

        avg_batch_loss = jnp.sum(
            batch_image_ts_loss, where=valid_lightcurve_mask
        ) / jnp.sum(valid_lightcurve_mask)

        return avg_batch_loss
    
    return batch_loss_fn

def make_function_list(components: list[Callable | str]):

    if not isinstance(components, list):
        components = [components,]
    new_list = []
    for component in components:
        if isinstance(component, str):
            component = globals()[component]
        if not isinstance(component, Callable):
            raise ValueError(f"Components must be str or Callable, got {type(component)}")
        new_list.append(component)
    
    return new_list

def compose_loss_components(
    loss_components: list[Callable],
    loss_modifiers: list[Callable],
    loss_modifier_kwargs: dict,
) -> list[Callable]:
    """
    For each loss component, apply each modifier in sequence (threading the output
    of one into the next), and return the list of fully-wrapped loss functions.
    """
    wrapped_components = []
    for comp in loss_components:
        # start with the original component
        fn = comp
        # apply each modifier, passing in the same kwargs
        for mod in loss_modifiers:
            fn = mod(fn, **loss_modifier_kwargs)
        wrapped_components.append(fn)
    return wrapped_components

def make_loss_and_metric_fn(
    loss_components: list[Callable | str],
    loss_fn_kwargs: dict,
    temporal_weight_fns: list[Callable | str],
    temporal_weight_fn_kwargs: dict,
    loss_scales: list[float] = None,
    loss_modifiers: list[Callable | str] = None,
    loss_modifier_kwargs: dict = {},
    metric_fns: list[Callable | str] = [temporal_predictions,],
    metric_fn_kwargs: dict = {},
):

    if type(loss_components) != list:
        loss_components = [loss_components]

    if loss_scales is None:
        loss_scales = [1.0] * len(loss_components)

    if loss_modifiers is None:
        unity_loss_modifier = lambda fn, **kwargs: fn
        loss_modifiers = [unity_loss_modifier, ] 

    if type(temporal_weight_fns) != list:
        temporal_weight_fns = [temporal_weight_fns] * len(loss_components)

    if type(metric_fns) != list:
        metric_fns = [metric_fns]

    loss_modifier_kwargs = {
        key: (jnp.array(value) if isinstance(value, list) else value)
        for key, value in loss_modifier_kwargs.items()
    }

    loss_components = make_function_list(loss_components)
    loss_modifiers = make_function_list(loss_modifiers)
    temporal_weight_fns = make_function_list(temporal_weight_fns)
    metric_fns = make_function_list(metric_fns)

    loss_components = compose_loss_components(
        loss_components=loss_components,
        loss_modifiers=loss_modifiers,
        loss_modifier_kwargs=loss_modifier_kwargs
    )

    batch_loss_fns = [
        make_batch_loss_fn(
            base_loss_fn,
            loss_fn_kwargs | {'scale': loss_scale},
            temporal_weight_fn,
            temporal_weight_fn_kwargs
        ) for base_loss_fn, loss_scale, temporal_weight_fn in 
        zip(loss_components, loss_scales, temporal_weight_fns)
    ]

    batch_metric_fns = [
        make_batch_loss_fn(
            metric_fn,
            metric_fn_kwargs,
            unit_weight_fn,
            {}
        ) for metric_fn in metric_fns
    ]

    def loss_fn(
        model,
        times: jnp.ndarray,
        s: jnp.ndarray,
        max_s: jnp.ndarray,
        interp_s: jnp.ndarray,
        interp_ts: jnp.ndarray,
        trigger_indices: jnp.ndarray,
        lengths: jnp.ndarray,
        labels: jnp.ndarray,
        peak_times: jnp.ndarray,
        valid_lightcurve_mask: jnp.ndarray,
        step: jnp.ndarray,
    ):
        
        logits = jax.vmap(
            model, in_axes=(0, 0, 0, 0, 0)
        )(s, interp_s, interp_ts, max_s, valid_lightcurve_mask)    # (N_batch, N_max_img, max_length, num_classes)
        
        losses = jnp.array(
            [
                _loss_fn(
                    logits,
                    labels,
                    trigger_indices,
                    lengths,
                    times,
                    peak_times,
                    valid_lightcurve_mask
                ) for _loss_fn in batch_loss_fns
            ]
        )
        loss = jnp.sum(losses)

        metrics = jnp.array(
            [
                _metric_fn(
                    logits,
                    labels,
                    trigger_indices,
                    lengths,
                    times,
                    peak_times,
                    valid_lightcurve_mask
                ) for _metric_fn in batch_metric_fns
            ]
        )

        median_earliest_time = batch_median_earliest_correct_prediction_time(
            logits,
            labels,
            times,
            lengths,
            trigger_indices,
            valid_lightcurve_mask
        )
        median_earliest_stable_time = batch_median_earliest_stable_correct_prediction_time(
            logits,
            labels,
            times,
            lengths,
            trigger_indices,
            valid_lightcurve_mask
        )
        stable_accuracy = batch_stable_accuracy(
            logits,
            labels,
            lengths,
            trigger_indices,
            valid_lightcurve_mask
        )

        transition_rate = batch_transition_rate(
            logits,
            lengths,
            trigger_indices,
            valid_lightcurve_mask
        )

        num_transitions = batch_number_of_transitions(
            logits,
            lengths,
            trigger_indices,
            valid_lightcurve_mask
        )

        metrics = jnp.concatenate(
            [
                stable_accuracy[None],
                median_earliest_time[None],
                median_earliest_stable_time[None],
                transition_rate[None],
                num_transitions[None],
                metrics
            ]
        )

        return loss, (losses, metrics)
    
    return loss_fn