import jax
jax.config.update("jax_enable_x64", True)
import time
import utils
import diffrax

import numpy as np
import equinox as eqx  
import jax.numpy as jnp

from pathlib import Path
from optax import tree_utils as otu

# times (N_batch, max_length)
# flux (N_batch, N_max_imgs, max_length, N_bands)
# partial_ts (N_batch, N_max_imgs, max_length, N_features)
# trigger_idx (N_batch,)
# lengths (N_batch, )
# binary_labels (N_batch, )
# multiclass_labels (N_batch, N_max_imgs)

def _interpolate_timeseries(times, flux, partial_ts):

    times_rect, flux_rect = diffrax.rectilinear_interpolation(
        times, flux, replace_nans_at_start=0.
    )
    colors = -jnp.diff(flux_rect, axis=-1)
    
    _, partial_rect = diffrax.rectilinear_interpolation(
        times, partial_ts, replace_nans_at_start=0.
    )

    s_rect = times_rect
    n_i = times_rect.shape[0]
    n_j = (n_i+1) / 2
    i_indeces = jnp.arange(n_i)
    j_indeces = jnp.arange(n_j)
    r_mask = (i_indeces % 2).astype(bool)
    indeces, _ = diffrax.rectilinear_interpolation(
        j_indeces, j_indeces
    )
    indeces = jnp.where(r_mask, indeces-1, indeces) / 1000
    s_rect = s_rect + indeces
    s = s_rect[::2]

    obs_rect = jnp.concatenate([s_rect[:, None], flux_rect, colors, partial_rect], axis=-1)

    return s, s_rect, obs_rect

image_mapped_interpolate_timeseries = jax.vmap(
    _interpolate_timeseries,
    in_axes=(None, 0, 0)
)
batch_mapped_interpolate_timeseries = jax.vmap(
    image_mapped_interpolate_timeseries,
    in_axes=(0, 0, 0)
)
interpolate_timeseries = eqx.filter_jit(batch_mapped_interpolate_timeseries)

def make_train_step(optimizer, loss_fn):

    gradient_and_loss_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit(donate="all")
    def train_step(model, data, optimizer_state):

        (
            times, flux, partial_ts, trigger_idx,
            lengths, peak_times, max_times, 
            binary_labels, multiclass_labels,
            valid_lightcurve_mask
        ) = data

        s, interp_s, interp_ts = batch_mapped_interpolate_timeseries(
            times, flux, partial_ts
        )
        s = s[:,0,:]
        
        max_s = max_times + (lengths-1) / 1000

        (loss, aux), gradients = gradient_and_loss_fn(
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

        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

        return loss, aux, model, optimizer_state
    
    return train_step

def make_val_step(loss_fn):
    
    @eqx.filter_jit(donate="all-except-first")
    def val_step(model, data, optimizer_state):
        
        (
            times, flux, partial_ts, trigger_idx,
            lengths, peak_times, max_times, 
            binary_labels, multiclass_labels,
            valid_lightcurve_mask
        ) = data

        s, interp_s, interp_ts = batch_mapped_interpolate_timeseries(
            times, flux, partial_ts
        )
        s = s[:,0,:]

        max_s = max_times + (lengths-1) / 1000

        loss, aux = loss_fn(
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
        
        return loss, aux, model, optimizer_state
    
    return val_step

def inner_loop(
    model,
    dataloader,
    optimizer_state,
    step_fn,
    fixed_lr = None,
    number_of_steps = None,
    verbose = False,
    meta_dataloader=None,
):
    
    if not number_of_steps:
        number_of_steps = len(dataloader)

    total_loss = 0
    
    losses = []
    metrics = []
    aux_vals = []
    lrs = []

    t_init = time.time()
    for step in range(number_of_steps):
        
        if fixed_lr is None:
            current_lr = optimizer_state[1].hyperparams['learning_rate']
        else:
            current_lr = fixed_lr
            
        data = next(dataloader)
        data = [output.numpy() for output in data]
        t_step_init = time.time()
        loss, aux, model, optimizer_state = step_fn(
            model, data, optimizer_state
        )
        step_duration = time.time() - t_step_init
        
        total_loss += loss
        step_losses = aux[0]
        step_metrics = aux[1]
        losses.append(step_losses)
        metrics.append(step_metrics)
        lrs.append(current_lr)
        
        # TODO: Replace with function that handles aux
        if verbose:

            stable_accuracy = step_metrics[0]
            earliest_time = step_metrics[1]
            stable_earliest_time = step_metrics[2]
            transition_rate = step_metrics[3]
            num_transitions = step_metrics[4]
            ts_accuracy = step_metrics[5]

            step_string = (
                f"Step: {step} | Loss: {loss:.2e} | " +
                f"Stable Acc.: {stable_accuracy*100:.2f}% | " +
                f"T_0: {earliest_time:.2f} | " +
                f"Stable T_0: {stable_earliest_time:.2f} | " +
                f"Flip Rate: {transition_rate*100:.2f}% | " +
                f"N Flips: {num_transitions} | " +
                f"TS Acc.: {ts_accuracy*100:.2f}% | " +
                f"LR: {current_lr:.6e} | " +
                f"Step Dur.: {step_duration / 60:.2f} min"
            )

            print(
                step_string
            )
        
        aux_vals.append(aux)
    
    total_time = time.time() - t_init
    avg_step_time = total_time / number_of_steps
    avg_loss = total_loss / number_of_steps
    
    losses = np.array(losses)
    metrics = np.array(metrics)
    lrs = np.array(lrs)

    avg_metrics = np.mean(np.array(metrics), axis=0)
    avg_losses = np.mean(np.array(losses), axis=0)
    init_lr, final_lr = lrs[0], lrs[-1]
    
    aux_vals = (
        avg_losses, avg_metrics, avg_step_time,
        losses, (init_lr, final_lr), metrics
    )

    if verbose:
        print("\n")

    return avg_loss, model, optimizer_state, aux_vals

def training_loop(
    model,
    loss_fn,
    num_loss_components: int,
    optimizer,
    train_dataloader,
    val_dataloader,
    meta_dataloader=None,
    number_of_epochs: int = 1,
    steps_per_epoch: int = 10,
    verbose_steps: bool = False,
    verbose_epochs: bool = True,
    save_path: str = None,
    patience: int = 10,
):
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    optimizer_state = optimizer.init(
        eqx.filter(model, eqx.is_inexact_array)
    )

    print(optimizer_state)

    train_step = make_train_step(optimizer, loss_fn)
    val_step = make_val_step(loss_fn)

    def infinite_dataloader(dataloader):
        while True:
            for data in dataloader:
                yield data

    if verbose_epochs:
        num_batches_in_dataset = len(train_dataloader)
        num_full_passes = number_of_epochs * steps_per_epoch / num_batches_in_dataset
        init_str = (
            f"\nNum. Full Passes: {num_full_passes:.2f} | "+
            f"Num. Epochs: {number_of_epochs} | " +
            f"Batches per Epoch: {steps_per_epoch}\n"
        )
        print(init_str)

    train_dataloader = infinite_dataloader(train_dataloader)
    val_dataloader = infinite_dataloader(val_dataloader)
    if meta_dataloader:
        meta_dataloader = infinite_dataloader(meta_dataloader)

    training_epoch_losses = np.zeros((number_of_epochs, num_loss_components))
    training_epoch_metrics = np.zeros((number_of_epochs, 6))

    val_epoch_losses = np.zeros((number_of_epochs, num_loss_components))
    val_epoch_metrics = np.zeros((number_of_epochs, 6))

    best_val_loss = -np.inf
    best_val_epoch = 0
    wait = 0

    t_training_init = time.time()
    for epoch in range(1, number_of_epochs+1):
        
        if verbose_steps:
            print("Training")

        t_epoch_init = time.time()
        (
            avg_train_loss, model,
            optimizer_state, train_aux
        ) = inner_loop(
            model=model,
            dataloader=train_dataloader,
            optimizer_state=optimizer_state,
            step_fn=train_step,
            number_of_steps=steps_per_epoch,
            verbose=verbose_steps
        )
        last_train_lr = train_aux[4][-1]

        if verbose_steps:
            print("\nValidation")
        (
            avg_val_loss, model, _, val_aux
        ) = inner_loop(
            model=model,
            dataloader=val_dataloader,
            optimizer_state=None,
            fixed_lr=last_train_lr,
            step_fn=val_step,
            number_of_steps=steps_per_epoch,
            verbose=verbose_steps
        )

        epoch_duration = time.time() - t_epoch_init

        if verbose_epochs:
            
            avg_train_metrics = train_aux[1]
            avg_train_step_time = train_aux[2]
            train_stable_accuracy = avg_train_metrics[0]
            train_earliest_time = avg_train_metrics[1]
            train_stable_earliest_time = avg_train_metrics[2]
            train_transition_rate = avg_train_metrics[3]
            train_num_transitions = avg_train_metrics[4]
            train_ts_accuracy = avg_train_metrics[5]
            train_init_lr, train_final_lr = train_aux[4]

            train_string = (
                f"Train - " +
                f"Loss: {avg_train_loss:.2e} | " +
                f"Stable Acc.: {train_stable_accuracy*100:.2f}% | " +
                f"T_0: {train_earliest_time:.2f} | " +
                f"Stable T_0: {train_stable_earliest_time:.2f} | " +
                f"Flip Rate: {train_transition_rate*100:.2f}% | " + 
                f"N Flips: {train_num_transitions} | " +
                f"TS Acc.: {train_ts_accuracy*100:.2f}% | " +
                f"LR Range: {train_init_lr:.2e} - {train_final_lr:.2e} | " +
                f"Step Dur.: {avg_train_step_time / 60:.2f} min"
            )

            avg_val_metrics = val_aux[1]
            avg_val_step_time = val_aux[2]
            val_stable_accuracy = avg_val_metrics[0]
            val_earliest_time = avg_val_metrics[1]
            val_stable_earliest_time = avg_val_metrics[2]
            val_transition_rate = avg_val_metrics[3]
            val_num_transitions = avg_val_metrics[4]
            val_ts_accuracy = avg_val_metrics[5]
            val_init_lr, val_final_lr = val_aux[4]

            val_string = (
                f"Val   - " +
                f"Loss: {avg_val_loss:.2e} | " +
                f"Stable Acc.: {val_stable_accuracy*100:.2f}% | " +
                f"T_0: {val_earliest_time:.2f} | " +
                f"Stable T_0: {val_stable_earliest_time:.2f} | " +
                f"Flip Rate: {val_transition_rate*100:.2f}% | " +
                f"N Flips: {val_num_transitions} | " +
                f"TS Acc.: {val_ts_accuracy*100:.2f}% | " +
                f"LR Range: {val_init_lr:.2e} - {val_final_lr:.2e} | " +
                f"Step Dur.: {avg_val_step_time / 60:.2f} min\n"
            )

            epoch_string = f"\nEpoch: {epoch} | Epoch Dur.: {epoch_duration / 60:.2f} min"
            
            print(epoch_string)
            print(train_string)
            print(val_string)
        
        training_epoch_losses[epoch-1] = train_aux[0]
        training_epoch_metrics[epoch-1] = avg_train_metrics
        val_epoch_losses[epoch-1] = val_aux[0]
        val_epoch_metrics[epoch-1] = avg_val_metrics

        if avg_val_loss > best_val_loss:
            
            if save_path:
                utils.save_model(save_path / "best_model.eqx", model)
                np.save(save_path / "train_losses.npy", training_epoch_losses)
                np.save(save_path / "train_metrics.npy", training_epoch_metrics)
                np.save(save_path / "val_losses.npy", val_epoch_losses)
                np.save(save_path / "val_metrics.npy", val_epoch_metrics)
            
            best_val_epoch = epoch-1
            best_val_loss = avg_val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\nStopping early at epoch {epoch} (no improvement in {patience} epochs)\n")
                break

            

    if save_path:
        utils.save_model(save_path / "final_model.eqx", model)
        np.save(save_path / "train_losses.npy", training_epoch_losses)
        np.save(save_path / "train_metrics.npy", training_epoch_metrics)
        np.save(save_path / "val_losses.npy", val_epoch_losses)
        np.save(save_path / "val_metrics.npy", val_epoch_metrics)

    training_duration = time.time() - t_training_init
    avg_epoch_duration = training_duration / number_of_epochs

    train_stable_accuracy = training_epoch_metrics[best_val_epoch, 0]
    train_earliest_time = training_epoch_metrics[best_val_epoch, 1]
    train_stable_earliest_time = training_epoch_metrics[best_val_epoch, 2]
    train_transition_rate = training_epoch_metrics[best_val_epoch, 3]
    train_num_transitions = training_epoch_metrics[best_val_epoch, 4]
    train_ts_accuracy = training_epoch_metrics[best_val_epoch, 5]

    best_train_string = (
        f"\nTrain - " +
        f"Loss: {avg_train_loss:.2e} | " +
        f"Stable Acc.: {train_stable_accuracy*100:.2f}% | " +
        f"T_0: {train_earliest_time:.2f} | " +
        f"Stable T_0: {train_stable_earliest_time:.2f} | " +
        f"Flip Rate: {train_transition_rate*100:.2f}% | " + 
        f"N Flips: {train_num_transitions} | " +
        f"TS Acc.: {train_ts_accuracy*100:.2f}%" 
    )

    val_stable_accuracy = val_epoch_metrics[best_val_epoch, 0]
    val_earliest_time = val_epoch_metrics[best_val_epoch, 1]
    val_stable_earliest_time = val_epoch_metrics[best_val_epoch, 2]
    val_transition_rate = val_epoch_metrics[best_val_epoch, 3]
    val_num_transitions = val_epoch_metrics[best_val_epoch, 4]
    val_ts_accuracy = val_epoch_metrics[best_val_epoch, 5]

    best_val_string = (
        f"Val   - " +
        f"Loss: {avg_val_loss:.2e} | " +
        f"Stable Acc.: {val_stable_accuracy*100:.2f}% | " +
        f"T_0: {val_earliest_time:.2f} | " +
        f"Stable T_0: {val_stable_earliest_time:.2f} | " +
        f"Flip Rate: {val_transition_rate*100:.2f}% | " +
        f"N Flips: {val_num_transitions} | " +
        f"TS Acc.: {val_ts_accuracy*100:.2f}%\n"
    )

    training_string = (
        f"\nTraining Time: {training_duration / 60:.2f} min | " +
        f"Avg. Epoch Time: {avg_epoch_duration / 60:.2f} | " +
        f"Best Epoch: {best_val_epoch}\n" 
    )
    print(training_string)
    print("Best Val Epoch Metrics:")
    print(best_train_string)
    print(best_val_string)

    train_log = (training_epoch_losses, training_epoch_metrics)
    val_log = (val_epoch_losses, val_epoch_metrics)

    return model, optimizer_state, train_log, val_log