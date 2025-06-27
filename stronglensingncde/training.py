import jax
import time
import optax
import diffrax

import numpy as np
import equinox as eqx  
import jax.numpy as jnp
import stronglensingncde.utils as utils

from pathlib import Path
from collections.abc import Callable
from jax.tree_util import tree_map_with_path, keystr

def interleave_with_avg(x: jnp.ndarray) -> jnp.ndarray:
    """
    Interleave a 1D array by inserting midpoints
    """

    N = x.shape[0]
    # allocate output
    y = jnp.empty((2 * N - 1,), dtype=x.dtype)
    # place originals at even indices
    y = y.at[0::2].set(x)
    # place pairwise averages at odd indices
    y = y.at[1::2].set((x[:-1] + x[1:]) * 0.5)

    return y

def _interpolate_timeseries(times, flux, partial_ts):

    _, flux_rect = diffrax.rectilinear_interpolation(
        times, flux, replace_nans_at_start=0.
    )
    colors = -jnp.diff(flux_rect, axis=-1)
    
    s_rect, partial_rect = diffrax.rectilinear_interpolation(
        times, partial_ts, replace_nans_at_start=0.
    )


    s = times
    #s_rect = interleave_with_avg(times)

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

def make_lr_schedule(
    lr_schedule_fn: Callable | str,
    warmup_steps,
    total_steps,
    lr_schedule_settings: dict = {}
):
    
    if isinstance(lr_schedule_fn, str):
        # If a string is provided, use the corresponding Optax schedule
        try:
            lr_schedule_fn = getattr(optax.schedules, lr_schedule_fn)
        except AttributeError:
            raise ValueError(f"Schedule {lr_schedule_fn} not found in optax.schedules.")
    
    lr_schedule = lr_schedule_fn(
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        **lr_schedule_settings
    )

    return lr_schedule

def make_optimizer_mask_fn(target='ncde'):

    def mask_fn(grads):

        return tree_map_with_path(
            lambda path, x: target in keystr(path),
            grads
        )
    
    return mask_fn

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
        
        max_s = max_times #+ (lengths-1) / 1000

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

        return loss, aux, model, optimizer_state, gradients
    
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

        max_s = max_times #+ (lengths-1) / 1000

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
        
        return loss, aux, model, optimizer_state, None
    
    return val_step

def inner_loop(
    model,
    dataloader,
    optimizer_state,
    step_fn,
    fixed_lr = None,
    number_of_steps = None,
    verbose = False,
    exception_path = None,
    only_use_first_column = False,
    except_on_failure = False,
):
    
    if not number_of_steps:
        number_of_steps = len(dataloader)

    total_loss = 0
    
    losses = []
    metrics = []
    aux_vals = []
    failure_rate = []
    lrs = []

    t_init = time.time()
    for step in range(number_of_steps):
        
        t_step_init = time.time()

        if fixed_lr is None:
            current_lr = optimizer_state[1].inner_state.hyperparams['learning_rate']
            current_lr = current_lr#.astype(jnp.float64)
        else:
            current_lr = fixed_lr

        grads_contain_inf = False
        grads_contain_nan = False

        data = next(dataloader)
        data = [output.numpy() for output in data]

        if only_use_first_column:
            (
                times, flux, partial_ts, trigger_idx,
                lengths, peak_times, max_times, 
                binary_labels, multiclass_labels,
                valid_lightcurve_mask
            ) = data

            flux = flux[:, 0:1]
            partial_ts = partial_ts[:, 0:1]
            peak_times = peak_times[:, 0:1]
            multiclass_labels = multiclass_labels[:, 0:1]
            valid_lightcurve_mask = valid_lightcurve_mask[:, 0:1]

            data = (
                times, flux, partial_ts, trigger_idx,
                lengths, peak_times, max_times, 
                binary_labels, multiclass_labels,
                valid_lightcurve_mask
            )   
        
        loss, aux, model, optimizer_state, gradients = step_fn(
            model, data, optimizer_state
        )
        has_gradients = not isinstance(gradients, type(None))
        if has_gradients:
            grads_contain_inf = utils.tree_contains_inf(gradients)
            grads_contain_nan = utils.tree_contains_nan(gradients)

        step_solution_flags = aux[2]
        is_failure = step_solution_flags != 0.
        step_num_failures = np.sum(is_failure)
        step_failure_rate = step_num_failures / len(data[0])
        failure_rate.append(step_failure_rate)

        invalid_grads = grads_contain_inf | grads_contain_nan
        loss_is_invalid = ~jnp.isfinite(loss)
        loss_or_grads_invalid = invalid_grads | loss_is_invalid
        
        if except_on_failure:
            has_failure = step_num_failures != 0
            loss_or_grads_invalid = loss_or_grads_invalid | has_failure

        if loss_or_grads_invalid:

            if exception_path:
                exception_path = exception_path / 'exception'
                exception_path.mkdir(parents=True, exist_ok=True)

                names = (
                    "times", "flux", "partial_ts", "trigger_idx",
                    "lengths", "peak_times", "max_times", 
                    "binary_labels", "multiclass_labels",
                    "valid_lightcurve_mask"
                )

                for arr, name in zip(data, names):
                    np.save(exception_path / f"{name}.npy", arr)

                for arr, name in zip(aux, [losses, metrics]):
                    np.save(exception_path / f"{name}.npy", arr)
                
                np.save(exception_path / f"solution_flags.npy", step_solution_flags)
                utils.save_model(exception_path / "model_at_failure.eqx", model)
                utils.save_model(exception_path / "grads_at_failure.eqx", gradients)

                exception_string += f"Metadata has been saved to:\n{exception_path}"  
                
            exception_string = (
                "Gradients or loss is invalid. They contain:\n" +
                f"Infs: {grads_contain_inf}\n" +
                f"NaNs: {grads_contain_nan}\n" +
                f"Loss: {loss_is_invalid}\n" +
                f"Num. Failures: {step_num_failures} / {step_failure_rate*100:.2f}%\n" 
            ) 

            print(exception_string)
            raise ValueError(exception_string)

        total_loss += loss
        step_losses = aux[0]
        step_metrics = aux[1]

        losses.append(step_losses)
        metrics.append(step_metrics)
        lrs.append(current_lr)
        
        step_duration = time.time() - t_step_init

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
                f"Failure Rate: {step_failure_rate*100:.2f}% | " +
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
    
    avg_loss = total_loss / number_of_steps
    losses = np.array(losses)
    metrics = np.array(metrics)
    failure_rate = np.array(failure_rate)
    lrs = np.array(lrs)

    avg_metrics = np.mean(metrics, axis=0)
    avg_losses = np.mean(losses, axis=0)
    avg_failure_rate = np.mean(failure_rate)
    init_lr, final_lr = lrs[0], lrs[-1]
    
    total_time = time.time() - t_init
    avg_step_time = total_time / number_of_steps

    aux_vals = (
        avg_losses, avg_metrics, avg_step_time,
        losses, (init_lr, final_lr), metrics,
        avg_failure_rate
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
    number_of_epochs: int = 1,
    number_of_warmup_epochs: int = 1,
    steps_per_epoch: int = 10,
    verbose_steps: bool = False,
    verbose_epochs: bool = True,
    save_path: str = None,
    patience: int = 10,
    val_steps_per_epoch: int = None,
    save_every_n_epochs: int = None,
    only_use_first_column: bool = False,
    except_on_failure: bool = False,
):
    
    total_number_of_epochs = number_of_epochs + number_of_warmup_epochs
    if save_path:

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    optimizer_state = optimizer.init(
        eqx.filter(model, eqx.is_inexact_array)
    )
    
    train_step = make_train_step(optimizer, loss_fn)
    val_step = make_val_step(loss_fn)

    def infinite_dataloader(dataloader):
        while True:
            for data in dataloader:
                yield data

    if verbose_epochs:
        num_batches_in_dataset = len(train_dataloader)
        num_full_passes = total_number_of_epochs * steps_per_epoch / num_batches_in_dataset
        init_str = (
            f"\nNum. Full Passes: {num_full_passes:.2f} | "+
            f"Num. Warmup Epochs: {number_of_warmup_epochs} | " +
            f"Num. Train Epochs: {number_of_epochs} | " +
            f"Batches per Epoch: {steps_per_epoch}\n"
        )
        print(init_str)

    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataloader)
    if val_steps_per_epoch is None:
        val_steps_per_epoch = len(val_dataloader) 

    train_dataloader = infinite_dataloader(train_dataloader)
    val_dataloader = infinite_dataloader(val_dataloader)

    training_epoch_losses = np.zeros((total_number_of_epochs, num_loss_components))
    training_epoch_metrics = np.zeros((total_number_of_epochs, 6))

    val_epoch_losses = np.zeros((total_number_of_epochs, num_loss_components))
    val_epoch_metrics = np.zeros((total_number_of_epochs, 6))

    best_val_loss = np.inf
    best_val_epoch = 0
    wait = 0

    t_training_init = time.time()
    for epoch in range(1, total_number_of_epochs+1):
        
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
            verbose=verbose_steps,
            exception_path=save_path,
            only_use_first_column=only_use_first_column,
            except_on_failure=except_on_failure
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
            number_of_steps=val_steps_per_epoch,
            verbose=verbose_steps,
            exception_path=None,
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
            train_failure_rate = train_aux[-1]

            train_string = (
                f"Train - " +
                f"Loss: {avg_train_loss:.2e} | " +
                f"Failure Rate: {train_failure_rate*100:.2f}% | " +
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
            val_failure_rate = val_aux[-1]

            val_string = (
                f"Val   - " +
                f"Loss: {avg_val_loss:.2e} | " +
                f"Failure Rate: {val_failure_rate*100:.2f}% | " +
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

        if avg_val_loss < best_val_loss:
            
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

        if save_every_n_epochs:
            if (epoch-1) % save_every_n_epochs == 0:
                utils.save_model(save_path / "latest_model.eqx", model)
                np.save(save_path / "train_losses.npy", training_epoch_losses)
                np.save(save_path / "train_metrics.npy", training_epoch_metrics)
                np.save(save_path / "val_losses.npy", val_epoch_losses)
                np.save(save_path / "val_metrics.npy", val_epoch_metrics)

            

    if save_path:
        utils.save_model(save_path / "final_model.eqx", model)
        np.save(save_path / "train_losses.npy", training_epoch_losses)
        np.save(save_path / "train_metrics.npy", training_epoch_metrics)
        np.save(save_path / "val_losses.npy", val_epoch_losses)
        np.save(save_path / "val_metrics.npy", val_epoch_metrics)

    training_duration = time.time() - t_training_init
    avg_epoch_duration = training_duration / total_number_of_epochs

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