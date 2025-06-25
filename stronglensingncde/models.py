import jax
jax.config.update("jax_enable_x64", True)
import diffrax

import jax.nn as jnn
import equinox as eqx  
import jax.numpy as jnp
import jax.random as jr

from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray

def _apply_weight_norm(x):
    if isinstance(x, eqx.nn.Linear):
        x = eqx.nn.WeightNorm(x)
    return x

def apply_WeightNorm(model):

    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    output_model = jax.tree_util.tree_map(
        _apply_weight_norm, model, is_leaf=is_linear
    )

    return output_model

class SamplingWeights(eqx.Module):
    logits: Array

    def __init__(self, num_classes):

        self.logits = jnp.ones(num_classes)

    def __call__(self, *args, **kwargs):

        return jax.nn.softmax(self.logits)

class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(
        self,
        data_size,
        hidden_size,
        width_size,
        depth,
        activation=jnn.softplus,
        final_activation=jnn.tanh,
        dtype=None,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up. (Just like how GRUs and LSTMs constrain the
            # rate of change of their hidden states.)
            final_activation=final_activation,
            key=key,
            dtype=dtype
        )

    def __call__(self, t, y, args):

        output = self.mlp(y).reshape(self.hidden_size, self.data_size)
        output = output.astype(jnp.float64)

        return output

class OnlineNCDE(eqx.Module):
    initial: eqx.nn.MLP
    vector_field: VectorField
    adjoint: diffrax.AbstractAdjoint
    solver: diffrax.AbstractSolver
    max_steps: int
    atol: float
    rtol: float

    def __init__(
        self,
        data_size,
        hidden_size,
        width_size,
        depth,
        solver,
        adjoint,
        max_steps,
        rtol=1e-3,
        atol=1e-6,
        activation=jnn.softplus,
        weight_norm=False,
        dtype=None,
        *,
        key,
        **kwargs
    ):
        super().__init__()
        ikey, fkey, lkey = jr.split(key, 3)
        initial = eqx.nn.MLP(
            data_size,
            hidden_size,
            width_size,
            depth,
            key=ikey,
            activation=activation,
            dtype=dtype,
        )
        vector_field = VectorField(
            data_size,
            hidden_size,
            width_size,
            depth,
            key=fkey,
            activation=activation,
            dtype=dtype,
        )
        if weight_norm:
            initial = apply_WeightNorm(initial)
            vector_field = apply_WeightNorm(vector_field)
        
        self.initial = initial
        self.vector_field = vector_field
        self.solver = solver
        self.adjoint = adjoint
        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol

    def __call__(self, ts, ts_interp, obs_interp, tmax):
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.

        control = diffrax.LinearInterpolation(ts_interp, obs_interp)
        term = diffrax.ControlTerm(self.vector_field, control).to_ode()
        dt0 = None
        y0 = self.initial(control.evaluate(ts[0]))
        y0 = y0.astype(jnp.float64)

        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            term,
            self.solver,
            ts[0],
            tmax,
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol#, jump_ts=ts
            ),
            saveat=saveat,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )

        representations = solution.ys
        representations_shape = representations.shape
        representations_mask = ts <= tmax
        representations = jnp.where(
            representations_mask[:, None],
            representations,
            jnp.ones(representations_shape)*-99
        )

        return representations

class PoolingONCDEClassifier(eqx.Module):

    ncde: OnlineNCDE
    classifier: eqx.nn.MLP

    def __init__(
        self,
        input_feature_size: int,
        representation_size: int,
        ncde_width: int,
        ncde_depth: int,
        ncde_solver: Callable | str,
        ncde_adjoint: Callable | str,
        ncde_max_steps: int,
        ncde_rtol: float,
        ncde_atol: float,
        classifier_width: int,
        classifier_depth: int,
        num_classes: int,
        ncde_activation: Callable | str = jnn.softplus,
        classifier_activation: Callable | str = jnn.leaky_relu,
        ncde_weight_norm: bool = False,
        ncde_dtype = None,
        classifier_dtype = None,
        *,
        key,
        **kwargs
    ):
        super().__init__()
        ncde_key, classifier_key = jr.split(key, 2)

        is_reversible = False
        if isinstance(ncde_adjoint, str):
            is_reversible = 'Reversible' in ncde_adjoint
            try:
                ncde_adjoint = getattr(diffrax, ncde_adjoint)()
            except AttributeError:
                raise ValueError(f"Adjoint method {ncde_adjoint} not found in diffrax.")

        if isinstance(ncde_dtype, str):
            try:
                ncde_dtype = getattr(jnp, ncde_dtype)
            except AttributeError:
                raise ValueError(f"NCDE dtype {ncde_dtype} not found in jax.numpy")

        if isinstance(ncde_solver, str):
            try:
                ncde_solver = getattr(diffrax, ncde_solver)()
            except AttributeError:
                raise ValueError(f"Solver method {ncde_solver} not found in diffrax.")
            if is_reversible:
                ncde_solver = diffrax.Reversible(ncde_solver)
        
        if isinstance(ncde_activation, str):
            try:
                ncde_activation = getattr(jnn, ncde_activation)
            except AttributeError:
                raise ValueError(f"Activation {ncde_activation} not found in jax.nn.")

        if isinstance(classifier_activation, str):
            try:
                classifier_activation = getattr(jnn, classifier_activation)
            except AttributeError:
                raise ValueError(f"Activation {classifier_activation} not found in jax.nn.")
        
        if isinstance(classifier_dtype, str):
            try:
                classifier_dtype = getattr(jnp, classifier_dtype)
            except AttributeError:
                raise ValueError(f"NCDE dtype {classifier_dtype} not found in jax.numpy")

        self.ncde = OnlineNCDE(
            data_size=input_feature_size,
            hidden_size=representation_size,
            width_size=ncde_width,
            depth=ncde_depth,
            solver=ncde_solver,
            adjoint=ncde_adjoint,
            max_steps=ncde_max_steps,
            rtol=ncde_rtol,
            atol=ncde_atol,
            key=ncde_key,
            activation=ncde_activation,
            weight_norm=ncde_weight_norm,
            dtype=ncde_dtype,
        )

        self.classifier = eqx.nn.MLP(
            in_size=2*representation_size,
            out_size=num_classes,
            width_size=classifier_width,
            depth=classifier_depth,
            activation=classifier_activation,
            key=classifier_key,
            dtype=classifier_dtype,
        )

    def __call__(self, ts, ts_interp, obs_interp, t_max, valid_mask):

        representations = jax.vmap(
            self.ncde, in_axes=(None, 0, 0, None)
        )(ts, ts_interp, obs_interp, t_max) # (N_max_img, max_length, representation_size)

        #pooled_representations = jnp.mean(representations, axis=0) # (max_length, representation_size)
        pooled_representations = jnp.sum(representations, axis=0) / jnp.sum(valid_mask, axis=0) # (max_length, representation_size)

        contextualized_representations = jnp.concatenate(
            [
                representations,
                jnp.broadcast_to(pooled_representations[None, :, :], representations.shape)
            ],
            axis=-1
        )   # (N_max_img, max_length, 2 * representation_size)

        logits = jax.vmap(jax.vmap(self.classifier))(contextualized_representations)   # (N_max_img, max_length, num_classes)
        
        return logits, contextualized_representations