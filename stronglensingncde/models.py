import jax
import math
import diffrax

import jax.nn as jnn
import equinox as eqx  
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from typing import Sequence, Any
from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray

def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], lim: float
) -> jax.Array:
    return jr.uniform(key, shape, minval=-lim, maxval=lim)

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

def _cast_tree(tree, from_dtype, to_dtype):
    """
    Recursively cast all JAX arrays in a pytree from one floating dtype to another.
    """
    def _cast(x):
        if isinstance(x, jnp.ndarray) and x.dtype == from_dtype:
            return x.astype(to_dtype)
        return x
    return jax.tree_map(_cast, tree)

class SamplingWeights(eqx.Module):
    logits: Array

    def __init__(self, num_classes):

        self.logits = jnp.ones(num_classes)

    def __call__(self, *args, **kwargs):

        return jax.nn.softmax(self.logits)

class GRUDCell(eqx.Module):

    idecay_weight: Array
    idecay_bias: Array
    nbias: Array
    hdecay: eqx.nn.Linear
    igates: eqx.nn.Linear
    hgates: eqx.nn.Linear
    mgates: eqx.nn.Linear
    use_bias: bool


    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        
        idwkey, idbkey, nbkey, hdkey, igkey, hgkey, mgkey = jr.split(key, 7)
        lim = math.sqrt(1 / hidden_size)

        self.idecay_weight = default_init(
            key=idwkey,
            shape=input_size,
            lim=lim
        )
        self.idecay_bias = default_init(
            key=idbkey,
            shape=input_size,
            lim=lim
        )
        self.nbias = default_init(
            key=nbkey,
            shape=hidden_size,
            lim=lim
        )
        self.hdecay = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            use_bias=use_bias,
            key=hdkey,
        )
        self.igates = eqx.nn.Linear(
            in_features=input_size,
            out_features=3*hidden_size,
            use_bias=False,
            key=igkey,
        )
        self.mgates = eqx.nn.Linear(
            in_features=input_size,
            out_features=3*hidden_size,
            use_bias=use_bias,
            key=mgkey,
        )
        self.hgates = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=3*hidden_size,
            use_bias=False,
            key=hgkey,
        )

        self.use_bias = use_bias
        
    def __call__(self, input, delay, mask, hidden):
        pass

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
        )

    def __call__(self, t, y, args):

        output = self.mlp(y).reshape(self.hidden_size, self.data_size)

        return output
    
class StackedVectorField(eqx.Module):
    num_vector_fields: int = eqx.field(static=True)
    vector_fields: tuple[VectorField, ...]

    def __init__(
        self,
        num_vector_fields: int,
        data_size: int,
        hidden_size: int | Sequence[int],
        width_size: int | Sequence[int],
        depth: int | Sequence[int],
        activation=jnn.softplus,
        final_activation=jnn.tanh,
        *,
        key,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.num_vector_fields = num_vector_fields

        if isinstance(hidden_size, int):
            hidden_sizes = tuple([hidden_size] * num_vector_fields)

        if isinstance(width_size, int):
            width_size = tuple([width_size] * num_vector_fields)

        if isinstance(depth, int):
            depth = tuple([depth] * num_vector_fields)

        keys = jr.split(key, num_vector_fields)
        vector_fields = []
        vector_fields.append(
            VectorField(
                data_size=data_size,
                hidden_size=hidden_sizes[0],
                width_size=width_size[0],
                depth=depth[0],
                activation=activation,
                final_activation=final_activation,
                key=keys[0]
            )
        )
        if num_vector_fields > 1:
            for i in range(num_vector_fields-1):
                vector_fields.append(
                    VectorField(
                        data_size=hidden_sizes[i],
                        hidden_size=hidden_sizes[i+1],
                        width_size=width_size[i+1],
                        depth=depth[i+1],
                        activation=activation,
                        final_activation=final_activation,
                        key=keys[i+1]
                    )
                )
        vector_fields = tuple(vector_fields)
        self.vector_fields = vector_fields
    
    def __call__(self, t, y, args) -> tuple[Array, ...]:

        outputs = []
        outputs.append(
            self.vector_fields[0](t, y[0], args)
        )
        if self.num_vector_fields > 1:
            for i in range(self.num_vector_fields-1):
                vector_field_output = self.vector_fields[i+1](
                    t, y[i+1], args
                )
                modified_vector_field_output = jnp.matmul(
                    vector_field_output, outputs[-1]
                )
                outputs.append(modified_vector_field_output)
        
        outputs = tuple(outputs)

        return outputs

class StackedInitialHiddenState(eqx.Module):
    num_hidden_states: int = eqx.field(static=True)
    hidden_states: tuple[eqx.nn.MLP, ...]

    def __init__(
        self,
        num_hidden_states: int,
        data_size: int,
        hidden_size: int | Sequence[int],
        width_size: int | Sequence[int],
        depth: int | Sequence[int],
        activation=jnn.softplus,
        *,
        key,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.num_hidden_states = num_hidden_states

        if isinstance(hidden_size, int):
            hidden_sizes = tuple([hidden_size] * num_hidden_states)

        if isinstance(width_size, int):
            width_sizes = tuple([width_size] * num_hidden_states)

        if isinstance(depth, int):
            depths = tuple([depth] * num_hidden_states)

        keys = jr.split(key, num_hidden_states)
        hidden_states = []

        hidden_states.append(
            eqx.nn.MLP(
                in_size=data_size,
                out_size=hidden_sizes[0],
                width_size=width_sizes[0],
                depth=depths[0],
                key=keys[0],
                activation=activation,
            )
        )
        if num_hidden_states > 1:
            for i in range(num_hidden_states - 1):
                hidden_states.append(
                    eqx.nn.MLP(
                        in_size=hidden_sizes[i],
                        out_size=hidden_sizes[i+1],
                        width_size=width_sizes[i+1],
                        depth=depths[i+1],
                        key=keys[i+1],
                        activation=activation,
                    )
                )
        hidden_states = tuple(hidden_states)
        self.hidden_states = hidden_states

    def __call__(self, x) -> tuple[Array, ...]:

        outputs = []
        outputs.append(
            self.hidden_states[0](x[0])
        )
        if self.num_hidden_states > 1:
            for i in range(self.num_hidden_states - 1):
                outputs.append(
                    self.hidden_states[i+1](outputs[i])
                )
        outputs = tuple(outputs)

        return outputs

class StackedLinearInterpolation(eqx.Module):
    num_stacks: int = eqx.field(static=True)
    linear_interp: diffrax.LinearInterpolation

    def __init__(self, ts, ys, num_stacks):

        super().__init__()
        self.num_stacks = num_stacks
        self.linear_interp = diffrax.LinearInterpolation(ts, ys)
    
    def evaluate(self, t0, t1=None, left=True):
        interp = self.linear_interp.evaluate(t0, t1, left)
        interp_tuple = tuple([interp] * self.num_stacks)
        return interp_tuple

    def __call__(self, t0, t1=None, left=True):
        return self.evaluate(t0, t1, left)

class MixedPrecisionWrapper(eqx.Module):
    """
    Wraps another Equinox module, casting inputs to float64 before the call
    and casting outputs back to float32 after the call.
    """
    module: eqx.Module

    def __call__(self, *args, **kwargs):
        # Cast all float32 inputs to float64
        args64 = _cast_tree(args, jnp.float32, jnp.float64)
        kwargs64 = _cast_tree(kwargs, jnp.float32, jnp.float64)

        # Forward through the wrapped module
        output = self.module(*args64, **kwargs64)

        # Cast float64 outputs back to float32
        output32 = _cast_tree(output, jnp.float64, jnp.float32)
        return output32

class OnlineNCDE(eqx.Module):
    initial: StackedInitialHiddenState
    vector_field: StackedVectorField
    adjoint: diffrax.AbstractAdjoint
    solver: diffrax.AbstractSolver
    num_stacks: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    pcoeff: float = eqx.field(static=True)
    icoeff: float = eqx.field(static=True)
    use_jump_ts: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)

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
        pcoeff=0.3,
        icoeff=0.4,
        activation=jnn.softplus,
        weight_norm=False,
        num_stacks = 1,
        use_jump_ts = False,
        throw=True,
        cast_f64=False,
        *,
        key,
        **kwargs
    ):
        super().__init__()
        ikey, fkey, lkey = jr.split(key, 3)
        initial = StackedInitialHiddenState(
            num_hidden_states=num_stacks,
            data_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=ikey,
            activation=activation,
        )

        vector_field = StackedVectorField(
            num_vector_fields=num_stacks,
            data_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=fkey,
            activation=activation,
        )
        if weight_norm:
            initial = apply_WeightNorm(initial)
            vector_field = apply_WeightNorm(vector_field)
        
        if cast_f64:
            initial = MixedPrecisionWrapper(initial)
            vector_field = MixedPrecisionWrapper(vector_field)

        self.initial = initial
        self.vector_field = vector_field
        self.num_stacks = num_stacks
        self.solver = solver
        self.adjoint = adjoint
        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol
        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.use_jump_ts = use_jump_ts
        self.throw = throw

    def __call__(self, ts, ts_interp, obs_interp, tmax):

        control = StackedLinearInterpolation(ts_interp, obs_interp, self.num_stacks)
        term = diffrax.ControlTerm(self.vector_field, control).to_ode()
        dt0 = None
        y0 = self.initial(control(ts[0]))

        if self.use_jump_ts:
            jump_ts = ts_interp
        else:
            jump_ts = None
            

        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            term,
            self.solver,
            ts[0],
            tmax,
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol, jump_ts=jump_ts,
                pcoeff=self.pcoeff, icoeff=self.icoeff
            ),
            saveat=saveat,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
            throw=self.throw,
        )

        representations = solution.ys[-1]
        representations_shape = representations.shape
        representations_mask = ts <= tmax
        representations = jnp.where(
            representations_mask[:, None],
            representations,
            jnp.ones(representations_shape)*-99
        )

        return representations, solution

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
        ncde_num_stacks: int = 1,
        ncde_activation: Callable | str = jnn.softplus,
        ncde_use_jump_ts: bool = False,
        ncde_throw: bool = True,
        ncde_pcoeff: float = 0.3,
        ncde_icoeff: float = 0.4,
        ncde_weight_norm: bool = False,
        ncde_cast_f64: bool = False,
        checkpoint_ncde: bool = False,
        classifier_activation: Callable | str = jnn.leaky_relu,
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

        ncde = OnlineNCDE(
            num_stacks=ncde_num_stacks,
            data_size=input_feature_size,
            hidden_size=representation_size,
            width_size=ncde_width,
            depth=ncde_depth,
            solver=ncde_solver,
            adjoint=ncde_adjoint,
            max_steps=ncde_max_steps,
            rtol=ncde_rtol,
            atol=ncde_atol,
            pcoeff=ncde_pcoeff,
            icoeff=ncde_icoeff,
            key=ncde_key,
            activation=ncde_activation,
            weight_norm=ncde_weight_norm,
            use_jump_ts=ncde_use_jump_ts,
            throw=ncde_throw,
            cast_f64=ncde_cast_f64,
        )

        if checkpoint_ncde:
            ncde = eqx.filter_checkpoint(ncde)

        self.ncde = ncde

        self.classifier = eqx.nn.MLP(
            in_size=2*representation_size,
            out_size=num_classes,
            width_size=classifier_width,
            depth=classifier_depth,
            activation=classifier_activation,
            key=classifier_key,
        )

    def __call__(self, ts, ts_interp, obs_interp, t_max, valid_mask):

        representations, solution = jax.vmap(
            self.ncde, in_axes=(None, 0, 0, None)
        )(ts, ts_interp, obs_interp, t_max) # (N_max_img, max_length, representation_size)

        solution_flags = solution.result._value

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
        
        return logits, contextualized_representations, solution_flags