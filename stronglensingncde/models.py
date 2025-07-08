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

def make_stacked_diffusion(num_stacks, additive_scale, multiplicative_scale):

    def _diffusion(t, y, args):
        print(y.shape)
        additive = additive_scale * jnp.ones_like(y)
        multiplicative = multiplicative_scale * y

        return additive + multiplicative

    def diffusion(t, y, args):

        outputs = []
        for i in range(num_stacks):
            outputs.append(
                _diffusion(t, y[i], args)
            )
        outputs = tuple(outputs)
        return outputs
    
    return diffusion

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
        
    def __call__(self, x, x_prev, x_mean, delay, mask, hidden):

        # decay factors
        gamma_x = jnp.exp(
            -jnp.max(
                jnp.zeros_like(self.idecay_weight),
                self.idecay_weight * delay + self.idecay_bias
            )
        )
        gamma_h = jnp.exp(
            -jnp.max(
                self.zeros_like(self.nbias),
                self.hdecay(hidden)
            )
        )

        # Impute missing values using decay and training mean
        x_imputed = mask * x + (1 - mask) * ( gamma_x * x_prev + (1 - gamma_x) * x_mean )
        
        # hidden state decay
        h = gamma_h * hidden

        igates = jnp.split(self.igates(x_imputed), 3)
        mgates = jnp.split(self.mgates(x_imputed), 3)
        hgates = jnp.split(self.hgates(hidden), 3)

        rt = jnn.sigmoid(igates[0] + hgates[0]+mgates[0])
        zt = jnn.sigmoid(igates[1] + hgates[1] + mgates[1])
        nt = jnn.tanh(igates[2] + rt * (hgates[2] + self.nbias) + mgates[2])
        ht = (1 - zt) * h + zt * nt

        return ht

class OnlineGRUD(eqx.Module):
    grud_cell: GRUDCell
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        
        grud_cell = GRUDCell(
            input_size=input_size,
            hidden_size=hidden_size,
            use_bias=use_bias,
            key=key,
        )
        self.grud_cell = grud_cell
        self.hidden_size = hidden_size

    def __call__(self, x, x_mean, delay, mask):

        n_steps = x.shape[0]
        init_hidden_state = jnp.zeros((self.hidden_size,))
        hidden_states = [
            self.grud_cell(
                x=x[0],
                x_prev=x_mean,
                x_mean=x_mean,
                delay=delay[0],
                mask=mask[0],
                hidden=init_hidden_state
            )
        ]
        for i in range(1, n_steps):
            hidden_states.append(
                self.grud_cell(
                    x=x[i],
                    x_prev=x[i-1],
                    x_mean=x_mean,
                    delay=delay[i],
                    mask=mask[i],
                    hidden=hidden_states[-1]
                )
            )

        hidden_states = jnp.stack(hidden_states, axis=0)

        return hidden_states

class PoolingOnlineGRUDClassifier(eqx.Module):
    grud: OnlineGRUD
    classifier: eqx.nn.MLP
    input_means: Array = eqx.field(static=True)

    def __init__(
        self,
        input_feature_size: int,
        representation_size: int,
        grud_hidden_size: int,
        grud_use_bias: bool = True,
        classifier_width: int = 64,
        classifier_depth: int = 2,
        classifier_activation: Callable | str = jnn.leaky_relu,
        num_classes: int = 10,
        input_means: Array | None = None,
        *,
        key: PRNGKeyArray,
        **kwargs
    ):
        super().__init__(**kwargs)
        grud_key, classifier_key = jr.split(key, 2)

        grud = OnlineGRUD(
            input_size=input_feature_size,
            hidden_size=grud_hidden_size,
            use_bias=grud_use_bias,
            key=grud_key
        )
        self.grud = grud

        self.classifier = eqx.nn.MLP(
            in_size=2*representation_size,
            out_size=num_classes,
            width_size=classifier_width,
            depth=classifier_depth,
            activation=classifier_activation,
            key=classifier_key
        )
        
        if isinstance(input_means, jnp.ndarray):
            self.input_means = input_means
        else:
            self.input_means = jnp.zeros((input_feature_size,))


    def call(self, x, x_mean, delay, mask, valid_mask):
        
        representations = jax.vmap(self.grud)(x, x_mean, delay, mask)

        pooled_representations = jnp.sum(representations, axis=0) / jnp.sum(valid_mask, axis=0)

        contextualized_representations = jnp.concatenate(
            [
                representations,
                jnp.broadcast_to(pooled_representations[None, :], representations.shape)
            ],
            axis=-1
        )  

        logits = jax.vmap(jax.vmap(self.classifier))(contextualized_representations) 

        return logits, contextualized_representations

    def transform_input(self, ts, ts_interp, obs_interp):

        delta = 1

    def __call__(self, ts, ts_interp, obs_interp, t_max, redshifts, valid_mask):
        pass

class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    vf: eqx.nn.Linear
    gate: eqx.nn.Linear | None
    final_activation: Callable
    data_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    gated: bool = eqx.field(static=True)

    def __init__(
        self,
        data_size,
        hidden_size,
        width_size,
        depth,
        activation=jnn.softplus,
        final_activation=jnn.tanh,
        gated: bool = False,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.gated = gated
        self.final_activation = final_activation

        depth = max(0, depth-2)

        mlpkey, vfkey, gatekey = jr.split(key, 3)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=width_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=activation,
            key=mlpkey,
        )

        gate = None
        if gated:
            gate = eqx.nn.Linear(
                in_features=width_size,
                out_features=hidden_size * data_size,
                key=gatekey
            )
        self.gate = gate
        self.vf = eqx.nn.Linear(
            in_features=width_size,
            out_features=hidden_size * data_size,
            key=vfkey,
        )

    def __call__(self, t, y, args):

        mlp_output = self.mlp(y)
        vf_output = self.vf(mlp_output)
        vf_output = self.final_activation(vf_output)
        if self.gated:
            gate_output = self.gate(mlp_output)
            gate_output = jnn.sigmoid(gate_output)
            vf_output = jnp.multiply(vf_output, gate_output)
        
        vf_output = vf_output.reshape(self.hidden_size, self.data_size)

        return vf_output
    
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
        gated: bool = False,
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
                key=keys[0],
                gated=gated
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
                        key=keys[i+1],
                        gated=gated
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
    hidden_states: tuple[eqx.nn.MLP, ...]
    num_hidden_states: int = eqx.field(static=True)
    metadata_size: int = eqx.field(static=True)

    def __init__(
        self,
        num_hidden_states: int,
        data_size: int,
        hidden_size: int | Sequence[int],
        width_size: int | Sequence[int],
        depth: int | Sequence[int],
        activation=jnn.softplus,
        metadata_size: int = 0,
        *,
        key,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.num_hidden_states = num_hidden_states
        self.metadata_size = metadata_size

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
                in_size=data_size+metadata_size,
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

    def __call__(self, x, metadata) -> tuple[Array, ...]:

        if self.metadata_size > 0:
            x = jnp.concatenate(
                [x, metadata], axis=-1
            )
        
        outputs = []
        outputs.append(
            self.hidden_states[0](x)
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

class StackedBrownianMotion(eqx.Module):
    num_stacks: int = eqx.field(static=True)
    brownian_motions: tuple[diffrax.VirtualBrownianTree, ...]

    def __init__(self, num_stacks, t0, t1, tol, key):

        super().__init__()
        self.num_stacks = num_stacks
        keys = jr.split(key, num_stacks)
        brownian_motions = []
        brownian_motions.append(
            diffrax.VirtualBrownianTree(
                t0, t1, tol=tol, shape=(), key=keys[0]
            )
        )
        if num_stacks > 1:
            for i in range(num_stacks-1):
                brownian_motions.append(
                    diffrax.VirtualBrownianTree(
                        t0, t1, tol=tol, shape=(), key=keys[i+1]
                    )
                )
        brownian_motions = tuple(brownian_motions)
        self.brownian_motions = brownian_motions

    def evaluate(self, t0, t1=None, left=True, use_levy=False):
        outputs = []
        outputs.append(
            self.brownian_motions[0].evaluate(
                t0=t0, t1=t1, left=left, use_levy=use_levy
            )
        )
        if self.num_stacks > 1:
            for i in range(self.num_stacks-1):
                outputs.append(
                    self.brownian_motions[i+1].evaluate(
                        t0=t0, t1=t1, left=left, use_levy=use_levy
                    )
                )
        outputs = tuple(outputs)

        return outputs

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
    diffusion_term: Callable | None
    inference: bool
    num_stacks: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    pcoeff: float = eqx.field(static=True)
    icoeff: float = eqx.field(static=True)
    dtmin: float = eqx.field(static=True)
    use_jump_ts: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)
    use_noise: bool = eqx.field(static=True)
    additive_noise_scale: float = eqx.field(static=True)
    multiplicative_noise_scale: float = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        data_size,
        metadata_size,
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
        gated: bool = False,
        use_noise: bool = False,
        additive_noise_scale: float = None,
        multiplicative_noise_scale: float = None,
        dtmin: float = None,
        *,
        key,
        **kwargs
    ):
        super().__init__()
        ikey, fkey, lkey = jr.split(key, 3)
        initial = StackedInitialHiddenState(
            num_hidden_states=num_stacks,
            data_size=data_size,
            metadata_size=metadata_size,
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
            gated=gated,
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
        self.dtmin = dtmin
        self.use_jump_ts = use_jump_ts
        self.throw = throw
        self.use_noise = use_noise
        self.additive_noise_scale = additive_noise_scale
        self.multiplicative_noise_scale = multiplicative_noise_scale
        self.hidden_size = hidden_size

        if use_noise:

            is_invalid_add = additive_noise_scale == None
            is_invalid_mul = multiplicative_noise_scale == None
            is_invalid_noise = is_invalid_add or is_invalid_mul
            if is_invalid_noise:
                raise ValueError(
                    (
                        "Noise scales must be specified if using noise, " + 
                        f"currently additive noise scale is {additive_noise_scale} " +
                        f"and multiplicative noise scale is {multiplicative_noise_scale}"
                    )
                )
            if not dtmin:
                raise ValueError(
                    f"Mininum step size must be fixed if noise, currently {dtmin}"
                )
            self.diffusion_term = make_stacked_diffusion(
                num_stacks=num_stacks,
                additive_scale=additive_noise_scale,
                multiplicative_scale=multiplicative_noise_scale
            )
            self.inference = False
        else:
            self.diffusion_term = None
            self.inference = True

    def __call__(self, ts, ts_interp, obs_interp, tmax, metadata, key):

        control = StackedLinearInterpolation(ts_interp, obs_interp, self.num_stacks)
        terms = diffrax.ControlTerm(self.vector_field, control).to_ode()

        use_noise = self.use_noise and not self.inference
        if use_noise:
            brownian_motion = StackedBrownianMotion(
                num_stacks=self.num_stacks,
                t0=ts[0],
                t1=ts[-1]+2*self.dtmin,
                tol=self.dtmin/2,
                key=key
            )
            diffusion = self.diffusion_term
            terms = diffrax.MultiTerm(terms, diffrax.ControlTerm(diffusion, brownian_motion))

        dt0 = None
        x0 = control(ts[0])[0]
        y0 = self.initial(x0, metadata)

        if self.use_jump_ts:
            jump_ts = ts
        else:
            jump_ts = None
        

        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            terms,
            self.solver,
            ts[0],
            tmax,
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol, jump_ts=jump_ts,
                pcoeff=self.pcoeff, icoeff=self.icoeff,
                dtmin=self.dtmin,
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
        metadata_size: int,
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
        ncde_gated: bool = False,
        ncde_use_noise: bool = False,
        ncde_additive_noise_scale: float = None,
        ncde_multiplicative_noise_scale: float = None,
        ncde_dtmin: float = None,
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
            metadata_size=metadata_size,
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
            gated=ncde_gated,
            use_noise=ncde_use_noise,
            additive_noise_scale=ncde_additive_noise_scale,
            multiplicative_noise_scale=ncde_multiplicative_noise_scale,
            dtmin=ncde_dtmin,
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

    def __call__(self, ts, ts_interp, obs_interp, t_max, redshifts, valid_mask, key):

        keys = jr.split(key, ts_interp.shape[0])
        representations, solution = jax.vmap(
            self.ncde, in_axes=(None, 0, 0, None, 0, 0)
        )(ts, ts_interp, obs_interp, t_max, redshifts, keys) # (N_max_img, max_length, representation_size)

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