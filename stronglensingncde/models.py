import jax
jax.config.update("jax_enable_x64", True)
import diffrax

import jax.nn as jnn
import equinox as eqx  
import jax.numpy as jnp
import jax.random as jr

from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray

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

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up. (Just like how GRUs and LSTMs constrain the
            # rate of change of their hidden states.)
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)

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
        *,
        key,
        **kwargs
    ):
        super().__init__()
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.vector_field = VectorField(data_size, hidden_size, width_size, depth, key=fkey)
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
        )

        self.classifier = eqx.nn.MLP(
            in_size=2*representation_size,
            out_size=num_classes,
            width_size=classifier_width,
            depth=classifier_depth,
            activation=jnn.leaky_relu,
            final_activation=lambda x: x,
            key=classifier_key,
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
        
        return logits