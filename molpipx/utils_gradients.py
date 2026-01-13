from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacrev, lax, value_and_grad

from jaxtyping import Array, Float, PyTree


@partial(jax.jit, static_argnums=(0, ))
def get_pip_grad(model_pip: Callable, x: Float[Array, "..."], params_pip: PyTree) -> Float[Array, "..."]:
    """Computes the gradients of the PIP vectors with respect to the input coordinates.

    Args:
        model_pip (Callable): The PIP model function (e.g., ``model.apply``).
        x (Array): Input geometries with shape (Batch, Na, 3).
        params_pip (PyTree): The parameters of the PIP model.

    Returns:
        Array: Gradients of the PIP model, shape (Batch, N_pips * Na * 3).
    """

    @jit
    def grad_pip_rev_i(xyzi: Any):
        grad_pip = jacrev(model_pip, argnums=(1,))(params_pip,
                                                   lax.expand_dims(xyzi, dimensions=(0,)))
        grad_pip_og = lax.squeeze(
            grad_pip[0], dimensions=(0, 2))  # size = (npip, natoms,3)
        d_pip, n_a, n_d = grad_pip_og.shape
        grad_pip = jnp.reshape(grad_pip_og, (d_pip, n_a*n_d))
        return jnp.transpose(grad_pip)

    return vmap(grad_pip_rev_i, in_axes=(0,))(x)


@partial(jax.jit, static_argnums=(0, ))
def get_forces(model: Callable, x: Float[Array, "..."], params: PyTree) -> Float[Array, "..."]:
    """Compute the forces for a Flax based PIP model using reverse mode differentiation.

    Args:
        model (Callable): The energy model function (e.g., ``energy_model.apply``).
        x (Array): Input geometries with shape (Batch, Na, 3).
        params (PyTree): The parameters of the energy model.

    Returns:
        Array: Forces (gradients) for each atom, shape (Batch, Na, 3).
    """

    @jit
    def grad_forces_rev_i(xyzi: Any):
        g_forces = jacrev(model, argnums=(1,))(params,
                                               lax.expand_dims(xyzi, dimensions=(0,)))
        g_forces = lax.squeeze(
            g_forces[0], dimensions=(0, 1, 2))  # size = (natoms, 3)
        return g_forces

    return jit(vmap(grad_forces_rev_i, in_axes=(0,)))(x)


@partial(jax.jit, static_argnums=(0,))
def get_energy_and_forces(model: Callable, x: Float[Array, "..."], params: PyTree) -> Float[Array, "..."]:
    """Compute the energy and the forces for a Flax based PIP model using ``value_and_grad`` function.

    Args:
        model (Callable): The energy model function.
        x (Array): Input geometries with shape (Batch, Na, 3).
        params (PyTree): The parameters of the model.

    Returns:
        Tuple[Array, Array]: A tuple containing:
            * Energy values (Batch, 1)
            * Forces/Gradients (Batch, Na, 3)
    """

    @jit
    def energy_float(params, x):
        return jnp.sum(model(params, x))

    @jit
    def grad_forces_rev_i(xyzi: Any):
        energy, g_forces = value_and_grad(energy_float, argnums=(1))(
            params, lax.expand_dims(xyzi, dimensions=(0,)))
        return energy, g_forces[0]

    return vmap(grad_forces_rev_i, in_axes=(0,))(x)

def get_forces_gp(train_data,gp_model,x):
    """Computes forces (gradients) for a Gaussian Process model.

    Args:
        train_data (Dataset): The training dataset used by the GP.
        gp_model (GP): The GPJax model instance.
        x (Array): Input geometries.

    Returns:
        Tuple: A tuple containing:
            * Forces (gradients of the mean)
            * A tuple of (predictive_mean, predictive_std)
    """

    def gp_prediction(x):
        latent_dist = gp_model(x, train_data=train_data)
        predictive_dist = gp_model.likelihood(latent_dist)
        predictive_mean = predictive_dist.mean
        predictive_std = jnp.sqrt(predictive_dist.variance)
        return jnp.sum(predictive_mean), jnp.sum(predictive_std)

    def grad_forces_rev_i(xyzi: Any):
        mu_and_std, g_forces = jax.value_and_grad(gp_prediction, argnums=(0,),has_aux=True)(jax.lax.expand_dims(xyzi, dimensions=(0,)))
        return g_forces[0], mu_and_std

    return jax.vmap(grad_forces_rev_i, in_axes=(0,))(x)
