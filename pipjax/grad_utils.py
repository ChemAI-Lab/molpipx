from typing import Any, Callable
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap, jacrev, lax

from jaxtyping import Array, Float, PyTree


def get_pip_grad(model_pip: Callable, x: Float[Array, "..."], params_pip: PyTree) -> Float[Array, "..."]:
    """Compute the gradients of the PIP model

    Returns:
        array: gradients of the PIP model, (batch, number of pip * number of atoms * 3)
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


def get_forces(model: Callable, x: Float[Array, "..."], params: PyTree) -> Float[Array, "..."]:
    """Compute the forces for a Flax based PIP model using reverse mode differentiation.

    Returns:
        array: forces of the PIP model, (batch, number of atoms * 3)
    """

    @jit
    def grad_forces_rev_i(xyzi: Any):
        g_forces = jacrev(model, argnums=(1,))(params,
                                               lax.expand_dims(xyzi, dimensions=(0,)))
        g_forces = lax.squeeze(
            g_forces[0], dimensions=(0, 1, 2))  # size = (natoms, 3)
        return g_forces

    return vmap(grad_forces_rev_i, in_axes=(0,))(x)
