from typing import Any
import jax
import jax.numpy as jnp
from jax import jit
from optax.losses import squared_error

from jaxtyping import Array, Float, PyTree


@jit
def all_distances(xi: Float[Array, "..."]) -> Float[Array, "..."]:
    """all distances between a geometry of a given molecule
    x: xyz coordinates

    Returns:
        r: all possible ||xi - xj||2 between the different atomic positions
    """
    n_atoms = xi.shape[0]
    z = xi[:, None] - xi[None, :]  # compute all difference
    i0 = jnp.triu_indices(
        n_atoms, 1
    )  # select upper diagonal (LEXIC ORDER)
    diff = z[i0]
    r = jnp.linalg.norm(diff, axis=1)  # compute the bond length
    return r


@jit
def softplus_inverse(x: Float[Array, "dim1"]) -> Float[Array, "dim1"]:
    """Computes the inverse softplus
    Returns:
        x: inverse softplus
    """
    return jnp.log(jnp.exp(x) - 1)


@jit
def morse_variables(x: Float[Array, "dim1"], l: Float[Array, ""]) -> Float[Array, "dim1"]:
    """
    compute morse variables using a single l parameter

    Returns:
        morese_variables: returns morse variables
    """
    r = all_distances(x)
    return jnp.exp(-l*r)


@jit
def mse_loss(predictions: Float[Array, "dim1"], targets: Float[Array, "dim1"]) -> Float:
    """Compute the Mean squared error

    Args:
        predictions: 
        targets: 

    Returns:
        Float: mean squared error
    """
    return jnp.mean(squared_error(predictions-targets))


@jit
def mae_loss(predictions: Float[Array, "dim1"], targets: Float[Array, "dim1"]) -> Float:
    """Computes the mean absolute error

    Args:
        predictions: 
        targets: 

    Returns:
        Float: mean absolute error
    """
    return jnp.mean(jnp.abs(predictions-targets))


def flax_params(w: Float[Array, '...'], params: PyTree) -> PyTree:
    """Array to Flax PyTree parameters

    Args:
        w (Float[Array]): Array with linear parameters for PIP
        params (PyTree): Flax PyTree parameters

    Returns:
        PyTree: Updated Flax PyTree parameters
    """

    w_base = params['params']['Dense_0']['kernel']
    w = jnp.reshape(w, w_base.shape)
    params['params']['Dense_0']['kernel'] = w
    return params


def flax_params_aniso(l: Float[Array, '...'], params: PyTree) -> PyTree:
    """Array to Flax PyTree parameters
    warning: only works for the anisotropic PIP, atom-atom type length scale parameters

    Args:
        l (Float[Array]): Array with length scale parameters
        params (PyTree): Flax PyTree parameters

    Returns:
        PyTree: Updated Flax PyTree parameters
    """
    l_base = params['params']['VmapJitPIPAniso_0']['lambda']
    l = jnp.reshape(l, l_base.shape)
    params['params']['VmapJitPIPAniso_0']['lambda'] = l
    return params