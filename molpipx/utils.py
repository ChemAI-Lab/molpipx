from typing import Any
import jax
import jax.numpy as jnp
from jax import jit
from optax.losses import squared_error

from jaxtyping import Array, Float, PyTree


@jit
def all_distances(xi: Float[Array, "..."]) -> Float[Array, "..."]:
    """Computes all pairwise distances between atoms in a molecule.

    Calculates the Euclidean distance (L2 norm) between every pair of atoms in the
    input geometry. It returns the upper triangular part of the distance matrix
    in lexicographical order.

    Args:
        xi (Array): Cartesian coordinates of the atoms, shape (N_atoms, 3).

    Returns:
        Array: A flattened array containing all unique pairwise distances.
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
    """Computes the inverse of the softplus function.

    Args:
        x (Array): Input value (must be positive).

    Returns:
        Array: The inverse softplus of the input.
    """
    return jnp.log(jnp.exp(x) - 1)


@jit
def morse_variables(x: Float[Array, "dim1"], l: Float[Array, ""]) -> Float[Array, "dim1"]:
    """Computes Morse-like variables using a single length scale parameter.

    Args:
        x (Array): Cartesian coordinates of the atoms (N_atoms, 3).
        l (float): Length scale parameter (decay rate).

    Returns:
        Array: The computed Morse variables for all pairwise distances.
    """
    r = all_distances(x)
    return jnp.exp(-l*r)


@jit
def mse_loss(predictions: Float[Array, "dim1"], targets: Float[Array, "dim1"]) -> Float:
    """Computes the Mean Squared Error (MSE) loss.

    Args:
        predictions (Array): The predicted values.
        targets (Array): The ground truth values.

    Returns:
        Float: The mean squared error.
    """
    return jnp.mean(squared_error(predictions-targets))


@jit
def mae_loss(predictions: Float[Array, "dim1"], targets: Float[Array, "dim1"]) -> Float:
    """Computes the Mean Absolute Error (MAE) loss.

    Args:
        predictions (Array): The predicted values.
        targets (Array): The ground truth values.

    Returns:
        Float: The mean absolute error.
    """
    return jnp.mean(jnp.abs(predictions-targets))


def flax_params(w: Float[Array, '...'], params: PyTree) -> PyTree:
    """Updates the weights of the first Dense layer in a Flax PyTree.

    Args:
        w (Array): Array containing the new linear weights (e.g., from a least-squares solution).
        params (PyTree): The existing Flax parameter PyTree.

    Returns:
        PyTree: The updated parameter PyTree.
    """

    w_base = params['params']['Dense_0']['kernel']
    w = jnp.reshape(w, w_base.shape)
    params['params']['Dense_0']['kernel'] = w
    return params


def flax_params_aniso(l: Float[Array, '...'], params: PyTree) -> PyTree:
    """Updates the length scale parameters for an Anisotropic PIP model.

    Warning:
        This function assumes a specific Flax model structure (Anisotropic PIP).

    Args:
        l (Array): Array containing the new length scale parameters.
        params (PyTree): The existing Flax parameter PyTree.

    Returns:
        PyTree: The updated parameter PyTree.
    """
    l_base = params['params']['VmapJitPIPAniso_0']['lambda']
    l = jnp.reshape(l, l_base.shape)
    params['params']['VmapJitPIPAniso_0']['lambda'] = l
    return params