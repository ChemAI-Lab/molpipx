from typing import Any
import jax
import jax.numpy as jnp
from jax import jit

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
