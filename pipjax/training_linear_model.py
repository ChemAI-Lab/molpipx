from typing import Callable
import jax
import jax.numpy as jnp
from jax import lax, jit
from jaxtyping import Array, Float, PyTree, Key

from pipjax.utils_gradients import get_pip_grad


def training(model_pip: Callable,
             X_tr: Float[Array, "..."],
             y_tr: Float[Array, "..."]) -> Float[Array, "..."]:
    """Simple training function for PIP models.
    Warning: Geometries must be in Bohr Units

    Args:
        model_pip: Callable, Flax class to compute the PIP vectors
        X_tr: Training Geometries, (batch,number of atoms, 3) 
        y_tr: Training Energies, (batch,1)

    Returns:
        Array: Optimal linear parameters
    """
    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)

    xyz0 = X_tr[0]  # single point to initialize the PIP model
    params_pip = model_pip.init(key, xyz0[jnp.newaxis])

    # PIP matrix training
    Pip_tr = model_pip.apply(params_pip, X_tr)

    # Solving the linear system of equations  PIP x = Energy
    results = jnp.linalg.lstsq(Pip_tr, y_tr)

    theta = results[0]
    return theta


def training_w_gradients(model_pip: Callable,
                         X_tr: Float[Array, "..."],
                         F_tr: Float[Array, "..."],
                         y_tr: Float[Array, "..."]) -> Float[Array, "..."]:
    """Simple training function for PIP models with Forces.
    Warning: Geometries must be in Bohr Units and Forces in Ha/Bohr Units

    Args:
        model_pip: Callable, Flax class to compute the PIP vectors
        X_tr: Training Geometries, (batch,number of atoms, 3) 
        F_tr: Training Forces, (batch, number of atoms, 3)
        y_tr: Training Energies, (batch,1)

    Returns:
       Array: Optimal linear parameters
    """

    n, n_atoms, _ = X_tr.shape

    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)

    xyz0 = X_tr[0]  # single point to initialize the PIP model
    params_pip = model_pip.init(key, xyz0[jnp.newaxis])
    x0_pip = model_pip.apply(params_pip, xyz0[jnp.newaxis])
    n_pip = x0_pip.shape[1]

    # Training PIP matrix
    Pip_tr = model_pip.apply(params_pip, X_tr)
    GPIP_tr = get_pip_grad(model_pip.apply, X_tr, params_pip)

    # (bs*n_atoms*3,number of pip)
    GPIP_tr_w_grad = GPIP_tr.reshape(n*n_atoms*3, n_pip)
    Pip_tr_w_grad_full = lax.concatenate((Pip_tr, GPIP_tr_w_grad), dimension=0)

    y_tr_w_grad_full = lax.concatenate(
        (y_tr, F_tr.reshape(n*n_atoms*3, 1)), dimension=0)
    # ---------

    # Optimization
    results_w_grad = jnp.linalg.lstsq(Pip_tr_w_grad_full, y_tr_w_grad_full)
    theta = results_w_grad[0]
    return theta
