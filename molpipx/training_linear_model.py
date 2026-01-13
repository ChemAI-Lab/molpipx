from typing import Any, Callable
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from molpipx.utils_gradients import get_pip_grad


def training(model_pip: Callable,
             X_tr: Float[Array, "..."],
             y_tr: Float[Array, "..."]) -> Float[Array, "..."]:
    """Computes optimal linear parameters for a PIP model using energy data.

    Solves the linear system of equations ``PIP * theta = Energy`` using least squares
    to find the optimal coefficients (``theta``) for the polynomial expansion.

    Warning:
        Geometries must be provided in **Bohr Units**.

    Args:
        model_pip (Callable): A Flax module instance initialized to compute PIP vectors.
        X_tr (Array): Training geometries with shape (Batch, N_atoms, 3).
        y_tr (Array): Training energies with shape (Batch, 1).

    Returns:
        Array: The optimal linear parameters (theta) for the model.
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
        model_pip (Callable): A Flax module instance initialized to compute PIP vectors.
        X_tr (Array): Training geometries with shape (Batch, N_atoms, 3).
        F_tr (Array): Training forces with shape (Batch, N_atoms, 3).
        y_tr (Array): Training energies with shape (Batch, 1).

    Returns:
        Array: The optimal linear parameters (theta) for the model.
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
