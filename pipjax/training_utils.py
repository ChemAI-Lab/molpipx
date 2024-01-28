from typing import Any, Callable
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap, jacrev, lax


def get_pip_grad(model_pip: Callable, x: Any, params_pip: Any):

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


def get_forces(model: Callable, x: Any, params: Any):

    @jit
    def grad_forces_rev_i(xyzi: Any):
        g_forces = jacrev(model, argnums=(1,))(params,
                                               lax.expand_dims(xyzi, dimensions=(0,)))
        g_forces = lax.squeeze(
            g_forces[0], dimensions=(0, 1, 2))  # size = (natoms, 3)
        return g_forces

    return vmap(grad_forces_rev_i, in_axes=(0,))(x)


def split_train_and_test_data(Geometries: Any, Energies: Any,
                              N: int, key: Any,
                              ):

    _, key = jrnd.split(key)
    ni = jnp.arange(Energies.shape[0], dtype=jnp.int32)
    i0 = jrnd.permutation(key, ni)
    i0_tr = i0[:N]
    i0_tst = i0[N:]

    X_tr = Geometries[i0_tr]
    y_tr = Energies[i0_tr]

    X_tst = Geometries[i0_tst]
    y_tst = Energies[i0_tst]

    return (X_tr, y_tr), (X_tst, y_tst)


def split_train_and_test_data_w_forces(Geometries: Any, Forces: Any, Energies: Any,
                                       N: int, key: Any,
                                       ):

    _, key = jrnd.split(key)
    ni = jnp.arange(Energies.shape[0], dtype=jnp.int32)
    i0 = jrnd.permutation(key, ni)
    i0_tr = i0[:N]
    i0_tst = i0[N:]

    X_tr = Geometries[i0_tr]
    G_tr = Forces[i0_tr]
    y_tr = Energies[i0_tr]

    X_tst = Geometries[i0_tst]
    G_tst = Forces[i0_tst]
    y_tst = Energies[i0_tst]

    return (X_tr, G_tr, y_tr), (X_tst, G_tst, y_tst)
