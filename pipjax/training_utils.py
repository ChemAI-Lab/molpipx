from typing import Any, Callable
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap, jacrev, lax


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
