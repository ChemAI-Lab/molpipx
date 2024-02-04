from typing import Any, Callable
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap, jacrev, lax

from jaxtyping import Key, Float, Array


def split_train_and_test_data(Geometries: Float[Array, "..."], Energies: Float[Array, "..."],
                              N: int, key: Key, Nval: int = 0,
                              ) -> ((Float, Float), (Float, Float)):
    """Split the data into training and validation data.

    Args:
        Geometries:
        Energies:
        N: number of training data
        key: Key to split the training
        Nval: number of validation data

    Returns:
        Tuple: (Training geometries and energy), (Validation geometries and energy)
    """

    _, key = jrnd.split(key)
    ni = jnp.arange(Energies.shape[0], dtype=jnp.int32)
    i0 = jrnd.permutation(key, ni)
    i0_tr = i0[:N]
    i0_tst = i0[N:]

    X_tr = Geometries[i0_tr]
    y_tr = Energies[i0_tr]

    if Nval == None or int(Nval) == 0:
        X_tst = Geometries[i0_tst]
        y_tst = Energies[i0_tst]
    elif Nval > 0:
        i0_val = i0_tst[:Nval]
        X_tst = Geometries[i0_val]
        y_tst = Energies[i0_val]

    return (X_tr, y_tr), (X_tst, y_tst)


def split_train_and_test_data_w_forces(Geometries: Float[Array, "..."], Forces: Float[Array, "..."],
                                       Energies: Float[Array, "..."],
                                       N: int, key: Key, Nval: int = 0,
                                       ) -> ((Float, Float, Float), (Float, Float, Float)):
    """Split the data into training and validation.

    Args:
        Geometries:
        Forces:
        Energies:
        N: number of training data
        key: Key to split the training
        Nval: number of validation data
    Returns:
        Tuple: (Training geometries, forces and energy), (Validation geometries, forces and energy)
    """

    _, key = jrnd.split(key)
    ni = jnp.arange(Energies.shape[0], dtype=jnp.int32)
    i0 = jrnd.permutation(key, ni)
    i0_tr = i0[:N]
    i0_tst = i0[N:]

    X_tr = Geometries[i0_tr]
    G_tr = Forces[i0_tr]
    y_tr = Energies[i0_tr]

    if Nval == None or int(Nval) == 0:
        X_tst = Geometries[i0_tst]
        G_tst = Forces[i0_tst]
        y_tst = Energies[i0_tst]
    elif Nval > 0:
        i0_val = i0_tst[:Nval]
        X_tst = Geometries[i0_val]
        G_tst = Forces[i0_val]
        y_tst = Energies[i0_val]

    return (X_tr, G_tr, y_tr), (X_tst, G_tst, y_tst)
