from typing import Any, Callable

import jax.numpy as jnp
import jax.random as jrnd

from jaxtyping import Key, Float, Array


def split_train_and_test_data(Geometries: Float[Array, "..."], Energies: Float[Array, "..."],
                              N: int, key: Key, Nval: int = 0,
                              ) -> ((Float, Float), (Float, Float)):
    """Splits geometry and energy datasets into training and validation sets.

    Args:
        Geometries (Array): The complete dataset of geometries (Batch, Na, 3).
        Energies (Array): The complete dataset of corresponding energies (Batch, 1).
        N (int): The number of samples to include in the training set.
        key (Key): A JAX random key used to shuffle the data.
        Nval (int, optional): The number of samples to include in the validation set. 
            If 0 or None, the remaining samples after selecting ``N`` are used. Defaults to 0.

    Returns:
        Tuple: Two tuples containing the split data:
            * **Train**: ``(X_tr, y_tr)``
            * **Validation**: ``(X_val, y_val)``
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
    """Splits geometry, force, and energy datasets into training and validation sets.

    Args:
        Geometries (Array): The complete dataset of geometries (Batch, Na, 3).
        Forces (Array): The complete dataset of corresponding forces (Batch, Na, 3).
        Energies (Array): The complete dataset of corresponding energies (Batch, 1).
        N (int): The number of samples to include in the training set.
        key (Key): A JAX random key used to shuffle the data.
        Nval (int, optional): The number of samples to include in the validation set.
            If 0 or None, the remaining samples are used. Defaults to 0.

    Returns:
        Tuple: Two tuples containing the split data:
            * **Train**: ``(X_tr, G_tr, y_tr)``
            * **Validation**: ``(X_val, G_val, y_val)``
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
