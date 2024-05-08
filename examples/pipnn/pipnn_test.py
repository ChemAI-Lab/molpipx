from typing import Tuple, List, Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jrnd


from pipjax import split_train_and_test_data
from pipjax import PIPNN
from pipjax import flax_params_aniso, flax_params
from pipjax import mse_loss, softplus_inverse
from pipjax import get_functions

from jaxtyping import (
    Array,
    Float,
    PyTree,
    install_import_hook,
)

import flax
from flax import linen as nn
import optax
from optax.tree_utils import tree_l2_norm as l2_norm


def trainig():
    
    