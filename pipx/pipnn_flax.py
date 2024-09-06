from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jaxtyping import Array, Float, PyTree

import flax
from flax import linen as nn

from pipx.utils import all_distances, softplus_inverse, morse_variables
from pipx.pip_flax import PIPlayer


@nn.jit
class MLP(nn.Module):
    features: Tuple[int]
    act_fun: Callable = nn.tanh

    def setup(self):
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(1)

    @nn.compact
    def __call__(self, x):

        z = x
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = self.act_fun(z)

        return self.last_layer(z)


@nn.jit
class PIPNN(nn.Module):
    f_mono: Callable
    f_poly: Callable
    features: Tuple[int]  # fetures per layer in Tuple
    l: float = float(1.)
    act_fun: Callable = nn.tanh  # lambda x:  f(x)

    def setup(self):
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(1)
        self.pip_layer = PIPlayer(self.f_mono, self.f_poly, self.l)

    @nn.compact
    def __call__(self, x):

        z = self.pip_layer(x)
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = self.act_fun(z)

        return self.last_layer(z)
