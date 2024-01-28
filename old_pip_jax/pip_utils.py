from typing import Any, Callable


import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

import flax
from flax import linen as nn


@jit
def all_distances(xi: Any):
    n_atoms = xi.shape[0]
    z = xi[:, None] - xi[None, :]  # compute all difference
    i0 = jnp.triu_indices(
        n_atoms, 1
    )  # select upper diagonal (LEXIC ORDER)
    diff = z[i0]
    r = jnp.linalg.norm(diff, axis=1)  # compute the bond length
    return r


@jit
def morse_variables(x: Any, l: any):
    r = all_distances(x)
    return jnp.exp(-l*r)


def softplus_inverse(x: Any):
    return jnp.log(jnp.exp(x) - 1)


@nn.jit
class PIPi(nn.Module):
    f_mono: Callable
    f_poly: Callable
    l: float = float(1.)
    bias_init: Callable = nn.initializers.constant

    @nn.compact
    def __call__(self, inputs):
        f_mono = self.f_mono
        f_poly = self.f_poly
        _lambda = self.param('lambda',
                             # Initialization function
                             self.bias_init(softplus_inverse(self.l)),
                             (1,))  # shape info.
        l = nn.softplus(_lambda)  # positive initialization

        d = all_distances(inputs)
        morse = jnp.exp(-l*d)

        # mono = f_mono(morse)
        pip = f_poly(morse)
        return pip


@nn.jit
class PIP(nn.Module):
    f_mono: Callable
    f_poly: Callable
    l: float = float(jnp.exp(1))

    @nn.compact
    def __call__(self, inputs):
        vmap_pipblock = nn.vmap(PIPi, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0,))(self.f_mono, self.f_poly)

        return vmap_pipblock(inputs)


@nn.jit
class PIPLinear(nn.Module):
    f_mono: Callable
    f_poly: Callable
    l: float = float(jnp.exp(1))

    @nn.compact
    def __call__(self, inputs):
        vmap_pipblock = nn.vmap(PIPi, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0,))(self.f_mono, self.f_poly)
        layer = nn.Dense(1, use_bias=False)
        pip = vmap_pipblock(inputs)
        # e = layer(pip)
        return layer(pip)
