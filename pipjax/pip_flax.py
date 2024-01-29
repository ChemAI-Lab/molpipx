from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jaxtyping import Array, Float, PyTree

import flax
from flax import linen as nn

from pipjax.utils import all_distances, softplus_inverse, morse_variables


@nn.jit
class PIP(nn.Module):
    """Permutationally Invariant Polynomials layer.

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
        bias_init: initializer function for the the l parameter
    """
    f_mono: Callable
    f_poly: Callable
    l: float = float(1.)
    bias_init: Callable = nn.initializers.constant

    @nn.compact
    def __call__(self, input):
        """Applies the PIP transformation to the inputs.
        (warning: this function only works for a single input)

        Args:
            inputs: geometry, array of number of atoms by 3 (Na x 3)

        Returns:
            pip_vector: pip vector
        """
        f_mono = self.f_mono
        f_poly = self.f_poly
        _lambda = self.param('lambda',
                             # Initialization function
                             self.bias_init(softplus_inverse(self.l)),
                             (1,))  # shape info.
        l = nn.softplus(_lambda)  # positive initialization

        d = all_distances(input)  # compute distances
        morse = jnp.exp(-l*d)  # comput morse variables

        # mono = f_mono(morse)
        # compute PIP vector, morse is computed insed f_pip
        pip = f_poly(morse)
        return pip


@nn.jit
class PIPlayer(nn.Module):
    """Vectorize version of ``PIP``.
    see ``PIP`` for more information.

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
    """
    f_mono: Callable
    f_poly: Callable
    l: float = float(jnp.exp(1))

    @nn.compact
    def __call__(self, inputs):
        """Applies a vectorize map to PIP.apply.

        Args:
            inputs: geometries, (batch,number of atoms, 3)

        Returns:
            pip_vector: pip vector for each geometry, (batch, number of pips)
        """
        vmap_pipblock = nn.vmap(PIP, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0,))(self.f_mono, self.f_poly)

        return vmap_pipblock(inputs)


@nn.jit
class EnergyPIP(nn.Module):
    """Energy model using PIP."

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
    """
    f_mono: Callable
    f_poly: Callable
    l: float = float(jnp.exp(1))

    @nn.compact
    def __call__(self, inputs):
        """Applies the ``PIPLayer`` and a ``nn.Dense`` modules to compute the energy.

        Args:
            inputs: geometries, (batch,number of atoms, 3)

        Returns:
            energy: energy of each geometry, (batch, 1)
        """
        vmap_pipblock = nn.vmap(PIP, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0,))(self.f_mono, self.f_poly)
        layer = nn.Dense(1, use_bias=False)

        pip = vmap_pipblock(inputs)  # computes the pip vectors
        energy = layer(pip)  # linear layer
        return energy
