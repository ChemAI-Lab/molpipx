from typing import Any, Callable

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

from molpipx.utils import all_distances, softplus_inverse


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
    trainable_l: bool = False
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

        if self.trainable_l:
            _lambda = self.param('lambda', self.bias_init(softplus_inverse(self.l)), (1,))
            l = nn.softplus(_lambda)
        else:
            l = jnp.ones(1)

        d = all_distances(input)  # compute distances
        morse = jnp.exp(-l*d)  # comput morse variables

        # compute PIP vector, morse is computed inside f_pip
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
    trainable_l: bool = False

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
                                in_axes=(0,))(self.f_mono, self.f_poly, self.l, self.trainable_l)

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
    trainable_l: bool = False

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
                                in_axes=(0,))(self.f_mono, self.f_poly, self.l)
        layer = nn.Dense(1, use_bias=False)

        pip = vmap_pipblock(inputs)  # computes the pip vectors
        energy = layer(pip)  # linear layer
        return energy
    
class PIPlayerGP(nn.Module):
    """Wrapper for PIPlayer to reshape the inputs before passing to PIPlayer."""

    f_mono: Callable
    f_poly: Callable
    l: float = float(jnp.exp(1))
    trainable_l:bool = False

    @nn.compact
    def __call__(self, inputs):
        """Reshape inputs and apply PIPlayer.

        Args:
            inputs: geometries, can be either (batch, number of atoms, 3) or (batch * number of atoms * 3)

        Returns:
            pip_vector: pip vector for each geometry, (batch, number of pips)
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, inputs.shape[0] // 3, 3)

        pip_layer = PIPlayer(f_mono=self.f_mono, f_poly=self.f_poly, l=self.l, trainable_l = self.trainable_l)
        pip_vector = pip_layer(inputs)

        return pip_vector
