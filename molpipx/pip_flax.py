from typing import Any, Callable

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

import math

from molpipx.utils import all_distances, softplus_inverse


@nn.jit
class PIP(nn.Module):
    """Permutationally Invariant Polynomials layer.

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
        bias_init: initializer function for the the l parameter
        trainable_l (bool): If True, the length scale ``l`` is optimized during training.
    """
    f_mono: Callable
    f_poly: Callable
    l: float = float(1.)
    trainable_l: bool = False
    bias_init: Callable = nn.initializers.constant

    @nn.compact
    def __call__(self, input):
        """Applies the PIP transformation to a single input geometry.

        Warning:
            This function works on a single geometry (Na x 3). Use ``PIPlayer`` for batches.

        Args:
            input (Array): Geometry array of shape (Number of atoms, 3).

        Returns:
            Array: The PIP vector.
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
    """Vectorized wrapper for ``PIP`` to handle batches of geometries.

    See ``PIP`` for more information.

    Attributes:
        f_mono (Callable): Function that returns the monomials.
        f_poly (Callable): Function that returns the polynomials.
        l (float): Initial value of the Morse variables length scale parameter.
        trainable_l (bool): If True, the length scale ``l`` is optimized.
    """
    f_mono: Callable
    f_poly: Callable
    # l: float = float(jnp.exp(1))
    l: float = math.exp(1)
    trainable_l: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Applies a vectorized map of PIP to a batch of inputs.

        Args:
            inputs (Array): Batch of geometries with shape (Batch, Number of atoms, 3).

        Returns:
            Array: PIP vectors for each geometry, shape (Batch, Number of PIPs).
        """
        vmap_pipblock = nn.vmap(PIP, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0,))(self.f_mono, self.f_poly, self.l, self.trainable_l)

        return vmap_pipblock(inputs)


@nn.jit
class EnergyPIP(nn.Module):
    """End-to-end energy model combining ``PIPlayer`` and a linear output layer.

    Attributes:
        f_mono (Callable): Function that returns the monomials.
        f_poly (Callable): Function that returns the polynomials.
        l (float): Initial value of the Morse variables length scale parameter.
        trainable_l (bool): If True, the length scale ``l`` is optimized.
    """
    f_mono: Callable
    f_poly: Callable
    # l: float = float(jnp.exp(1))
    l: float = math.exp(1)
    trainable_l: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Applies the ``PIPLayer`` and a ``nn.Dense`` modules to compute the energy.

        Args:
            inputs (Array): Batch of geometries with shape (Batch, Number of atoms, 3).

        Returns:
            Array: Energy values for each geometry, shape (Batch, 1).
        """
        vmap_pipblock = nn.vmap(PIP, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0,))(self.f_mono, self.f_poly, self.l)
        layer = nn.Dense(1, use_bias=False)

        pip = vmap_pipblock(inputs)  # computes the pip vectors
        energy = layer(pip)  # linear layer
        return energy
    
class PIPlayerGP(nn.Module):
    """Wrapper for PIPlayer to reshape the inputs before passing to PIPlayer.
    Attributes:
        f_mono (Callable): Function that returns the monomials.
        f_poly (Callable): Function that returns the polynomials.
        l (float): Initial value of the Morse variables length scale parameter.
        trainable_l (bool): If True, the length scale ``l`` is optimized.
    """

    f_mono: Callable
    f_poly: Callable
    # l: float = float(jnp.exp(1))
    l: float = math.exp(1)
    trainable_l:bool = False

    @nn.compact
    def __call__(self, inputs):
        """Reshapes flattened inputs and applies the PIP transformation.

        Args:
            inputs (Array): Geometries. Can be (Batch, Na, 3) or flattened (Batch * Na * 3).

        Returns:
            Array: PIP vectors for each geometry, shape (Batch, Number of PIPs).
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, inputs.shape[0] // 3, 3)

        pip_layer = PIPlayer(f_mono=self.f_mono, f_poly=self.f_poly, l=self.l, trainable_l = self.trainable_l)
        pip_vector = pip_layer(inputs)

        return pip_vector
