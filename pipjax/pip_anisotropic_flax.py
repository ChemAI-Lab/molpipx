from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax, vmap

from jaxtyping import Array, Float, PyTree

import flax
from flax import linen as nn

from pipjax.utils import all_distances, softplus_inverse, morse_variables


def get_mask(atom_types):
    pairs_ = []
    unique_pairs_ = []
    for i in range(len(atom_types)):
        for j in range(i+1, len(atom_types)):
            pair = atom_types[i] + atom_types[j]
            pair = ''.join(sorted(pair))
            pairs_.append(pair)
            # print('%s, '%pair, end='')
            if pair not in unique_pairs_:
                unique_pairs_.append(pair)

    npairs = len(pairs_)
    mask = []
    for up in unique_pairs_:
        maski = []
        for pi in pairs_:
            if up == pi:
                maski.append(1.)
            else:
                maski.append(0.)
        mask.append(maski)
    return jnp.array(mask, dtype=jnp.float32)


@nn.jit
class PIPAniso(nn.Module):
    """Permutationally Invariant Polynomials layer.

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
        bias_init: initializer function for the the l parameter
    """
    f_mono: Callable
    f_poly: Callable
    n_pairs: int = 1
    l: float = float(1.)
    bias_init: Callable = nn.initializers.constant

    @nn.compact
    def __call__(self, input, mask):
        """Applies the PIP transformation to the inputs.
        Anisotropic version, where we considered different atom-atom type length scale parameter
        (warning: this function only works for a single input)

        Args:
            inputs: geometry, array of number of atoms by 3 (Na x 3)
            mask: array with the mask for each atom-atom type length scale parameter

        Returns:
            pip_vector: pip vector
        """
        f_mono = self.f_mono
        f_poly = self.f_poly
        _lambda = self.param('lambda',
                             # Initialization function
                             self.bias_init(softplus_inverse(self.l)),
                             (self.n_pairs,))  # shape info.
        l = nn.softplus(_lambda)  # positive initialization

        d = all_distances(input)  # compute distances

        morse_ = jax.vmap(lambda li, di, maski: (li*maski) *
                          di, in_axes=(0, None, 0))(l, d, mask)
        morse = jnp.exp(-1*jnp.sum(morse_, axis=0))

        # mono = f_mono(morse)
        # compute PIP vector, morse is computed insed f_pip
        pip = f_poly(morse)
        return pip


@nn.jit
class LayerPIPAniso(nn.Module):
    """Vectorized version of ``PIP``.
    see ``PIP`` for more information.

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
    """
    f_mono: Callable
    f_poly: Callable
    n_pairs: int = 1
    l: float = float(jnp.exp(1))

    @nn.compact
    def __call__(self, inputs, mask):
        """Applies a vectorize map to PIP.apply.

        Args:
            inputs: geometries, (batch,number of atoms, 3)
            mask: mask parameters for each atom-atom type length scale parameter

        Returns:
            pip_vector: pip vector for each geometry, (batch, number of pips)
        """
        vmap_pipblock = nn.vmap(PIPAniso, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0, None))(self.f_mono, self.f_poly, self.n_pairs)

        return vmap_pipblock(inputs, mask)


@nn.jit
class EnergyPIPAniso(nn.Module):
    """Energy model using PIP."

    Attributes:
        f_mono: callable function that returns the monomials
        f_poly: callable function that returns the polynomials
        l: initial value of the morse variables length scale parameter
    """
    f_mono: Callable
    f_poly: Callable
    n_pairs: int = 1
    l: float = float(jnp.exp(1))

    @nn.compact
    def __call__(self, inputs, mask):
        """Applies the ``PIPLayer`` and a ``nn.Dense`` modules to compute the energy.

        Args:
            inputs: geometries, (batch,number of atoms, 3)
            mask: array with the mask for each atom-atom type length scale parameter

        Returns:
            energy: energy of each geometry, (batch, 1)
        """
        vmap_pipblock = nn.vmap(PIPAniso, variable_axes={'params': None, },
                                split_rngs={'params': False, },
                                in_axes=(0, None))(self.f_mono, self.f_poly, self.n_pairs)

        layer = nn.Dense(1, use_bias=False)

        pip = vmap_pipblock(inputs, mask)  # computes the pip vectors
        energy = layer(pip)  # linear layer
        return energy
