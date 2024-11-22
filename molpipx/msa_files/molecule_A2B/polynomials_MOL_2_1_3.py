import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B.monomials_MOL_2_1_3 import f_monomials as f_monos

# File created from ./MOL_2_1_3.POLY

N_POLYS = 13

# Total number of monomials = 13


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(13)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2)
    poly_2 = jnp.take(mono, 3)
    poly_3 = jnp.take(mono, 4)
    poly_4 = poly_2 * poly_1
    poly_5 = poly_1 * poly_1 - poly_3 - poly_3
    poly_6 = poly_2 * poly_2
    poly_7 = poly_2 * poly_3
    poly_8 = poly_3 * poly_1
    poly_9 = poly_2 * poly_5
    poly_10 = poly_2 * poly_4
    poly_11 = poly_1 * poly_5 - poly_8
    poly_12 = poly_2 * poly_6

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      poly_11,    poly_12,])

    return poly
