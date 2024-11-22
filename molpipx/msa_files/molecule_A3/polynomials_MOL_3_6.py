import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A3.monomials_MOL_3_6 import f_monomials as f_monos

# File created from ./MOL_3_6.POLY

N_POLYS = 23

# Total number of monomials = 23


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(23)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2) + jnp.take(mono, 3)
    poly_2 = jnp.take(mono, 4) + jnp.take(mono, 5) + jnp.take(mono, 6)
    poly_3 = poly_1 * poly_1 - poly_2 - poly_2
    poly_4 = jnp.take(mono, 7)
    poly_5 = poly_1 * poly_2 - poly_4 - poly_4 - poly_4
    poly_6 = poly_1 * poly_3 - poly_5
    poly_7 = poly_4 * poly_1
    poly_8 = poly_2 * poly_2 - poly_7 - poly_7
    poly_9 = poly_2 * poly_3 - poly_7
    poly_10 = poly_1 * poly_6 - poly_9
    poly_11 = poly_4 * poly_2
    poly_12 = poly_4 * poly_3
    poly_13 = poly_1 * poly_8 - poly_11
    poly_14 = poly_2 * poly_6 - poly_12
    poly_15 = poly_1 * poly_10 - poly_14
    poly_16 = poly_4 * poly_4
    poly_17 = poly_4 * poly_5
    poly_18 = poly_4 * poly_6
    poly_19 = poly_2 * poly_8 - poly_17
    poly_20 = poly_1 * poly_13 - poly_17 - poly_19 - poly_19
    poly_21 = poly_2 * poly_10 - poly_18
    poly_22 = poly_1 * poly_15 - poly_21

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      poly_11,    poly_12,    poly_13,    poly_14,    poly_15,
                      poly_16,    poly_17,    poly_18,    poly_19,    poly_20,
                      poly_21,    poly_22,])

    return poly
