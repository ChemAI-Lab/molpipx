import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2.monomials_MOL_2_2_3 import f_monomials as f_monos

# File created from ./MOL_2_2_3.POLY

N_POLYS = 33

# Total number of monomials = 33


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(33)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = jnp.take(mono, 2) + jnp.take(mono, 3) + \
        jnp.take(mono, 4) + jnp.take(mono, 5)
    poly_3 = jnp.take(mono, 6)
    poly_4 = poly_1 * poly_2
    poly_5 = jnp.take(mono, 7) + jnp.take(mono, 8)
    poly_6 = jnp.take(mono, 9) + jnp.take(mono, 10)
    poly_7 = jnp.take(mono, 11) + jnp.take(mono, 12)
    poly_8 = poly_1 * poly_3
    poly_9 = poly_3 * poly_2
    poly_10 = poly_1 * poly_1
    poly_11 = poly_2 * poly_2 - poly_7 - poly_6 - poly_5 - poly_7 - poly_6 - poly_5
    poly_12 = poly_3 * poly_3
    poly_13 = poly_1 * poly_5
    poly_14 = poly_1 * poly_6
    poly_15 = poly_1 * poly_7
    poly_16 = jnp.take(mono, 13) + jnp.take(mono, 14) + \
        jnp.take(mono, 15) + jnp.take(mono, 16)
    poly_17 = poly_1 * poly_9
    poly_18 = poly_3 * poly_5
    poly_19 = poly_3 * poly_6
    poly_20 = poly_3 * poly_7
    poly_21 = poly_1 * poly_4
    poly_22 = poly_1 * poly_11
    poly_23 = poly_2 * poly_5 - poly_16
    poly_24 = poly_2 * poly_6 - poly_16
    poly_25 = poly_2 * poly_7 - poly_16
    poly_26 = poly_1 * poly_8
    poly_27 = poly_3 * poly_11
    poly_28 = poly_1 * poly_12
    poly_29 = poly_3 * poly_9
    poly_30 = poly_1 * poly_10
    poly_31 = poly_2 * poly_11 - poly_25 - poly_24 - poly_23
    poly_32 = poly_3 * poly_12

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      poly_11,    poly_12,    poly_13,    poly_14,    poly_15,
                      poly_16,    poly_17,    poly_18,    poly_19,    poly_20,
                      poly_21,    poly_22,    poly_23,    poly_24,    poly_25,
                      poly_26,    poly_27,    poly_28,    poly_29,    poly_30,
                      poly_31,    poly_32,])

    return poly
