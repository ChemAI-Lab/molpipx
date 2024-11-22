import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A4.monomials_MOL_4_4 import f_monomials as f_monos

# File created from ./MOL_4_4.POLY

N_POLYS = 22

# Total number of monomials = 22


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(22)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2) + jnp.take(mono, 3) + \
        jnp.take(mono, 4) + jnp.take(mono, 5) + jnp.take(mono, 6)
    poly_2 = jnp.take(mono, 7) + jnp.take(mono, 8) + jnp.take(mono, 9)
    poly_3 = jnp.take(mono, 10) + jnp.take(mono, 11) + jnp.take(mono, 12) + jnp.take(mono, 13) + jnp.take(mono, 14) + jnp.take(mono, 15) + \
        jnp.take(mono, 16) + jnp.take(mono, 17) + jnp.take(mono, 18) + \
        jnp.take(mono, 19) + jnp.take(mono, 20) + jnp.take(mono, 21)
    poly_4 = poly_1 * poly_1 - poly_3 - poly_2 - poly_3 - poly_2
    poly_5 = jnp.take(mono, 22) + jnp.take(mono, 23) + jnp.take(mono, 24) + jnp.take(mono, 25) + jnp.take(mono, 26) + jnp.take(mono, 27) + \
        jnp.take(mono, 28) + jnp.take(mono, 29) + jnp.take(mono, 30) + \
        jnp.take(mono, 31) + jnp.take(mono, 32) + jnp.take(mono, 33)
    poly_6 = jnp.take(mono, 34) + jnp.take(mono, 35) + \
        jnp.take(mono, 36) + jnp.take(mono, 37)
    poly_7 = jnp.take(mono, 38) + jnp.take(mono, 39) + \
        jnp.take(mono, 40) + jnp.take(mono, 41)
    poly_8 = poly_1 * poly_2 - poly_5
    poly_9 = poly_1 * poly_3 - poly_6 - poly_7 - \
        poly_5 - poly_6 - poly_7 - poly_5 - poly_6 - poly_7
    poly_10 = poly_1 * poly_4 - poly_9 - poly_8
    poly_11 = jnp.take(mono, 42) + jnp.take(mono, 43) + jnp.take(mono, 44)
    poly_12 = jnp.take(mono, 45) + jnp.take(mono, 46) + jnp.take(mono, 47) + jnp.take(mono, 48) + jnp.take(mono, 49) + jnp.take(mono, 50) + \
        jnp.take(mono, 51) + jnp.take(mono, 52) + jnp.take(mono, 53) + \
        jnp.take(mono, 54) + jnp.take(mono, 55) + jnp.take(mono, 56)
    poly_13 = poly_2 * poly_3 - poly_12
    poly_14 = poly_1 * poly_5 - poly_12 - poly_11 - \
        poly_13 - poly_12 - poly_11 - poly_11 - poly_11
    poly_15 = poly_1 * poly_6 - poly_12
    poly_16 = poly_1 * poly_7 - poly_12
    poly_17 = poly_2 * poly_2 - poly_11 - poly_11
    poly_18 = poly_3 * poly_3 - poly_12 - poly_11 - poly_15 - poly_16 - poly_14 - \
        poly_12 - poly_11 - poly_15 - poly_16 - \
        poly_14 - poly_12 - poly_11 - poly_12 - poly_11
    poly_19 = poly_2 * poly_4 - poly_14
    poly_20 = poly_3 * poly_4 - poly_15 - poly_16 - poly_13
    poly_21 = poly_1 * poly_10 - poly_20 - poly_19

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      poly_11,    poly_12,    poly_13,    poly_14,    poly_15,
                      poly_16,    poly_17,    poly_18,    poly_19,    poly_20,
                      poly_21,])

    return poly
