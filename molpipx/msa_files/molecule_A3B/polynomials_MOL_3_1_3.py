import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A3B.monomials_MOL_3_1_3 import f_monomials as f_monos

# File created from ./MOL_3_1_3.POLY

N_POLYS = 23

# Total number of monomials = 23


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(23)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2) + jnp.take(mono, 3)
    poly_2 = jnp.take(mono, 4) + jnp.take(mono, 5) + jnp.take(mono, 6)
    poly_3 = jnp.take(mono, 7) + jnp.take(mono, 8) + jnp.take(mono, 9)
    poly_4 = jnp.take(mono, 10) + jnp.take(mono, 11) + jnp.take(mono, 12)
    poly_5 = poly_1 * poly_2 - poly_4
    poly_6 = jnp.take(mono, 13) + jnp.take(mono, 14) + jnp.take(mono, 15)
    poly_7 = poly_1 * poly_1 - poly_3 - poly_3
    poly_8 = poly_2 * poly_2 - poly_6 - poly_6
    poly_9 = jnp.take(mono, 16)
    poly_10 = jnp.take(mono, 17) + jnp.take(mono, 18) + jnp.take(mono, 19) + \
        jnp.take(mono, 20) + jnp.take(mono, 21) + jnp.take(mono, 22)
    poly_11 = poly_2 * poly_3 - poly_10
    poly_12 = jnp.take(mono, 23) + jnp.take(mono, 24) + jnp.take(mono, 25) + \
        jnp.take(mono, 26) + jnp.take(mono, 27) + jnp.take(mono, 28)
    poly_13 = jnp.take(mono, 29)
    poly_14 = poly_1 * poly_6 - poly_12
    poly_15 = poly_1 * poly_3 - poly_9 - poly_9 - poly_9
    poly_16 = poly_1 * poly_4 - poly_10
    poly_17 = poly_2 * poly_7 - poly_16
    poly_18 = poly_2 * poly_4 - poly_12
    poly_19 = poly_1 * poly_8 - poly_18
    poly_20 = poly_2 * poly_6 - poly_13 - poly_13 - poly_13
    poly_21 = poly_1 * poly_7 - poly_15
    poly_22 = poly_2 * poly_8 - poly_20

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      poly_11,    poly_12,    poly_13,    poly_14,    poly_15,
                      poly_16,    poly_17,    poly_18,    poly_19,    poly_20,
                      poly_21,    poly_22,])

    return poly
