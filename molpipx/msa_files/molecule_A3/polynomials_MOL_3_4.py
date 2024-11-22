import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A3.monomials_MOL_3_4 import f_monomials as f_monos

# File created from ./MOL_3_4.POLY

N_POLYS = 11

# Total number of monomials = 11


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(11)

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

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      ])

    return poly
