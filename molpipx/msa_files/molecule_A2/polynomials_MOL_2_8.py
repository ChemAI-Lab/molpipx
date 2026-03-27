import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2.monomials_MOL_2_8 import f_monomials as f_monos


# File created from /home/ravh011/molpipx/molpipx/msa_files/molecule_A2/msa_2_files/MOL_2_8.POLY


N_POLYS = 9

# Total number of monomials = 9


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(9)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = poly_1 * poly_1
    poly_3 = poly_1 * poly_2
    poly_4 = poly_1 * poly_3
    poly_5 = poly_1 * poly_4
    poly_6 = poly_1 * poly_5
    poly_7 = poly_1 * poly_6
    poly_8 = poly_1 * poly_7

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,])

    return poly
