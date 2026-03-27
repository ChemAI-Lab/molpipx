import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2.monomials_MOL_2_4 import f_monomials as f_monos


# File created from /home/ravh011/molpipx/molpipx/msa_files/molecule_A2/msa_2_files/MOL_2_4.POLY


N_POLYS = 5

# Total number of monomials = 5


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(5)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = poly_1 * poly_1
    poly_3 = poly_1 * poly_2
    poly_4 = poly_1 * poly_3

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,])

    return poly
