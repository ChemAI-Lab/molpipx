import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABC.monomials_MOL_1_1_1_3 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_3.POLY 

N_POLYS = 20

# Total number of monomials = 20 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(20) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = poly_1 * poly_2 
    poly_5 = poly_1 * poly_3 
    poly_6 = poly_2 * poly_3 
    poly_7 = poly_1 * poly_1 
    poly_8 = poly_2 * poly_2 
    poly_9 = poly_3 * poly_3 
    poly_10 = poly_1 * poly_6 
    poly_11 = poly_1 * poly_4 
    poly_12 = poly_1 * poly_8 
    poly_13 = poly_1 * poly_5 
    poly_14 = poly_2 * poly_6 
    poly_15 = poly_1 * poly_9 
    poly_16 = poly_2 * poly_9 
    poly_17 = poly_1 * poly_7 
    poly_18 = poly_2 * poly_8 
    poly_19 = poly_3 * poly_9 

#    stack all polynomials 
    poly = jnp.stack([    poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5, 
    poly_6,    poly_7,    poly_8,    poly_9,    poly_10, 
    poly_11,    poly_12,    poly_13,    poly_14,    poly_15, 
    poly_16,    poly_17,    poly_18,    poly_19,    ]) 

    return poly 



