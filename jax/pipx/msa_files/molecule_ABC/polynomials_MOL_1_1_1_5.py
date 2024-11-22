import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABC.monomials_MOL_1_1_1_5 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_5.POLY 

N_POLYS = 56

# Total number of monomials = 56 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(56) 

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
    poly_20 = poly_1 * poly_10 
    poly_21 = poly_1 * poly_14 
    poly_22 = poly_1 * poly_16 
    poly_23 = poly_1 * poly_11 
    poly_24 = poly_1 * poly_12 
    poly_25 = poly_1 * poly_18 
    poly_26 = poly_1 * poly_13 
    poly_27 = poly_2 * poly_14 
    poly_28 = poly_1 * poly_15 
    poly_29 = poly_2 * poly_16 
    poly_30 = poly_1 * poly_19 
    poly_31 = poly_2 * poly_19 
    poly_32 = poly_1 * poly_17 
    poly_33 = poly_2 * poly_18 
    poly_34 = poly_3 * poly_19 
    poly_35 = poly_1 * poly_20 
    poly_36 = poly_1 * poly_21 
    poly_37 = poly_1 * poly_27 
    poly_38 = poly_1 * poly_22 
    poly_39 = poly_1 * poly_29 
    poly_40 = poly_1 * poly_31 
    poly_41 = poly_1 * poly_23 
    poly_42 = poly_1 * poly_24 
    poly_43 = poly_1 * poly_25 
    poly_44 = poly_1 * poly_33 
    poly_45 = poly_1 * poly_26 
    poly_46 = poly_2 * poly_27 
    poly_47 = poly_1 * poly_28 
    poly_48 = poly_2 * poly_29 
    poly_49 = poly_1 * poly_30 
    poly_50 = poly_2 * poly_31 
    poly_51 = poly_1 * poly_34 
    poly_52 = poly_2 * poly_34 
    poly_53 = poly_1 * poly_32 
    poly_54 = poly_2 * poly_33 
    poly_55 = poly_3 * poly_34 

#    stack all polynomials 
    poly = jnp.stack([    poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5, 
    poly_6,    poly_7,    poly_8,    poly_9,    poly_10, 
    poly_11,    poly_12,    poly_13,    poly_14,    poly_15, 
    poly_16,    poly_17,    poly_18,    poly_19,    poly_20, 
    poly_21,    poly_22,    poly_23,    poly_24,    poly_25, 
    poly_26,    poly_27,    poly_28,    poly_29,    poly_30, 
    poly_31,    poly_32,    poly_33,    poly_34,    poly_35, 
    poly_36,    poly_37,    poly_38,    poly_39,    poly_40, 
    poly_41,    poly_42,    poly_43,    poly_44,    poly_45, 
    poly_46,    poly_47,    poly_48,    poly_49,    poly_50, 
    poly_51,    poly_52,    poly_53,    poly_54,    poly_55, 
    ]) 

    return poly 



