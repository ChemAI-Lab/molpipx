import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABCD.monomials_MOL_1_1_1_1_3 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_1_3.POLY 

N_POLYS = 84

# Total number of monomials = 84 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(84) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = jnp.take(mono,4) 
    poly_5 = jnp.take(mono,5) 
    poly_6 = jnp.take(mono,6) 
    poly_7 = poly_1 * poly_2 
    poly_8 = poly_1 * poly_3 
    poly_9 = poly_2 * poly_3 
    poly_10 = poly_1 * poly_4 
    poly_11 = poly_2 * poly_4 
    poly_12 = poly_3 * poly_4 
    poly_13 = poly_1 * poly_5 
    poly_14 = poly_2 * poly_5 
    poly_15 = poly_3 * poly_5 
    poly_16 = poly_4 * poly_5 
    poly_17 = poly_1 * poly_6 
    poly_18 = poly_2 * poly_6 
    poly_19 = poly_3 * poly_6 
    poly_20 = poly_4 * poly_6 
    poly_21 = poly_5 * poly_6 
    poly_22 = poly_1 * poly_1 
    poly_23 = poly_2 * poly_2 
    poly_24 = poly_3 * poly_3 
    poly_25 = poly_4 * poly_4 
    poly_26 = poly_5 * poly_5 
    poly_27 = poly_6 * poly_6 
    poly_28 = poly_1 * poly_9 
    poly_29 = poly_1 * poly_11 
    poly_30 = poly_1 * poly_12 
    poly_31 = poly_2 * poly_12 
    poly_32 = poly_1 * poly_14 
    poly_33 = poly_1 * poly_15 
    poly_34 = poly_2 * poly_15 
    poly_35 = poly_1 * poly_16 
    poly_36 = poly_2 * poly_16 
    poly_37 = poly_3 * poly_16 
    poly_38 = poly_1 * poly_18 
    poly_39 = poly_1 * poly_19 
    poly_40 = poly_2 * poly_19 
    poly_41 = poly_1 * poly_20 
    poly_42 = poly_2 * poly_20 
    poly_43 = poly_3 * poly_20 
    poly_44 = poly_1 * poly_21 
    poly_45 = poly_2 * poly_21 
    poly_46 = poly_3 * poly_21 
    poly_47 = poly_4 * poly_21 
    poly_48 = poly_1 * poly_7 
    poly_49 = poly_1 * poly_23 
    poly_50 = poly_1 * poly_8 
    poly_51 = poly_2 * poly_9 
    poly_52 = poly_1 * poly_24 
    poly_53 = poly_2 * poly_24 
    poly_54 = poly_1 * poly_10 
    poly_55 = poly_2 * poly_11 
    poly_56 = poly_3 * poly_12 
    poly_57 = poly_1 * poly_25 
    poly_58 = poly_2 * poly_25 
    poly_59 = poly_3 * poly_25 
    poly_60 = poly_1 * poly_13 
    poly_61 = poly_2 * poly_14 
    poly_62 = poly_3 * poly_15 
    poly_63 = poly_4 * poly_16 
    poly_64 = poly_1 * poly_26 
    poly_65 = poly_2 * poly_26 
    poly_66 = poly_3 * poly_26 
    poly_67 = poly_4 * poly_26 
    poly_68 = poly_1 * poly_17 
    poly_69 = poly_2 * poly_18 
    poly_70 = poly_3 * poly_19 
    poly_71 = poly_4 * poly_20 
    poly_72 = poly_5 * poly_21 
    poly_73 = poly_1 * poly_27 
    poly_74 = poly_2 * poly_27 
    poly_75 = poly_3 * poly_27 
    poly_76 = poly_4 * poly_27 
    poly_77 = poly_5 * poly_27 
    poly_78 = poly_1 * poly_22 
    poly_79 = poly_2 * poly_23 
    poly_80 = poly_3 * poly_24 
    poly_81 = poly_4 * poly_25 
    poly_82 = poly_5 * poly_26 
    poly_83 = poly_6 * poly_27 

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
    poly_56,    poly_57,    poly_58,    poly_59,    poly_60, 
    poly_61,    poly_62,    poly_63,    poly_64,    poly_65, 
    poly_66,    poly_67,    poly_68,    poly_69,    poly_70, 
    poly_71,    poly_72,    poly_73,    poly_74,    poly_75, 
    poly_76,    poly_77,    poly_78,    poly_79,    poly_80, 
    poly_81,    poly_82,    poly_83,    ]) 

    return poly 



