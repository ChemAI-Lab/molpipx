import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3BC.monomials_MOL_3_1_1_3 import f_monomials as f_monos 

# File created from ./MOL_3_1_1_3.POLY 

N_POLYS = 75

# Total number of monomials = 75 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(75) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) + jnp.take(mono,3) + jnp.take(mono,4) 
    poly_3 = jnp.take(mono,5) + jnp.take(mono,6) + jnp.take(mono,7) 
    poly_4 = jnp.take(mono,8) + jnp.take(mono,9) + jnp.take(mono,10) 
    poly_5 = poly_1 * poly_2 
    poly_6 = jnp.take(mono,11) + jnp.take(mono,12) + jnp.take(mono,13) 
    poly_7 = poly_1 * poly_3 
    poly_8 = jnp.take(mono,14) + jnp.take(mono,15) + jnp.take(mono,16) + jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) 
    poly_9 = jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) 
    poly_10 = poly_2 * poly_3 - poly_8 
    poly_11 = poly_1 * poly_4 
    poly_12 = jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) 
    poly_13 = jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) 
    poly_14 = poly_2 * poly_4 - poly_12 
    poly_15 = poly_3 * poly_4 - poly_13 
    poly_16 = jnp.take(mono,29) + jnp.take(mono,30) + jnp.take(mono,31) 
    poly_17 = poly_1 * poly_1 
    poly_18 = poly_2 * poly_2 - poly_6 - poly_6 
    poly_19 = poly_3 * poly_3 - poly_9 - poly_9 
    poly_20 = poly_4 * poly_4 - poly_16 - poly_16 
    poly_21 = poly_1 * poly_6 
    poly_22 = jnp.take(mono,32) 
    poly_23 = poly_1 * poly_8 
    poly_24 = jnp.take(mono,33) + jnp.take(mono,34) + jnp.take(mono,35) 
    poly_25 = poly_1 * poly_9 
    poly_26 = jnp.take(mono,36) + jnp.take(mono,37) + jnp.take(mono,38) 
    poly_27 = jnp.take(mono,39) 
    poly_28 = poly_1 * poly_10 
    poly_29 = poly_3 * poly_6 - poly_24 
    poly_30 = poly_2 * poly_9 - poly_26 
    poly_31 = poly_1 * poly_12 
    poly_32 = poly_1 * poly_13 
    poly_33 = jnp.take(mono,40) + jnp.take(mono,41) + jnp.take(mono,42) 
    poly_34 = poly_1 * poly_14 
    poly_35 = jnp.take(mono,43) + jnp.take(mono,44) + jnp.take(mono,45) + jnp.take(mono,46) + jnp.take(mono,47) + jnp.take(mono,48) 
    poly_36 = poly_2 * poly_13 - poly_33 
    poly_37 = poly_4 * poly_6 - poly_35 
    poly_38 = poly_1 * poly_15 
    poly_39 = poly_3 * poly_12 - poly_33 
    poly_40 = jnp.take(mono,49) + jnp.take(mono,50) + jnp.take(mono,51) + jnp.take(mono,52) + jnp.take(mono,53) + jnp.take(mono,54) 
    poly_41 = poly_4 * poly_8 - poly_39 - poly_36 
    poly_42 = poly_4 * poly_9 - poly_40 
    poly_43 = poly_4 * poly_10 - poly_33 
    poly_44 = poly_1 * poly_16 
    poly_45 = jnp.take(mono,55) + jnp.take(mono,56) + jnp.take(mono,57) + jnp.take(mono,58) + jnp.take(mono,59) + jnp.take(mono,60) 
    poly_46 = jnp.take(mono,61) + jnp.take(mono,62) + jnp.take(mono,63) + jnp.take(mono,64) + jnp.take(mono,65) + jnp.take(mono,66) 
    poly_47 = jnp.take(mono,67) 
    poly_48 = poly_2 * poly_16 - poly_45 
    poly_49 = poly_3 * poly_16 - poly_46 
    poly_50 = poly_1 * poly_5 
    poly_51 = poly_1 * poly_18 
    poly_52 = poly_2 * poly_6 - poly_22 - poly_22 - poly_22 
    poly_53 = poly_1 * poly_7 
    poly_54 = poly_2 * poly_8 - poly_29 - poly_24 - poly_24 
    poly_55 = poly_2 * poly_10 - poly_29 
    poly_56 = poly_1 * poly_19 
    poly_57 = poly_3 * poly_8 - poly_30 - poly_26 - poly_26 
    poly_58 = poly_3 * poly_9 - poly_27 - poly_27 - poly_27 
    poly_59 = poly_2 * poly_19 - poly_57 
    poly_60 = poly_1 * poly_11 
    poly_61 = poly_2 * poly_12 - poly_35 
    poly_62 = poly_3 * poly_13 - poly_40 
    poly_63 = poly_4 * poly_18 - poly_61 
    poly_64 = poly_4 * poly_19 - poly_62 
    poly_65 = poly_1 * poly_20 
    poly_66 = poly_4 * poly_12 - poly_45 
    poly_67 = poly_4 * poly_13 - poly_46 
    poly_68 = poly_2 * poly_20 - poly_66 
    poly_69 = poly_3 * poly_20 - poly_67 
    poly_70 = poly_4 * poly_16 - poly_47 - poly_47 - poly_47 
    poly_71 = poly_1 * poly_17 
    poly_72 = poly_2 * poly_18 - poly_52 
    poly_73 = poly_3 * poly_19 - poly_58 
    poly_74 = poly_4 * poly_20 - poly_70 

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
    poly_71,    poly_72,    poly_73,    poly_74,    ]) 

    return poly 



