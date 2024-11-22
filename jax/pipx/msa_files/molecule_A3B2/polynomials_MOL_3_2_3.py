import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3B2.monomials_MOL_3_2_3 import f_monomials as f_monos 

# File created from ./MOL_3_2_3.POLY 

N_POLYS = 48

# Total number of monomials = 48 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(48) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) + jnp.take(mono,3) + jnp.take(mono,4) + jnp.take(mono,5) + jnp.take(mono,6) + jnp.take(mono,7) 
    poly_3 = jnp.take(mono,8) + jnp.take(mono,9) + jnp.take(mono,10) 
    poly_4 = poly_1 * poly_2 
    poly_5 = jnp.take(mono,11) + jnp.take(mono,12) + jnp.take(mono,13) + jnp.take(mono,14) + jnp.take(mono,15) + jnp.take(mono,16) 
    poly_6 = jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) + jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) 
    poly_7 = jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) 
    poly_8 = poly_1 * poly_3 
    poly_9 = jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) + jnp.take(mono,29) + jnp.take(mono,30) + jnp.take(mono,31) 
    poly_10 = poly_2 * poly_3 - poly_9 
    poly_11 = jnp.take(mono,32) + jnp.take(mono,33) + jnp.take(mono,34) 
    poly_12 = poly_1 * poly_1 
    poly_13 = poly_2 * poly_2 - poly_7 - poly_6 - poly_5 - poly_7 - poly_6 - poly_5 
    poly_14 = poly_3 * poly_3 - poly_11 - poly_11 
    poly_15 = poly_1 * poly_5 
    poly_16 = poly_1 * poly_6 
    poly_17 = jnp.take(mono,35) + jnp.take(mono,36) + jnp.take(mono,37) + jnp.take(mono,38) + jnp.take(mono,39) + jnp.take(mono,40) 
    poly_18 = jnp.take(mono,41) + jnp.take(mono,42) 
    poly_19 = poly_1 * poly_7 
    poly_20 = jnp.take(mono,43) + jnp.take(mono,44) + jnp.take(mono,45) + jnp.take(mono,46) + jnp.take(mono,47) + jnp.take(mono,48) + jnp.take(mono,49) + jnp.take(mono,50) + jnp.take(mono,51) + jnp.take(mono,52) + jnp.take(mono,53) + jnp.take(mono,54) 
    poly_21 = poly_1 * poly_9 
    poly_22 = jnp.take(mono,55) + jnp.take(mono,56) + jnp.take(mono,57) 
    poly_23 = poly_1 * poly_10 
    poly_24 = jnp.take(mono,58) + jnp.take(mono,59) + jnp.take(mono,60) + jnp.take(mono,61) + jnp.take(mono,62) + jnp.take(mono,63) + jnp.take(mono,64) + jnp.take(mono,65) + jnp.take(mono,66) + jnp.take(mono,67) + jnp.take(mono,68) + jnp.take(mono,69) 
    poly_25 = jnp.take(mono,70) + jnp.take(mono,71) + jnp.take(mono,72) + jnp.take(mono,73) + jnp.take(mono,74) + jnp.take(mono,75) + jnp.take(mono,76) + jnp.take(mono,77) + jnp.take(mono,78) + jnp.take(mono,79) + jnp.take(mono,80) + jnp.take(mono,81) 
    poly_26 = poly_3 * poly_5 - poly_24 
    poly_27 = poly_3 * poly_6 - poly_25 
    poly_28 = poly_3 * poly_7 - poly_22 
    poly_29 = poly_1 * poly_11 
    poly_30 = jnp.take(mono,82) + jnp.take(mono,83) + jnp.take(mono,84) + jnp.take(mono,85) + jnp.take(mono,86) + jnp.take(mono,87) + jnp.take(mono,88) + jnp.take(mono,89) + jnp.take(mono,90) + jnp.take(mono,91) + jnp.take(mono,92) + jnp.take(mono,93) 
    poly_31 = jnp.take(mono,94) 
    poly_32 = poly_2 * poly_11 - poly_30 
    poly_33 = poly_1 * poly_4 
    poly_34 = poly_1 * poly_13 
    poly_35 = poly_2 * poly_5 - poly_20 - poly_17 - poly_17 
    poly_36 = poly_2 * poly_6 - poly_20 - poly_18 - poly_17 - poly_18 - poly_18 
    poly_37 = poly_2 * poly_7 - poly_20 
    poly_38 = poly_1 * poly_8 
    poly_39 = poly_2 * poly_9 - poly_25 - poly_24 - poly_22 - poly_22 
    poly_40 = poly_3 * poly_13 - poly_39 
    poly_41 = poly_1 * poly_14 
    poly_42 = poly_3 * poly_9 - poly_30 
    poly_43 = poly_2 * poly_14 - poly_42 
    poly_44 = poly_3 * poly_11 - poly_31 - poly_31 - poly_31 
    poly_45 = poly_1 * poly_12 
    poly_46 = poly_2 * poly_13 - poly_37 - poly_36 - poly_35 
    poly_47 = poly_3 * poly_14 - poly_44 

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
    poly_46,    poly_47,    ]) 

    return poly 



