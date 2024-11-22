import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A4B.monomials_MOL_4_1_3 import f_monomials as f_monos 

# File created from ./MOL_4_1_3.POLY 

N_POLYS = 30

# Total number of monomials = 30 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(30) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) + jnp.take(mono,2) + jnp.take(mono,3) + jnp.take(mono,4) 
    poly_2 = jnp.take(mono,5) + jnp.take(mono,6) + jnp.take(mono,7) + jnp.take(mono,8) + jnp.take(mono,9) + jnp.take(mono,10) 
    poly_3 = jnp.take(mono,11) + jnp.take(mono,12) + jnp.take(mono,13) + jnp.take(mono,14) + jnp.take(mono,15) + jnp.take(mono,16) 
    poly_4 = jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) + jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) + jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) + jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) 
    poly_5 = jnp.take(mono,29) + jnp.take(mono,30) + jnp.take(mono,31) 
    poly_6 = poly_1 * poly_2 - poly_4 
    poly_7 = jnp.take(mono,32) + jnp.take(mono,33) + jnp.take(mono,34) + jnp.take(mono,35) + jnp.take(mono,36) + jnp.take(mono,37) + jnp.take(mono,38) + jnp.take(mono,39) + jnp.take(mono,40) + jnp.take(mono,41) + jnp.take(mono,42) + jnp.take(mono,43) 
    poly_8 = poly_1 * poly_1 - poly_3 - poly_3 
    poly_9 = poly_2 * poly_2 - poly_7 - poly_5 - poly_7 - poly_5 
    poly_10 = jnp.take(mono,44) + jnp.take(mono,45) + jnp.take(mono,46) + jnp.take(mono,47) 
    poly_11 = jnp.take(mono,48) + jnp.take(mono,49) + jnp.take(mono,50) + jnp.take(mono,51) + jnp.take(mono,52) + jnp.take(mono,53) 
    poly_12 = jnp.take(mono,54) + jnp.take(mono,55) + jnp.take(mono,56) + jnp.take(mono,57) + jnp.take(mono,58) + jnp.take(mono,59) + jnp.take(mono,60) + jnp.take(mono,61) + jnp.take(mono,62) + jnp.take(mono,63) + jnp.take(mono,64) + jnp.take(mono,65) + jnp.take(mono,66) + jnp.take(mono,67) + jnp.take(mono,68) + jnp.take(mono,69) + jnp.take(mono,70) + jnp.take(mono,71) + jnp.take(mono,72) + jnp.take(mono,73) + jnp.take(mono,74) + jnp.take(mono,75) + jnp.take(mono,76) + jnp.take(mono,77) 
    poly_13 = poly_1 * poly_5 
    poly_14 = poly_2 * poly_3 - poly_12 - poly_11 
    poly_15 = jnp.take(mono,78) + jnp.take(mono,79) + jnp.take(mono,80) + jnp.take(mono,81) + jnp.take(mono,82) + jnp.take(mono,83) + jnp.take(mono,84) + jnp.take(mono,85) + jnp.take(mono,86) + jnp.take(mono,87) + jnp.take(mono,88) + jnp.take(mono,89) 
    poly_16 = jnp.take(mono,90) + jnp.take(mono,91) + jnp.take(mono,92) + jnp.take(mono,93) + jnp.take(mono,94) + jnp.take(mono,95) + jnp.take(mono,96) + jnp.take(mono,97) + jnp.take(mono,98) + jnp.take(mono,99) + jnp.take(mono,100) + jnp.take(mono,101) + jnp.take(mono,102) + jnp.take(mono,103) + jnp.take(mono,104) + jnp.take(mono,105) + jnp.take(mono,106) + jnp.take(mono,107) + jnp.take(mono,108) + jnp.take(mono,109) + jnp.take(mono,110) + jnp.take(mono,111) + jnp.take(mono,112) + jnp.take(mono,113) 
    poly_17 = jnp.take(mono,114) + jnp.take(mono,115) + jnp.take(mono,116) + jnp.take(mono,117) + jnp.take(mono,118) + jnp.take(mono,119) + jnp.take(mono,120) + jnp.take(mono,121) + jnp.take(mono,122) + jnp.take(mono,123) + jnp.take(mono,124) + jnp.take(mono,125) 
    poly_18 = jnp.take(mono,126) + jnp.take(mono,127) + jnp.take(mono,128) + jnp.take(mono,129) 
    poly_19 = poly_1 * poly_7 - poly_16 - poly_15 
    poly_20 = jnp.take(mono,130) + jnp.take(mono,131) + jnp.take(mono,132) + jnp.take(mono,133) 
    poly_21 = poly_1 * poly_3 - poly_10 - poly_10 - poly_10 
    poly_22 = poly_1 * poly_4 - poly_12 - poly_11 - poly_11 
    poly_23 = poly_2 * poly_8 - poly_22 
    poly_24 = poly_2 * poly_4 - poly_16 - poly_15 - poly_13 - poly_15 
    poly_25 = poly_2 * poly_5 - poly_17 
    poly_26 = poly_1 * poly_9 - poly_24 
    poly_27 = poly_2 * poly_7 - poly_18 - poly_20 - poly_17 - poly_18 - poly_20 - poly_17 - poly_18 - poly_20 
    poly_28 = poly_1 * poly_8 - poly_21 
    poly_29 = poly_2 * poly_9 - poly_27 - poly_25 

#    stack all polynomials 
    poly = jnp.stack([    poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5, 
    poly_6,    poly_7,    poly_8,    poly_9,    poly_10, 
    poly_11,    poly_12,    poly_13,    poly_14,    poly_15, 
    poly_16,    poly_17,    poly_18,    poly_19,    poly_20, 
    poly_21,    poly_22,    poly_23,    poly_24,    poly_25, 
    poly_26,    poly_27,    poly_28,    poly_29,    ]) 

    return poly 



