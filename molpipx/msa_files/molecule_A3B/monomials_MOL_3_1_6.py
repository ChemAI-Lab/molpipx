import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_3_1_6.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 6
N_ATOMS = 4
N_XYZ = N_ATOMS * 3

# Total number of monomials = 33 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(33) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,5) 
    mono_2 = jnp.take(r,4) 
    mono_3 = jnp.take(r,2) 
    mono_4 = jnp.take(r,3) 
    mono_5 = jnp.take(r,1) 
    mono_6 = jnp.take(r,0) 
    mono_7 = mono_1 * mono_2 
    mono_8 = mono_1 * mono_3 
    mono_9 = mono_2 * mono_3 
    mono_10 = mono_3 * mono_4 
    mono_11 = mono_2 * mono_5 
    mono_12 = mono_1 * mono_6 
    mono_13 = mono_4 * mono_5 
    mono_14 = mono_4 * mono_6 
    mono_15 = mono_5 * mono_6 
    mono_16 = mono_1 * mono_9 
    mono_17 = mono_1 * mono_10 
    mono_18 = mono_2 * mono_10 
    mono_19 = mono_1 * mono_11 
    mono_20 = mono_3 * mono_11 
    mono_21 = mono_2 * mono_12 
    mono_22 = mono_3 * mono_12 
    mono_23 = mono_2 * mono_13 
    mono_24 = mono_3 * mono_13 
    mono_25 = mono_1 * mono_14 
    mono_26 = mono_3 * mono_14 
    mono_27 = mono_1 * mono_15 
    mono_28 = mono_2 * mono_15 
    mono_29 = mono_4 * mono_15 
    mono_30 = mono_2 * mono_24 
    mono_31 = mono_1 * mono_26 
    mono_32 = mono_1 * mono_28 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    mono_5, 
    mono_6,    mono_7,    mono_8,    mono_9,    mono_10, 
    mono_11,    mono_12,    mono_13,    mono_14,    mono_15, 
    mono_16,    mono_17,    mono_18,    mono_19,    mono_20, 
    mono_21,    mono_22,    mono_23,    mono_24,    mono_25, 
    mono_26,    mono_27,    mono_28,    mono_29,    mono_30, 
    mono_31,    mono_32,    ]) 

    return mono 



