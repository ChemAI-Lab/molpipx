import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_2_1_1_1_5.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 22 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(22) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,7) 
    mono_4 = jnp.take(r,6) 
    mono_5 = jnp.take(r,3) 
    mono_6 = jnp.take(r,5) 
    mono_7 = jnp.take(r,2) 
    mono_8 = jnp.take(r,4) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 
    mono_11 = mono_4 * mono_5 
    mono_12 = mono_5 * mono_6 
    mono_13 = mono_4 * mono_7 
    mono_14 = mono_6 * mono_7 
    mono_15 = mono_5 * mono_8 
    mono_16 = mono_4 * mono_9 
    mono_17 = mono_7 * mono_8 
    mono_18 = mono_6 * mono_9 
    mono_19 = mono_8 * mono_9 
    mono_20 = mono_5 * mono_17 
    mono_21 = mono_4 * mono_18 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    mono_5, 
    mono_6,    mono_7,    mono_8,    mono_9,    mono_10, 
    mono_11,    mono_12,    mono_13,    mono_14,    mono_15, 
    mono_16,    mono_17,    mono_18,    mono_19,    mono_20, 
    mono_21,    ]) 

    return mono 



