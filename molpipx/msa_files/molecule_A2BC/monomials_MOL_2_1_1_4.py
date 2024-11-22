import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_2_1_1_4.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 6
N_ATOMS = 4
N_XYZ = N_ATOMS * 3

# Total number of monomials = 11 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(11) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,5) 
    mono_2 = jnp.take(r,4) 
    mono_3 = jnp.take(r,2) 
    mono_4 = jnp.take(r,3) 
    mono_5 = jnp.take(r,1) 
    mono_6 = jnp.take(r,0) 
    mono_7 = mono_2 * mono_3 
    mono_8 = mono_3 * mono_4 
    mono_9 = mono_2 * mono_5 
    mono_10 = mono_4 * mono_5 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    mono_5, 
    mono_6,    mono_7,    mono_8,    mono_9,    mono_10, 
    ]) 

    return mono 



