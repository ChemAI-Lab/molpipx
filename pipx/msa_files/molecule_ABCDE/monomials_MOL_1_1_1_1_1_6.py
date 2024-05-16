import jax 
import jax.numpy as jnp 
from jax import jit

# File created from /gpfs/fs1/home/r/ravh011/ravh011/PIPMSA_jax/pipjax/msa_files/molecule_ABCDE/MOL_1_1_1_1_1_6.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 11 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(11) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,7) 
    mono_4 = jnp.take(r,6) 
    mono_5 = jnp.take(r,5) 
    mono_6 = jnp.take(r,4) 
    mono_7 = jnp.take(r,3) 
    mono_8 = jnp.take(r,2) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    mono_5, 
    mono_6,    mono_7,    mono_8,    mono_9,    mono_10, 
    ]) 

    return mono 


