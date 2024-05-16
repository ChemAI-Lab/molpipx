import jax 
import jax.numpy as jnp 
from jax import jit

# File created from /gpfs/fs1/home/r/ravh011/ravh011/PIPMSA_jax/pipjax/msa_files/molecule_A2B/MOL_2_1_5.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 3
N_ATOMS = 3
N_XYZ = N_ATOMS * 3

# Total number of monomials = 5 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(5) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,2) 
    mono_2 = jnp.take(r,1) 
    mono_3 = jnp.take(r,0) 
    mono_4 = mono_1 * mono_2 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    ]) 

    return mono 



