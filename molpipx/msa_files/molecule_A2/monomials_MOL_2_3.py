import jax 
import jax.numpy as jnp 
from jax import jit

# File created from /home/ravh011/molpipx/molpipx/msa_files/molecule_A2/msa_2_files/MOL_2_3.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 1
N_ATOMS = 2
N_XYZ = N_ATOMS * 3

# Total number of monomials = 2 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(2) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,0) 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    ]) 

    return mono 



