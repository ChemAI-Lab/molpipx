import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_2_2_1_4.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 52 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(52) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,7) 
    mono_4 = jnp.take(r,6) 
    mono_5 = jnp.take(r,3) 
    mono_6 = jnp.take(r,5) 
    mono_7 = jnp.take(r,4) 
    mono_8 = jnp.take(r,2) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 
    mono_11 = mono_1 * mono_2 
    mono_12 = mono_4 * mono_5 
    mono_13 = mono_2 * mono_6 
    mono_14 = mono_1 * mono_7 
    mono_15 = mono_2 * mono_8 
    mono_16 = mono_1 * mono_9 
    mono_17 = mono_5 * mono_6 
    mono_18 = mono_5 * mono_7 
    mono_19 = mono_4 * mono_8 
    mono_20 = mono_4 * mono_9 
    mono_21 = mono_7 * mono_8 
    mono_22 = mono_6 * mono_9 
    mono_23 = mono_6 * mono_8 
    mono_24 = mono_7 * mono_9 
    mono_25 = mono_6 * mono_7 
    mono_26 = mono_8 * mono_9 
    mono_27 = mono_2 * mono_17 
    mono_28 = mono_1 * mono_18 
    mono_29 = mono_2 * mono_19 
    mono_30 = mono_1 * mono_20 
    mono_31 = mono_2 * mono_23 
    mono_32 = mono_1 * mono_24 
    mono_33 = mono_5 * mono_25 
    mono_34 = mono_4 * mono_26 
    mono_35 = mono_6 * mono_21 
    mono_36 = mono_6 * mono_24 
    mono_37 = mono_6 * mono_26 
    mono_38 = mono_7 * mono_26 
    mono_39 = mono_14 * mono_19 
    mono_40 = mono_15 * mono_18 
    mono_41 = mono_13 * mono_20 
    mono_42 = mono_16 * mono_17 
    mono_43 = mono_2 * mono_35 
    mono_44 = mono_1 * mono_36 
    mono_45 = mono_2 * mono_37 
    mono_46 = mono_1 * mono_38 
    mono_47 = mono_5 * mono_35 
    mono_48 = mono_5 * mono_36 
    mono_49 = mono_4 * mono_37 
    mono_50 = mono_4 * mono_38 
    mono_51 = mono_6 * mono_38 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    mono_5, 
    mono_6,    mono_7,    mono_8,    mono_9,    mono_10, 
    mono_11,    mono_12,    mono_13,    mono_14,    mono_15, 
    mono_16,    mono_17,    mono_18,    mono_19,    mono_20, 
    mono_21,    mono_22,    mono_23,    mono_24,    mono_25, 
    mono_26,    mono_27,    mono_28,    mono_29,    mono_30, 
    mono_31,    mono_32,    mono_33,    mono_34,    mono_35, 
    mono_36,    mono_37,    mono_38,    mono_39,    mono_40, 
    mono_41,    mono_42,    mono_43,    mono_44,    mono_45, 
    mono_46,    mono_47,    mono_48,    mono_49,    mono_50, 
    mono_51,    ]) 

    return mono 



