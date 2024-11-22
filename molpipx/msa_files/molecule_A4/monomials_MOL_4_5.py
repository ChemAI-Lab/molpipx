import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_4_5.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 6
N_ATOMS = 4
N_XYZ = N_ATOMS * 3

# Total number of monomials = 63 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(63) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,5) 
    mono_2 = jnp.take(r,4) 
    mono_3 = jnp.take(r,3) 
    mono_4 = jnp.take(r,2) 
    mono_5 = jnp.take(r,1) 
    mono_6 = jnp.take(r,0) 
    mono_7 = mono_3 * mono_4 
    mono_8 = mono_2 * mono_5 
    mono_9 = mono_1 * mono_6 
    mono_10 = mono_1 * mono_2 
    mono_11 = mono_1 * mono_3 
    mono_12 = mono_2 * mono_3 
    mono_13 = mono_1 * mono_4 
    mono_14 = mono_2 * mono_4 
    mono_15 = mono_1 * mono_5 
    mono_16 = mono_3 * mono_5 
    mono_17 = mono_4 * mono_5 
    mono_18 = mono_2 * mono_6 
    mono_19 = mono_3 * mono_6 
    mono_20 = mono_4 * mono_6 
    mono_21 = mono_5 * mono_6 
    mono_22 = mono_1 * mono_7 
    mono_23 = mono_2 * mono_7 
    mono_24 = mono_1 * mono_8 
    mono_25 = mono_2 * mono_16 
    mono_26 = mono_2 * mono_17 
    mono_27 = mono_3 * mono_17 
    mono_28 = mono_1 * mono_18 
    mono_29 = mono_1 * mono_19 
    mono_30 = mono_1 * mono_20 
    mono_31 = mono_3 * mono_20 
    mono_32 = mono_1 * mono_21 
    mono_33 = mono_2 * mono_21 
    mono_34 = mono_1 * mono_12 
    mono_35 = mono_1 * mono_17 
    mono_36 = mono_2 * mono_20 
    mono_37 = mono_3 * mono_21 
    mono_38 = mono_1 * mono_14 
    mono_39 = mono_1 * mono_16 
    mono_40 = mono_2 * mono_19 
    mono_41 = mono_4 * mono_21 
    mono_42 = mono_2 * mono_27 
    mono_43 = mono_1 * mono_31 
    mono_44 = mono_1 * mono_33 
    mono_45 = mono_1 * mono_23 
    mono_46 = mono_1 * mono_25 
    mono_47 = mono_1 * mono_26 
    mono_48 = mono_1 * mono_27 
    mono_49 = mono_1 * mono_40 
    mono_50 = mono_1 * mono_36 
    mono_51 = mono_2 * mono_31 
    mono_52 = mono_1 * mono_37 
    mono_53 = mono_2 * mono_37 
    mono_54 = mono_1 * mono_41 
    mono_55 = mono_2 * mono_41 
    mono_56 = mono_3 * mono_41 
    mono_57 = mono_1 * mono_42 
    mono_58 = mono_1 * mono_51 
    mono_59 = mono_1 * mono_53 
    mono_60 = mono_1 * mono_55 
    mono_61 = mono_1 * mono_56 
    mono_62 = mono_2 * mono_56 

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
    mono_51,    mono_52,    mono_53,    mono_54,    mono_55, 
    mono_56,    mono_57,    mono_58,    mono_59,    mono_60, 
    mono_61,    mono_62,    ]) 

    return mono 



