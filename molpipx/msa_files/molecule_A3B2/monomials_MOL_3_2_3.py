import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_3_2_3.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 95 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(95) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,7) 
    mono_4 = jnp.take(r,6) 
    mono_5 = jnp.take(r,5) 
    mono_6 = jnp.take(r,3) 
    mono_7 = jnp.take(r,2) 
    mono_8 = jnp.take(r,4) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 
    mono_11 = mono_3 * mono_4 
    mono_12 = mono_2 * mono_5 
    mono_13 = mono_3 * mono_6 
    mono_14 = mono_5 * mono_6 
    mono_15 = mono_2 * mono_7 
    mono_16 = mono_4 * mono_7 
    mono_17 = mono_2 * mono_4 
    mono_18 = mono_3 * mono_5 
    mono_19 = mono_2 * mono_6 
    mono_20 = mono_4 * mono_6 
    mono_21 = mono_3 * mono_7 
    mono_22 = mono_5 * mono_7 
    mono_23 = mono_2 * mono_3 
    mono_24 = mono_4 * mono_5 
    mono_25 = mono_6 * mono_7 
    mono_26 = mono_6 * mono_8 
    mono_27 = mono_7 * mono_8 
    mono_28 = mono_4 * mono_9 
    mono_29 = mono_5 * mono_9 
    mono_30 = mono_2 * mono_10 
    mono_31 = mono_3 * mono_10 
    mono_32 = mono_8 * mono_9 
    mono_33 = mono_8 * mono_10 
    mono_34 = mono_9 * mono_10 
    mono_35 = mono_3 * mono_20 
    mono_36 = mono_2 * mono_14 
    mono_37 = mono_3 * mono_14 
    mono_38 = mono_2 * mono_16 
    mono_39 = mono_3 * mono_16 
    mono_40 = mono_2 * mono_22 
    mono_41 = mono_2 * mono_20 
    mono_42 = mono_3 * mono_22 
    mono_43 = mono_2 * mono_11 
    mono_44 = mono_2 * mono_18 
    mono_45 = mono_2 * mono_24 
    mono_46 = mono_3 * mono_24 
    mono_47 = mono_2 * mono_13 
    mono_48 = mono_4 * mono_14 
    mono_49 = mono_2 * mono_21 
    mono_50 = mono_4 * mono_22 
    mono_51 = mono_2 * mono_25 
    mono_52 = mono_3 * mono_25 
    mono_53 = mono_4 * mono_25 
    mono_54 = mono_5 * mono_25 
    mono_55 = mono_6 * mono_27 
    mono_56 = mono_4 * mono_29 
    mono_57 = mono_2 * mono_31 
    mono_58 = mono_3 * mono_26 
    mono_59 = mono_5 * mono_26 
    mono_60 = mono_2 * mono_27 
    mono_61 = mono_4 * mono_27 
    mono_62 = mono_3 * mono_28 
    mono_63 = mono_2 * mono_29 
    mono_64 = mono_6 * mono_29 
    mono_65 = mono_7 * mono_28 
    mono_66 = mono_4 * mono_31 
    mono_67 = mono_5 * mono_30 
    mono_68 = mono_6 * mono_31 
    mono_69 = mono_7 * mono_30 
    mono_70 = mono_2 * mono_26 
    mono_71 = mono_4 * mono_26 
    mono_72 = mono_3 * mono_27 
    mono_73 = mono_5 * mono_27 
    mono_74 = mono_2 * mono_28 
    mono_75 = mono_3 * mono_29 
    mono_76 = mono_6 * mono_28 
    mono_77 = mono_7 * mono_29 
    mono_78 = mono_4 * mono_30 
    mono_79 = mono_5 * mono_31 
    mono_80 = mono_6 * mono_30 
    mono_81 = mono_7 * mono_31 
    mono_82 = mono_4 * mono_32 
    mono_83 = mono_5 * mono_32 
    mono_84 = mono_6 * mono_32 
    mono_85 = mono_7 * mono_32 
    mono_86 = mono_2 * mono_33 
    mono_87 = mono_3 * mono_33 
    mono_88 = mono_6 * mono_33 
    mono_89 = mono_7 * mono_33 
    mono_90 = mono_2 * mono_34 
    mono_91 = mono_3 * mono_34 
    mono_92 = mono_4 * mono_34 
    mono_93 = mono_5 * mono_34 
    mono_94 = mono_8 * mono_34 

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
    mono_61,    mono_62,    mono_63,    mono_64,    mono_65, 
    mono_66,    mono_67,    mono_68,    mono_69,    mono_70, 
    mono_71,    mono_72,    mono_73,    mono_74,    mono_75, 
    mono_76,    mono_77,    mono_78,    mono_79,    mono_80, 
    mono_81,    mono_82,    mono_83,    mono_84,    mono_85, 
    mono_86,    mono_87,    mono_88,    mono_89,    mono_90, 
    mono_91,    mono_92,    mono_93,    mono_94,    ]) 

    return mono 



