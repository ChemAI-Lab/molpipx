import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_4_1_3.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 134 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(134) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,6) 
    mono_4 = jnp.take(r,3) 
    mono_5 = jnp.take(r,7) 
    mono_6 = jnp.take(r,5) 
    mono_7 = jnp.take(r,4) 
    mono_8 = jnp.take(r,2) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 
    mono_11 = mono_1 * mono_2 
    mono_12 = mono_1 * mono_3 
    mono_13 = mono_2 * mono_3 
    mono_14 = mono_1 * mono_4 
    mono_15 = mono_2 * mono_4 
    mono_16 = mono_3 * mono_4 
    mono_17 = mono_3 * mono_5 
    mono_18 = mono_2 * mono_6 
    mono_19 = mono_1 * mono_7 
    mono_20 = mono_4 * mono_5 
    mono_21 = mono_4 * mono_6 
    mono_22 = mono_4 * mono_7 
    mono_23 = mono_2 * mono_8 
    mono_24 = mono_3 * mono_8 
    mono_25 = mono_1 * mono_9 
    mono_26 = mono_3 * mono_9 
    mono_27 = mono_1 * mono_10 
    mono_28 = mono_2 * mono_10 
    mono_29 = mono_7 * mono_8 
    mono_30 = mono_6 * mono_9 
    mono_31 = mono_5 * mono_10 
    mono_32 = mono_5 * mono_6 
    mono_33 = mono_5 * mono_7 
    mono_34 = mono_6 * mono_7 
    mono_35 = mono_5 * mono_8 
    mono_36 = mono_6 * mono_8 
    mono_37 = mono_5 * mono_9 
    mono_38 = mono_7 * mono_9 
    mono_39 = mono_8 * mono_9 
    mono_40 = mono_6 * mono_10 
    mono_41 = mono_7 * mono_10 
    mono_42 = mono_8 * mono_10 
    mono_43 = mono_9 * mono_10 
    mono_44 = mono_1 * mono_13 
    mono_45 = mono_1 * mono_15 
    mono_46 = mono_1 * mono_16 
    mono_47 = mono_2 * mono_16 
    mono_48 = mono_3 * mono_20 
    mono_49 = mono_2 * mono_21 
    mono_50 = mono_1 * mono_22 
    mono_51 = mono_2 * mono_24 
    mono_52 = mono_1 * mono_26 
    mono_53 = mono_1 * mono_28 
    mono_54 = mono_1 * mono_17 
    mono_55 = mono_2 * mono_17 
    mono_56 = mono_1 * mono_18 
    mono_57 = mono_3 * mono_18 
    mono_58 = mono_2 * mono_19 
    mono_59 = mono_3 * mono_19 
    mono_60 = mono_1 * mono_20 
    mono_61 = mono_2 * mono_20 
    mono_62 = mono_1 * mono_21 
    mono_63 = mono_3 * mono_21 
    mono_64 = mono_2 * mono_22 
    mono_65 = mono_3 * mono_22 
    mono_66 = mono_1 * mono_23 
    mono_67 = mono_1 * mono_24 
    mono_68 = mono_4 * mono_23 
    mono_69 = mono_4 * mono_24 
    mono_70 = mono_2 * mono_25 
    mono_71 = mono_2 * mono_26 
    mono_72 = mono_4 * mono_25 
    mono_73 = mono_4 * mono_26 
    mono_74 = mono_3 * mono_27 
    mono_75 = mono_3 * mono_28 
    mono_76 = mono_4 * mono_27 
    mono_77 = mono_4 * mono_28 
    mono_78 = mono_4 * mono_32 
    mono_79 = mono_4 * mono_33 
    mono_80 = mono_4 * mono_34 
    mono_81 = mono_3 * mono_35 
    mono_82 = mono_2 * mono_36 
    mono_83 = mono_3 * mono_37 
    mono_84 = mono_1 * mono_38 
    mono_85 = mono_3 * mono_39 
    mono_86 = mono_2 * mono_40 
    mono_87 = mono_1 * mono_41 
    mono_88 = mono_2 * mono_42 
    mono_89 = mono_1 * mono_43 
    mono_90 = mono_2 * mono_32 
    mono_91 = mono_3 * mono_32 
    mono_92 = mono_1 * mono_33 
    mono_93 = mono_3 * mono_33 
    mono_94 = mono_1 * mono_34 
    mono_95 = mono_2 * mono_34 
    mono_96 = mono_2 * mono_35 
    mono_97 = mono_3 * mono_36 
    mono_98 = mono_4 * mono_35 
    mono_99 = mono_4 * mono_36 
    mono_100 = mono_1 * mono_37 
    mono_101 = mono_3 * mono_38 
    mono_102 = mono_4 * mono_37 
    mono_103 = mono_4 * mono_38 
    mono_104 = mono_1 * mono_39 
    mono_105 = mono_2 * mono_39 
    mono_106 = mono_1 * mono_40 
    mono_107 = mono_2 * mono_41 
    mono_108 = mono_4 * mono_40 
    mono_109 = mono_4 * mono_41 
    mono_110 = mono_1 * mono_42 
    mono_111 = mono_3 * mono_42 
    mono_112 = mono_2 * mono_43 
    mono_113 = mono_3 * mono_43 
    mono_114 = mono_5 * mono_29 
    mono_115 = mono_6 * mono_29 
    mono_116 = mono_5 * mono_30 
    mono_117 = mono_6 * mono_38 
    mono_118 = mono_6 * mono_39 
    mono_119 = mono_7 * mono_39 
    mono_120 = mono_5 * mono_40 
    mono_121 = mono_5 * mono_41 
    mono_122 = mono_5 * mono_42 
    mono_123 = mono_7 * mono_42 
    mono_124 = mono_5 * mono_43 
    mono_125 = mono_6 * mono_43 
    mono_126 = mono_5 * mono_34 
    mono_127 = mono_5 * mono_39 
    mono_128 = mono_6 * mono_42 
    mono_129 = mono_7 * mono_43 
    mono_130 = mono_5 * mono_36 
    mono_131 = mono_5 * mono_38 
    mono_132 = mono_6 * mono_41 
    mono_133 = mono_8 * mono_43 

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
    mono_91,    mono_92,    mono_93,    mono_94,    mono_95, 
    mono_96,    mono_97,    mono_98,    mono_99,    mono_100, 
    mono_101,    mono_102,    mono_103,    mono_104,    mono_105, 
    mono_106,    mono_107,    mono_108,    mono_109,    mono_110, 
    mono_111,    mono_112,    mono_113,    mono_114,    mono_115, 
    mono_116,    mono_117,    mono_118,    mono_119,    mono_120, 
    mono_121,    mono_122,    mono_123,    mono_124,    mono_125, 
    mono_126,    mono_127,    mono_128,    mono_129,    mono_130, 
    mono_131,    mono_132,    mono_133,    ]) 

    return mono 



