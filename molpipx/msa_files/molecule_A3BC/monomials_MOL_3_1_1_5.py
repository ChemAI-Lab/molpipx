import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_3_1_1_5.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 140 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(140) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,6) 
    mono_4 = jnp.take(r,3) 
    mono_5 = jnp.take(r,7) 
    mono_6 = jnp.take(r,5) 
    mono_7 = jnp.take(r,2) 
    mono_8 = jnp.take(r,4) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 
    mono_11 = mono_2 * mono_3 
    mono_12 = mono_2 * mono_4 
    mono_13 = mono_3 * mono_4 
    mono_14 = mono_3 * mono_5 
    mono_15 = mono_2 * mono_6 
    mono_16 = mono_4 * mono_5 
    mono_17 = mono_4 * mono_6 
    mono_18 = mono_2 * mono_7 
    mono_19 = mono_3 * mono_7 
    mono_20 = mono_5 * mono_6 
    mono_21 = mono_5 * mono_7 
    mono_22 = mono_6 * mono_7 
    mono_23 = mono_4 * mono_8 
    mono_24 = mono_3 * mono_9 
    mono_25 = mono_2 * mono_10 
    mono_26 = mono_7 * mono_8 
    mono_27 = mono_6 * mono_9 
    mono_28 = mono_5 * mono_10 
    mono_29 = mono_8 * mono_9 
    mono_30 = mono_8 * mono_10 
    mono_31 = mono_9 * mono_10 
    mono_32 = mono_2 * mono_13 
    mono_33 = mono_3 * mono_16 
    mono_34 = mono_2 * mono_17 
    mono_35 = mono_2 * mono_19 
    mono_36 = mono_4 * mono_20 
    mono_37 = mono_3 * mono_21 
    mono_38 = mono_2 * mono_22 
    mono_39 = mono_5 * mono_22 
    mono_40 = mono_4 * mono_26 
    mono_41 = mono_3 * mono_27 
    mono_42 = mono_2 * mono_28 
    mono_43 = mono_2 * mono_23 
    mono_44 = mono_3 * mono_23 
    mono_45 = mono_2 * mono_24 
    mono_46 = mono_4 * mono_24 
    mono_47 = mono_3 * mono_25 
    mono_48 = mono_4 * mono_25 
    mono_49 = mono_5 * mono_26 
    mono_50 = mono_6 * mono_26 
    mono_51 = mono_5 * mono_27 
    mono_52 = mono_7 * mono_27 
    mono_53 = mono_6 * mono_28 
    mono_54 = mono_7 * mono_28 
    mono_55 = mono_3 * mono_29 
    mono_56 = mono_4 * mono_29 
    mono_57 = mono_2 * mono_30 
    mono_58 = mono_4 * mono_30 
    mono_59 = mono_2 * mono_31 
    mono_60 = mono_3 * mono_31 
    mono_61 = mono_6 * mono_29 
    mono_62 = mono_7 * mono_29 
    mono_63 = mono_5 * mono_30 
    mono_64 = mono_7 * mono_30 
    mono_65 = mono_5 * mono_31 
    mono_66 = mono_6 * mono_31 
    mono_67 = mono_8 * mono_31 
    mono_68 = mono_2 * mono_36 
    mono_69 = mono_3 * mono_36 
    mono_70 = mono_2 * mono_37 
    mono_71 = mono_3 * mono_38 
    mono_72 = mono_4 * mono_37 
    mono_73 = mono_4 * mono_38 
    mono_74 = mono_2 * mono_40 
    mono_75 = mono_3 * mono_40 
    mono_76 = mono_2 * mono_41 
    mono_77 = mono_4 * mono_41 
    mono_78 = mono_3 * mono_42 
    mono_79 = mono_4 * mono_42 
    mono_80 = mono_4 * mono_49 
    mono_81 = mono_4 * mono_50 
    mono_82 = mono_3 * mono_51 
    mono_83 = mono_3 * mono_52 
    mono_84 = mono_2 * mono_53 
    mono_85 = mono_2 * mono_54 
    mono_86 = mono_3 * mono_49 
    mono_87 = mono_2 * mono_50 
    mono_88 = mono_4 * mono_51 
    mono_89 = mono_2 * mono_52 
    mono_90 = mono_4 * mono_53 
    mono_91 = mono_3 * mono_54 
    mono_92 = mono_3 * mono_56 
    mono_93 = mono_2 * mono_58 
    mono_94 = mono_2 * mono_60 
    mono_95 = mono_4 * mono_61 
    mono_96 = mono_3 * mono_62 
    mono_97 = mono_4 * mono_63 
    mono_98 = mono_2 * mono_64 
    mono_99 = mono_3 * mono_65 
    mono_100 = mono_2 * mono_66 
    mono_101 = mono_6 * mono_62 
    mono_102 = mono_5 * mono_64 
    mono_103 = mono_5 * mono_66 
    mono_104 = mono_3 * mono_61 
    mono_105 = mono_4 * mono_62 
    mono_106 = mono_2 * mono_63 
    mono_107 = mono_4 * mono_64 
    mono_108 = mono_2 * mono_65 
    mono_109 = mono_3 * mono_66 
    mono_110 = mono_3 * mono_80 
    mono_111 = mono_2 * mono_81 
    mono_112 = mono_3 * mono_88 
    mono_113 = mono_2 * mono_83 
    mono_114 = mono_2 * mono_90 
    mono_115 = mono_2 * mono_91 
    mono_116 = mono_2 * mono_80 
    mono_117 = mono_3 * mono_81 
    mono_118 = mono_2 * mono_82 
    mono_119 = mono_4 * mono_83 
    mono_120 = mono_3 * mono_84 
    mono_121 = mono_4 * mono_85 
    mono_122 = mono_3 * mono_95 
    mono_123 = mono_3 * mono_105 
    mono_124 = mono_2 * mono_97 
    mono_125 = mono_2 * mono_107 
    mono_126 = mono_2 * mono_99 
    mono_127 = mono_2 * mono_109 
    mono_128 = mono_3 * mono_101 
    mono_129 = mono_4 * mono_101 
    mono_130 = mono_2 * mono_102 
    mono_131 = mono_4 * mono_102 
    mono_132 = mono_2 * mono_103 
    mono_133 = mono_3 * mono_103 
    mono_134 = mono_2 * mono_95 
    mono_135 = mono_2 * mono_96 
    mono_136 = mono_3 * mono_97 
    mono_137 = mono_3 * mono_98 
    mono_138 = mono_4 * mono_99 
    mono_139 = mono_4 * mono_100 

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
    mono_131,    mono_132,    mono_133,    mono_134,    mono_135, 
    mono_136,    mono_137,    mono_138,    mono_139,    ]) 

    return mono 



