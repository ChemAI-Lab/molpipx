import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_5_3.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 176 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(176) 

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
    mono_11 = mono_3 * mono_4 
    mono_12 = mono_2 * mono_5 
    mono_13 = mono_1 * mono_6 
    mono_14 = mono_3 * mono_7 
    mono_15 = mono_5 * mono_7 
    mono_16 = mono_6 * mono_7 
    mono_17 = mono_2 * mono_8 
    mono_18 = mono_4 * mono_8 
    mono_19 = mono_6 * mono_8 
    mono_20 = mono_1 * mono_9 
    mono_21 = mono_4 * mono_9 
    mono_22 = mono_5 * mono_9 
    mono_23 = mono_1 * mono_10 
    mono_24 = mono_2 * mono_10 
    mono_25 = mono_3 * mono_10 
    mono_26 = mono_1 * mono_2 
    mono_27 = mono_1 * mono_3 
    mono_28 = mono_2 * mono_3 
    mono_29 = mono_1 * mono_4 
    mono_30 = mono_2 * mono_4 
    mono_31 = mono_1 * mono_5 
    mono_32 = mono_3 * mono_5 
    mono_33 = mono_4 * mono_5 
    mono_34 = mono_2 * mono_6 
    mono_35 = mono_3 * mono_6 
    mono_36 = mono_4 * mono_6 
    mono_37 = mono_5 * mono_6 
    mono_38 = mono_1 * mono_7 
    mono_39 = mono_2 * mono_7 
    mono_40 = mono_4 * mono_7 
    mono_41 = mono_1 * mono_8 
    mono_42 = mono_3 * mono_8 
    mono_43 = mono_5 * mono_8 
    mono_44 = mono_7 * mono_8 
    mono_45 = mono_2 * mono_9 
    mono_46 = mono_3 * mono_9 
    mono_47 = mono_6 * mono_9 
    mono_48 = mono_7 * mono_9 
    mono_49 = mono_8 * mono_9 
    mono_50 = mono_4 * mono_10 
    mono_51 = mono_5 * mono_10 
    mono_52 = mono_6 * mono_10 
    mono_53 = mono_7 * mono_10 
    mono_54 = mono_8 * mono_10 
    mono_55 = mono_9 * mono_10 
    mono_56 = mono_3 * mono_40 
    mono_57 = mono_2 * mono_15 
    mono_58 = mono_3 * mono_15 
    mono_59 = mono_1 * mono_16 
    mono_60 = mono_3 * mono_16 
    mono_61 = mono_5 * mono_16 
    mono_62 = mono_2 * mono_18 
    mono_63 = mono_3 * mono_18 
    mono_64 = mono_2 * mono_43 
    mono_65 = mono_1 * mono_19 
    mono_66 = mono_2 * mono_19 
    mono_67 = mono_4 * mono_19 
    mono_68 = mono_6 * mono_44 
    mono_69 = mono_1 * mono_21 
    mono_70 = mono_3 * mono_21 
    mono_71 = mono_1 * mono_22 
    mono_72 = mono_2 * mono_22 
    mono_73 = mono_4 * mono_22 
    mono_74 = mono_1 * mono_47 
    mono_75 = mono_5 * mono_48 
    mono_76 = mono_4 * mono_49 
    mono_77 = mono_1 * mono_24 
    mono_78 = mono_1 * mono_25 
    mono_79 = mono_2 * mono_25 
    mono_80 = mono_3 * mono_50 
    mono_81 = mono_2 * mono_51 
    mono_82 = mono_1 * mono_52 
    mono_83 = mono_3 * mono_53 
    mono_84 = mono_2 * mono_54 
    mono_85 = mono_1 * mono_55 
    mono_86 = mono_1 * mono_11 
    mono_87 = mono_2 * mono_11 
    mono_88 = mono_1 * mono_12 
    mono_89 = mono_2 * mono_32 
    mono_90 = mono_2 * mono_33 
    mono_91 = mono_3 * mono_33 
    mono_92 = mono_1 * mono_34 
    mono_93 = mono_1 * mono_35 
    mono_94 = mono_1 * mono_36 
    mono_95 = mono_3 * mono_36 
    mono_96 = mono_1 * mono_37 
    mono_97 = mono_2 * mono_37 
    mono_98 = mono_1 * mono_14 
    mono_99 = mono_2 * mono_14 
    mono_100 = mono_1 * mono_15 
    mono_101 = mono_4 * mono_15 
    mono_102 = mono_2 * mono_16 
    mono_103 = mono_4 * mono_16 
    mono_104 = mono_1 * mono_17 
    mono_105 = mono_2 * mono_42 
    mono_106 = mono_1 * mono_18 
    mono_107 = mono_4 * mono_43 
    mono_108 = mono_3 * mono_19 
    mono_109 = mono_5 * mono_19 
    mono_110 = mono_2 * mono_44 
    mono_111 = mono_3 * mono_44 
    mono_112 = mono_4 * mono_44 
    mono_113 = mono_5 * mono_44 
    mono_114 = mono_1 * mono_45 
    mono_115 = mono_1 * mono_46 
    mono_116 = mono_2 * mono_21 
    mono_117 = mono_3 * mono_22 
    mono_118 = mono_4 * mono_47 
    mono_119 = mono_5 * mono_47 
    mono_120 = mono_1 * mono_48 
    mono_121 = mono_3 * mono_48 
    mono_122 = mono_4 * mono_48 
    mono_123 = mono_6 * mono_48 
    mono_124 = mono_1 * mono_49 
    mono_125 = mono_2 * mono_49 
    mono_126 = mono_5 * mono_49 
    mono_127 = mono_6 * mono_49 
    mono_128 = mono_1 * mono_50 
    mono_129 = mono_2 * mono_50 
    mono_130 = mono_1 * mono_51 
    mono_131 = mono_3 * mono_51 
    mono_132 = mono_2 * mono_52 
    mono_133 = mono_3 * mono_52 
    mono_134 = mono_1 * mono_53 
    mono_135 = mono_2 * mono_53 
    mono_136 = mono_5 * mono_53 
    mono_137 = mono_6 * mono_53 
    mono_138 = mono_1 * mono_54 
    mono_139 = mono_3 * mono_54 
    mono_140 = mono_4 * mono_54 
    mono_141 = mono_6 * mono_54 
    mono_142 = mono_2 * mono_55 
    mono_143 = mono_3 * mono_55 
    mono_144 = mono_4 * mono_55 
    mono_145 = mono_5 * mono_55 
    mono_146 = mono_1 * mono_28 
    mono_147 = mono_1 * mono_33 
    mono_148 = mono_2 * mono_36 
    mono_149 = mono_3 * mono_37 
    mono_150 = mono_1 * mono_44 
    mono_151 = mono_2 * mono_48 
    mono_152 = mono_3 * mono_49 
    mono_153 = mono_4 * mono_53 
    mono_154 = mono_5 * mono_54 
    mono_155 = mono_6 * mono_55 
    mono_156 = mono_1 * mono_30 
    mono_157 = mono_1 * mono_32 
    mono_158 = mono_2 * mono_35 
    mono_159 = mono_4 * mono_37 
    mono_160 = mono_1 * mono_39 
    mono_161 = mono_1 * mono_40 
    mono_162 = mono_2 * mono_40 
    mono_163 = mono_1 * mono_42 
    mono_164 = mono_1 * mono_43 
    mono_165 = mono_3 * mono_43 
    mono_166 = mono_2 * mono_46 
    mono_167 = mono_2 * mono_47 
    mono_168 = mono_3 * mono_47 
    mono_169 = mono_7 * mono_49 
    mono_170 = mono_4 * mono_51 
    mono_171 = mono_4 * mono_52 
    mono_172 = mono_5 * mono_52 
    mono_173 = mono_7 * mono_54 
    mono_174 = mono_7 * mono_55 
    mono_175 = mono_8 * mono_55 

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
    mono_136,    mono_137,    mono_138,    mono_139,    mono_140, 
    mono_141,    mono_142,    mono_143,    mono_144,    mono_145, 
    mono_146,    mono_147,    mono_148,    mono_149,    mono_150, 
    mono_151,    mono_152,    mono_153,    mono_154,    mono_155, 
    mono_156,    mono_157,    mono_158,    mono_159,    mono_160, 
    mono_161,    mono_162,    mono_163,    mono_164,    mono_165, 
    mono_166,    mono_167,    mono_168,    mono_169,    mono_170, 
    mono_171,    mono_172,    mono_173,    mono_174,    mono_175, 
    ]) 

    return mono 



