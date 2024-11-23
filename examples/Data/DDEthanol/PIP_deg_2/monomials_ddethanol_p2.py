import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_1_1_1_2_3_1_2.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 36
N_ATOMS = 9
N_XYZ = N_ATOMS * 3

# Total number of monomials = 209 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(209) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,35) 
    mono_2 = jnp.take(r,34) 
    mono_3 = jnp.take(r,32) 
    mono_4 = jnp.take(r,33) 
    mono_5 = jnp.take(r,31) 
    mono_6 = jnp.take(r,30) 
    mono_7 = jnp.take(r,29) 
    mono_8 = jnp.take(r,25) 
    mono_9 = jnp.take(r,28) 
    mono_10 = jnp.take(r,27) 
    mono_11 = jnp.take(r,26) 
    mono_12 = jnp.take(r,24) 
    mono_13 = jnp.take(r,23) 
    mono_14 = jnp.take(r,22) 
    mono_15 = jnp.take(r,21) 
    mono_16 = jnp.take(r,20) 
    mono_17 = jnp.take(r,19) 
    mono_18 = jnp.take(r,18) 
    mono_19 = jnp.take(r,17) 
    mono_20 = jnp.take(r,16) 
    mono_21 = jnp.take(r,15) 
    mono_22 = jnp.take(r,14) 
    mono_23 = jnp.take(r,13) 
    mono_24 = jnp.take(r,12) 
    mono_25 = jnp.take(r,11) 
    mono_26 = jnp.take(r,10) 
    mono_27 = jnp.take(r,9) 
    mono_28 = jnp.take(r,8) 
    mono_29 = jnp.take(r,7) 
    mono_30 = jnp.take(r,6) 
    mono_31 = jnp.take(r,5) 
    mono_32 = jnp.take(r,4) 
    mono_33 = jnp.take(r,3) 
    mono_34 = jnp.take(r,2) 
    mono_35 = jnp.take(r,1) 
    mono_36 = jnp.take(r,0) 
    mono_37 = mono_1 * mono_2 
    mono_38 = mono_1 * mono_3 
    mono_39 = mono_2 * mono_3 
    mono_40 = mono_3 * mono_4 
    mono_41 = mono_2 * mono_5 
    mono_42 = mono_1 * mono_6 
    mono_43 = mono_4 * mono_5 
    mono_44 = mono_4 * mono_6 
    mono_45 = mono_5 * mono_6 
    mono_46 = mono_7 * mono_8 
    mono_47 = mono_2 * mono_9 
    mono_48 = mono_3 * mono_9 
    mono_49 = mono_1 * mono_10 
    mono_50 = mono_3 * mono_10 
    mono_51 = mono_1 * mono_11 
    mono_52 = mono_2 * mono_11 
    mono_53 = mono_2 * mono_12 
    mono_54 = mono_3 * mono_12 
    mono_55 = mono_1 * mono_13 
    mono_56 = mono_3 * mono_13 
    mono_57 = mono_1 * mono_14 
    mono_58 = mono_2 * mono_14 
    mono_59 = mono_6 * mono_9 
    mono_60 = mono_5 * mono_10 
    mono_61 = mono_4 * mono_11 
    mono_62 = mono_6 * mono_12 
    mono_63 = mono_5 * mono_13 
    mono_64 = mono_4 * mono_14 
    mono_65 = mono_8 * mono_9 
    mono_66 = mono_8 * mono_10 
    mono_67 = mono_8 * mono_11 
    mono_68 = mono_7 * mono_12 
    mono_69 = mono_7 * mono_13 
    mono_70 = mono_7 * mono_14 
    mono_71 = mono_10 * mono_12 
    mono_72 = mono_11 * mono_12 
    mono_73 = mono_9 * mono_13 
    mono_74 = mono_11 * mono_13 
    mono_75 = mono_9 * mono_14 
    mono_76 = mono_10 * mono_14 
    mono_77 = mono_9 * mono_12 
    mono_78 = mono_10 * mono_13 
    mono_79 = mono_11 * mono_14 
    mono_80 = mono_9 * mono_10 
    mono_81 = mono_9 * mono_11 
    mono_82 = mono_10 * mono_11 
    mono_83 = mono_12 * mono_13 
    mono_84 = mono_12 * mono_14 
    mono_85 = mono_13 * mono_14 
    mono_86 = mono_2 * mono_17 
    mono_87 = mono_3 * mono_17 
    mono_88 = mono_1 * mono_18 
    mono_89 = mono_3 * mono_18 
    mono_90 = mono_1 * mono_19 
    mono_91 = mono_2 * mono_19 
    mono_92 = mono_6 * mono_17 
    mono_93 = mono_5 * mono_18 
    mono_94 = mono_4 * mono_19 
    mono_95 = mono_10 * mono_17 
    mono_96 = mono_11 * mono_17 
    mono_97 = mono_13 * mono_17 
    mono_98 = mono_14 * mono_17 
    mono_99 = mono_9 * mono_18 
    mono_100 = mono_11 * mono_18 
    mono_101 = mono_12 * mono_18 
    mono_102 = mono_14 * mono_18 
    mono_103 = mono_9 * mono_19 
    mono_104 = mono_10 * mono_19 
    mono_105 = mono_12 * mono_19 
    mono_106 = mono_13 * mono_19 
    mono_107 = mono_17 * mono_18 
    mono_108 = mono_17 * mono_19 
    mono_109 = mono_18 * mono_19 
    mono_110 = mono_8 * mono_20 
    mono_111 = mono_7 * mono_21 
    mono_112 = mono_12 * mono_20 
    mono_113 = mono_13 * mono_20 
    mono_114 = mono_14 * mono_20 
    mono_115 = mono_9 * mono_21 
    mono_116 = mono_10 * mono_21 
    mono_117 = mono_11 * mono_21 
    mono_118 = mono_20 * mono_21 
    mono_119 = mono_2 * mono_23 
    mono_120 = mono_3 * mono_23 
    mono_121 = mono_1 * mono_24 
    mono_122 = mono_3 * mono_24 
    mono_123 = mono_1 * mono_25 
    mono_124 = mono_2 * mono_25 
    mono_125 = mono_6 * mono_23 
    mono_126 = mono_5 * mono_24 
    mono_127 = mono_4 * mono_25 
    mono_128 = mono_10 * mono_23 
    mono_129 = mono_11 * mono_23 
    mono_130 = mono_13 * mono_23 
    mono_131 = mono_14 * mono_23 
    mono_132 = mono_9 * mono_24 
    mono_133 = mono_11 * mono_24 
    mono_134 = mono_12 * mono_24 
    mono_135 = mono_14 * mono_24 
    mono_136 = mono_9 * mono_25 
    mono_137 = mono_10 * mono_25 
    mono_138 = mono_12 * mono_25 
    mono_139 = mono_13 * mono_25 
    mono_140 = mono_18 * mono_23 
    mono_141 = mono_19 * mono_23 
    mono_142 = mono_17 * mono_24 
    mono_143 = mono_19 * mono_24 
    mono_144 = mono_17 * mono_25 
    mono_145 = mono_18 * mono_25 
    mono_146 = mono_23 * mono_24 
    mono_147 = mono_23 * mono_25 
    mono_148 = mono_24 * mono_25 
    mono_149 = mono_8 * mono_26 
    mono_150 = mono_7 * mono_27 
    mono_151 = mono_12 * mono_26 
    mono_152 = mono_13 * mono_26 
    mono_153 = mono_14 * mono_26 
    mono_154 = mono_9 * mono_27 
    mono_155 = mono_10 * mono_27 
    mono_156 = mono_11 * mono_27 
    mono_157 = mono_21 * mono_26 
    mono_158 = mono_20 * mono_27 
    mono_159 = mono_26 * mono_27 
    mono_160 = mono_2 * mono_30 
    mono_161 = mono_3 * mono_30 
    mono_162 = mono_1 * mono_31 
    mono_163 = mono_3 * mono_31 
    mono_164 = mono_1 * mono_32 
    mono_165 = mono_2 * mono_32 
    mono_166 = mono_6 * mono_30 
    mono_167 = mono_5 * mono_31 
    mono_168 = mono_4 * mono_32 
    mono_169 = mono_10 * mono_30 
    mono_170 = mono_11 * mono_30 
    mono_171 = mono_13 * mono_30 
    mono_172 = mono_14 * mono_30 
    mono_173 = mono_9 * mono_31 
    mono_174 = mono_11 * mono_31 
    mono_175 = mono_12 * mono_31 
    mono_176 = mono_14 * mono_31 
    mono_177 = mono_9 * mono_32 
    mono_178 = mono_10 * mono_32 
    mono_179 = mono_12 * mono_32 
    mono_180 = mono_13 * mono_32 
    mono_181 = mono_18 * mono_30 
    mono_182 = mono_19 * mono_30 
    mono_183 = mono_17 * mono_31 
    mono_184 = mono_19 * mono_31 
    mono_185 = mono_17 * mono_32 
    mono_186 = mono_18 * mono_32 
    mono_187 = mono_24 * mono_30 
    mono_188 = mono_25 * mono_30 
    mono_189 = mono_23 * mono_31 
    mono_190 = mono_25 * mono_31 
    mono_191 = mono_23 * mono_32 
    mono_192 = mono_24 * mono_32 
    mono_193 = mono_30 * mono_31 
    mono_194 = mono_30 * mono_32 
    mono_195 = mono_31 * mono_32 
    mono_196 = mono_8 * mono_33 
    mono_197 = mono_7 * mono_34 
    mono_198 = mono_12 * mono_33 
    mono_199 = mono_13 * mono_33 
    mono_200 = mono_14 * mono_33 
    mono_201 = mono_9 * mono_34 
    mono_202 = mono_10 * mono_34 
    mono_203 = mono_11 * mono_34 
    mono_204 = mono_21 * mono_33 
    mono_205 = mono_20 * mono_34 
    mono_206 = mono_27 * mono_33 
    mono_207 = mono_26 * mono_34 
    mono_208 = mono_33 * mono_34 

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
    mono_176,    mono_177,    mono_178,    mono_179,    mono_180, 
    mono_181,    mono_182,    mono_183,    mono_184,    mono_185, 
    mono_186,    mono_187,    mono_188,    mono_189,    mono_190, 
    mono_191,    mono_192,    mono_193,    mono_194,    mono_195, 
    mono_196,    mono_197,    mono_198,    mono_199,    mono_200, 
    mono_201,    mono_202,    mono_203,    mono_204,    mono_205, 
    mono_206,    mono_207,    mono_208,    ]) 

    return mono 



