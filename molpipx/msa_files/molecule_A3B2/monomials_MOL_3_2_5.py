import jax 
import jax.numpy as jnp 
from jax import jit

# File created from ./MOL_3_2_5.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 254 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(254) 

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
    mono_95 = mono_2 * mono_37 
    mono_96 = mono_3 * mono_48 
    mono_97 = mono_2 * mono_39 
    mono_98 = mono_2 * mono_50 
    mono_99 = mono_3 * mono_53 
    mono_100 = mono_2 * mono_54 
    mono_101 = mono_2 * mono_35 
    mono_102 = mono_2 * mono_48 
    mono_103 = mono_2 * mono_42 
    mono_104 = mono_3 * mono_50 
    mono_105 = mono_2 * mono_53 
    mono_106 = mono_3 * mono_54 
    mono_107 = mono_2 * mono_46 
    mono_108 = mono_2 * mono_52 
    mono_109 = mono_4 * mono_54 
    mono_110 = mono_2 * mono_55 
    mono_111 = mono_3 * mono_55 
    mono_112 = mono_4 * mono_55 
    mono_113 = mono_5 * mono_55 
    mono_114 = mono_2 * mono_56 
    mono_115 = mono_3 * mono_56 
    mono_116 = mono_4 * mono_64 
    mono_117 = mono_4 * mono_77 
    mono_118 = mono_2 * mono_66 
    mono_119 = mono_2 * mono_79 
    mono_120 = mono_2 * mono_68 
    mono_121 = mono_2 * mono_81 
    mono_122 = mono_3 * mono_71 
    mono_123 = mono_2 * mono_59 
    mono_124 = mono_3 * mono_61 
    mono_125 = mono_2 * mono_73 
    mono_126 = mono_3 * mono_76 
    mono_127 = mono_3 * mono_64 
    mono_128 = mono_2 * mono_65 
    mono_129 = mono_2 * mono_77 
    mono_130 = mono_5 * mono_80 
    mono_131 = mono_5 * mono_68 
    mono_132 = mono_4 * mono_69 
    mono_133 = mono_4 * mono_81 
    mono_134 = mono_2 * mono_58 
    mono_135 = mono_4 * mono_59 
    mono_136 = mono_2 * mono_72 
    mono_137 = mono_4 * mono_73 
    mono_138 = mono_2 * mono_62 
    mono_139 = mono_2 * mono_75 
    mono_140 = mono_6 * mono_65 
    mono_141 = mono_6 * mono_77 
    mono_142 = mono_4 * mono_67 
    mono_143 = mono_4 * mono_79 
    mono_144 = mono_6 * mono_69 
    mono_145 = mono_6 * mono_81 
    mono_146 = mono_5 * mono_84 
    mono_147 = mono_4 * mono_85 
    mono_148 = mono_3 * mono_88 
    mono_149 = mono_2 * mono_89 
    mono_150 = mono_3 * mono_92 
    mono_151 = mono_2 * mono_93 
    mono_152 = mono_4 * mono_84 
    mono_153 = mono_5 * mono_85 
    mono_154 = mono_2 * mono_88 
    mono_155 = mono_3 * mono_89 
    mono_156 = mono_2 * mono_92 
    mono_157 = mono_3 * mono_93 
    mono_158 = mono_4 * mono_83 
    mono_159 = mono_6 * mono_85 
    mono_160 = mono_2 * mono_87 
    mono_161 = mono_6 * mono_89 
    mono_162 = mono_2 * mono_91 
    mono_163 = mono_4 * mono_93 
    mono_164 = mono_3 * mono_35 
    mono_165 = mono_5 * mono_36 
    mono_166 = mono_6 * mono_37 
    mono_167 = mono_4 * mono_39 
    mono_168 = mono_2 * mono_40 
    mono_169 = mono_7 * mono_38 
    mono_170 = mono_2 * mono_96 
    mono_171 = mono_2 * mono_104 
    mono_172 = mono_2 * mono_99 
    mono_173 = mono_2 * mono_106 
    mono_174 = mono_2 * mono_109 
    mono_175 = mono_3 * mono_109 
    mono_176 = mono_3 * mono_112 
    mono_177 = mono_2 * mono_113 
    mono_178 = mono_3 * mono_116 
    mono_179 = mono_2 * mono_117 
    mono_180 = mono_2 * mono_131 
    mono_181 = mono_2 * mono_133 
    mono_182 = mono_2 * mono_112 
    mono_183 = mono_3 * mono_113 
    mono_184 = mono_2 * mono_116 
    mono_185 = mono_3 * mono_117 
    mono_186 = mono_4 * mono_120 
    mono_187 = mono_5 * mono_121 
    mono_188 = mono_2 * mono_111 
    mono_189 = mono_4 * mono_113 
    mono_190 = mono_2 * mono_115 
    mono_191 = mono_4 * mono_141 
    mono_192 = mono_2 * mono_143 
    mono_193 = mono_2 * mono_145 
    mono_194 = mono_4 * mono_146 
    mono_195 = mono_4 * mono_153 
    mono_196 = mono_4 * mono_159 
    mono_197 = mono_5 * mono_159 
    mono_198 = mono_2 * mono_148 
    mono_199 = mono_2 * mono_155 
    mono_200 = mono_2 * mono_161 
    mono_201 = mono_3 * mono_161 
    mono_202 = mono_2 * mono_150 
    mono_203 = mono_2 * mono_157 
    mono_204 = mono_2 * mono_163 
    mono_205 = mono_3 * mono_163 
    mono_206 = mono_3 * mono_152 
    mono_207 = mono_2 * mono_153 
    mono_208 = mono_5 * mono_154 
    mono_209 = mono_4 * mono_155 
    mono_210 = mono_6 * mono_157 
    mono_211 = mono_7 * mono_156 
    mono_212 = mono_2 * mono_158 
    mono_213 = mono_3 * mono_158 
    mono_214 = mono_2 * mono_159 
    mono_215 = mono_3 * mono_159 
    mono_216 = mono_4 * mono_160 
    mono_217 = mono_5 * mono_160 
    mono_218 = mono_4 * mono_161 
    mono_219 = mono_5 * mono_161 
    mono_220 = mono_6 * mono_162 
    mono_221 = mono_6 * mono_163 
    mono_222 = mono_7 * mono_162 
    mono_223 = mono_7 * mono_163 
    mono_224 = mono_3 * mono_96 
    mono_225 = mono_3 * mono_165 
    mono_226 = mono_2 * mono_166 
    mono_227 = mono_4 * mono_166 
    mono_228 = mono_2 * mono_167 
    mono_229 = mono_2 * mono_98 
    mono_230 = mono_3 * mono_99 
    mono_231 = mono_4 * mono_99 
    mono_232 = mono_2 * mono_100 
    mono_233 = mono_5 * mono_100 
    mono_234 = mono_3 * mono_169 
    mono_235 = mono_5 * mono_169 
    mono_236 = mono_2 * mono_164 
    mono_237 = mono_4 * mono_165 
    mono_238 = mono_2 * mono_103 
    mono_239 = mono_4 * mono_104 
    mono_240 = mono_6 * mono_106 
    mono_241 = mono_6 * mono_169 
    mono_242 = mono_6 * mono_122 
    mono_243 = mono_6 * mono_123 
    mono_244 = mono_7 * mono_124 
    mono_245 = mono_7 * mono_125 
    mono_246 = mono_4 * mono_126 
    mono_247 = mono_5 * mono_127 
    mono_248 = mono_4 * mono_128 
    mono_249 = mono_5 * mono_129 
    mono_250 = mono_2 * mono_130 
    mono_251 = mono_3 * mono_131 
    mono_252 = mono_2 * mono_132 
    mono_253 = mono_3 * mono_133 

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
    mono_206,    mono_207,    mono_208,    mono_209,    mono_210, 
    mono_211,    mono_212,    mono_213,    mono_214,    mono_215, 
    mono_216,    mono_217,    mono_218,    mono_219,    mono_220, 
    mono_221,    mono_222,    mono_223,    mono_224,    mono_225, 
    mono_226,    mono_227,    mono_228,    mono_229,    mono_230, 
    mono_231,    mono_232,    mono_233,    mono_234,    mono_235, 
    mono_236,    mono_237,    mono_238,    mono_239,    mono_240, 
    mono_241,    mono_242,    mono_243,    mono_244,    mono_245, 
    mono_246,    mono_247,    mono_248,    mono_249,    mono_250, 
    mono_251,    mono_252,    mono_253,    ]) 

    return mono 



