import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A2BC.monomials_MOL_2_1_1_5 import f_monomials as f_monos 

# File created from ./MOL_2_1_1_5.POLY 

N_POLYS = 256

# Total number of monomials = 256 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(256) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) + jnp.take(mono,3) 
    poly_3 = jnp.take(mono,4) + jnp.take(mono,5) 
    poly_4 = jnp.take(mono,6) 
    poly_5 = poly_1 * poly_2 
    poly_6 = jnp.take(mono,7) 
    poly_7 = poly_1 * poly_3 
    poly_8 = jnp.take(mono,8) + jnp.take(mono,9) 
    poly_9 = jnp.take(mono,10) 
    poly_10 = poly_2 * poly_3 - poly_8 
    poly_11 = poly_1 * poly_4 
    poly_12 = poly_4 * poly_2 
    poly_13 = poly_4 * poly_3 
    poly_14 = poly_1 * poly_1 
    poly_15 = poly_2 * poly_2 - poly_6 - poly_6 
    poly_16 = poly_3 * poly_3 - poly_9 - poly_9 
    poly_17 = poly_4 * poly_4 
    poly_18 = poly_1 * poly_6 
    poly_19 = poly_1 * poly_8 
    poly_20 = poly_1 * poly_9 
    poly_21 = poly_1 * poly_10 
    poly_22 = poly_6 * poly_3 
    poly_23 = poly_9 * poly_2 
    poly_24 = poly_1 * poly_12 
    poly_25 = poly_4 * poly_6 
    poly_26 = poly_1 * poly_13 
    poly_27 = poly_4 * poly_8 
    poly_28 = poly_4 * poly_9 
    poly_29 = poly_4 * poly_10 
    poly_30 = poly_1 * poly_5 
    poly_31 = poly_1 * poly_15 
    poly_32 = poly_6 * poly_2 
    poly_33 = poly_1 * poly_7 
    poly_34 = poly_2 * poly_8 - poly_22 
    poly_35 = poly_2 * poly_10 - poly_22 
    poly_36 = poly_1 * poly_16 
    poly_37 = poly_3 * poly_8 - poly_23 
    poly_38 = poly_9 * poly_3 
    poly_39 = poly_2 * poly_16 - poly_37 
    poly_40 = poly_1 * poly_11 
    poly_41 = poly_4 * poly_15 
    poly_42 = poly_4 * poly_16 
    poly_43 = poly_1 * poly_17 
    poly_44 = poly_4 * poly_12 
    poly_45 = poly_4 * poly_13 
    poly_46 = poly_1 * poly_14 
    poly_47 = poly_2 * poly_15 - poly_32 
    poly_48 = poly_3 * poly_16 - poly_38 
    poly_49 = poly_4 * poly_17 
    poly_50 = poly_1 * poly_22 
    poly_51 = poly_1 * poly_23 
    poly_52 = poly_6 * poly_9 
    poly_53 = poly_1 * poly_25 
    poly_54 = poly_1 * poly_27 
    poly_55 = poly_1 * poly_28 
    poly_56 = poly_1 * poly_29 
    poly_57 = poly_4 * poly_22 
    poly_58 = poly_4 * poly_23 
    poly_59 = poly_1 * poly_18 
    poly_60 = poly_1 * poly_32 
    poly_61 = poly_1 * poly_19 
    poly_62 = poly_1 * poly_34 
    poly_63 = poly_1 * poly_20 
    poly_64 = poly_1 * poly_21 
    poly_65 = poly_6 * poly_8 
    poly_66 = poly_1 * poly_35 
    poly_67 = poly_6 * poly_10 
    poly_68 = poly_9 * poly_15 
    poly_69 = poly_1 * poly_37 
    poly_70 = poly_1 * poly_38 
    poly_71 = poly_9 * poly_8 
    poly_72 = poly_1 * poly_39 
    poly_73 = poly_6 * poly_16 
    poly_74 = poly_9 * poly_10 
    poly_75 = poly_1 * poly_24 
    poly_76 = poly_1 * poly_41 
    poly_77 = poly_4 * poly_32 
    poly_78 = poly_1 * poly_26 
    poly_79 = poly_4 * poly_34 
    poly_80 = poly_4 * poly_35 
    poly_81 = poly_1 * poly_42 
    poly_82 = poly_4 * poly_37 
    poly_83 = poly_4 * poly_38 
    poly_84 = poly_4 * poly_39 
    poly_85 = poly_1 * poly_44 
    poly_86 = poly_4 * poly_25 
    poly_87 = poly_1 * poly_45 
    poly_88 = poly_4 * poly_27 
    poly_89 = poly_4 * poly_28 
    poly_90 = poly_4 * poly_29 
    poly_91 = poly_1 * poly_30 
    poly_92 = poly_1 * poly_31 
    poly_93 = poly_6 * poly_6 
    poly_94 = poly_1 * poly_47 
    poly_95 = poly_6 * poly_15 
    poly_96 = poly_1 * poly_33 
    poly_97 = poly_2 * poly_34 - poly_65 
    poly_98 = poly_2 * poly_35 - poly_67 
    poly_99 = poly_1 * poly_36 
    poly_100 = poly_2 * poly_37 - poly_73 
    poly_101 = poly_9 * poly_9 
    poly_102 = poly_2 * poly_39 - poly_73 
    poly_103 = poly_1 * poly_48 
    poly_104 = poly_3 * poly_37 - poly_71 
    poly_105 = poly_9 * poly_16 
    poly_106 = poly_2 * poly_48 - poly_104 
    poly_107 = poly_1 * poly_40 
    poly_108 = poly_4 * poly_47 
    poly_109 = poly_4 * poly_48 
    poly_110 = poly_1 * poly_43 
    poly_111 = poly_4 * poly_41 
    poly_112 = poly_4 * poly_42 
    poly_113 = poly_1 * poly_49 
    poly_114 = poly_4 * poly_44 
    poly_115 = poly_4 * poly_45 
    poly_116 = poly_1 * poly_46 
    poly_117 = poly_2 * poly_47 - poly_95 
    poly_118 = poly_3 * poly_48 - poly_105 
    poly_119 = poly_4 * poly_49 
    poly_120 = poly_1 * poly_52 
    poly_121 = poly_1 * poly_57 
    poly_122 = poly_1 * poly_58 
    poly_123 = poly_4 * poly_52 
    poly_124 = poly_1 * poly_50 
    poly_125 = poly_1 * poly_65 
    poly_126 = poly_1 * poly_51 
    poly_127 = poly_1 * poly_67 
    poly_128 = poly_1 * poly_68 
    poly_129 = poly_6 * poly_23 
    poly_130 = poly_1 * poly_71 
    poly_131 = poly_1 * poly_73 
    poly_132 = poly_1 * poly_74 
    poly_133 = poly_6 * poly_38 
    poly_134 = poly_1 * poly_53 
    poly_135 = poly_1 * poly_77 
    poly_136 = poly_1 * poly_54 
    poly_137 = poly_1 * poly_79 
    poly_138 = poly_1 * poly_55 
    poly_139 = poly_1 * poly_56 
    poly_140 = poly_4 * poly_65 
    poly_141 = poly_1 * poly_80 
    poly_142 = poly_4 * poly_67 
    poly_143 = poly_4 * poly_68 
    poly_144 = poly_1 * poly_82 
    poly_145 = poly_1 * poly_83 
    poly_146 = poly_4 * poly_71 
    poly_147 = poly_1 * poly_84 
    poly_148 = poly_4 * poly_73 
    poly_149 = poly_4 * poly_74 
    poly_150 = poly_1 * poly_86 
    poly_151 = poly_1 * poly_88 
    poly_152 = poly_1 * poly_89 
    poly_153 = poly_1 * poly_90 
    poly_154 = poly_4 * poly_57 
    poly_155 = poly_4 * poly_58 
    poly_156 = poly_1 * poly_59 
    poly_157 = poly_1 * poly_60 
    poly_158 = poly_1 * poly_93 
    poly_159 = poly_1 * poly_95 
    poly_160 = poly_1 * poly_61 
    poly_161 = poly_1 * poly_62 
    poly_162 = poly_1 * poly_97 
    poly_163 = poly_1 * poly_63 
    poly_164 = poly_1 * poly_64 
    poly_165 = poly_6 * poly_34 
    poly_166 = poly_1 * poly_66 
    poly_167 = poly_6 * poly_22 
    poly_168 = poly_1 * poly_98 
    poly_169 = poly_6 * poly_35 
    poly_170 = poly_9 * poly_47 
    poly_171 = poly_1 * poly_69 
    poly_172 = poly_1 * poly_100 
    poly_173 = poly_1 * poly_70 
    poly_174 = poly_9 * poly_34 
    poly_175 = poly_1 * poly_101 
    poly_176 = poly_1 * poly_72 
    poly_177 = poly_6 * poly_37 
    poly_178 = poly_9 * poly_23 
    poly_179 = poly_1 * poly_102 
    poly_180 = poly_6 * poly_39 
    poly_181 = poly_9 * poly_35 
    poly_182 = poly_1 * poly_104 
    poly_183 = poly_1 * poly_105 
    poly_184 = poly_9 * poly_37 
    poly_185 = poly_1 * poly_106 
    poly_186 = poly_6 * poly_48 
    poly_187 = poly_9 * poly_39 
    poly_188 = poly_1 * poly_75 
    poly_189 = poly_1 * poly_76 
    poly_190 = poly_4 * poly_93 
    poly_191 = poly_1 * poly_108 
    poly_192 = poly_4 * poly_95 
    poly_193 = poly_1 * poly_78 
    poly_194 = poly_4 * poly_97 
    poly_195 = poly_4 * poly_98 
    poly_196 = poly_1 * poly_81 
    poly_197 = poly_4 * poly_100 
    poly_198 = poly_4 * poly_101 
    poly_199 = poly_4 * poly_102 
    poly_200 = poly_1 * poly_109 
    poly_201 = poly_4 * poly_104 
    poly_202 = poly_4 * poly_105 
    poly_203 = poly_4 * poly_106 
    poly_204 = poly_1 * poly_85 
    poly_205 = poly_1 * poly_111 
    poly_206 = poly_4 * poly_77 
    poly_207 = poly_1 * poly_87 
    poly_208 = poly_4 * poly_79 
    poly_209 = poly_4 * poly_80 
    poly_210 = poly_1 * poly_112 
    poly_211 = poly_4 * poly_82 
    poly_212 = poly_4 * poly_83 
    poly_213 = poly_4 * poly_84 
    poly_214 = poly_1 * poly_114 
    poly_215 = poly_4 * poly_86 
    poly_216 = poly_1 * poly_115 
    poly_217 = poly_4 * poly_88 
    poly_218 = poly_4 * poly_89 
    poly_219 = poly_4 * poly_90 
    poly_220 = poly_1 * poly_91 
    poly_221 = poly_1 * poly_92 
    poly_222 = poly_1 * poly_94 
    poly_223 = poly_6 * poly_32 
    poly_224 = poly_1 * poly_117 
    poly_225 = poly_6 * poly_47 
    poly_226 = poly_1 * poly_96 
    poly_227 = poly_2 * poly_97 - poly_165 
    poly_228 = poly_2 * poly_98 - poly_169 
    poly_229 = poly_1 * poly_99 
    poly_230 = poly_2 * poly_100 - poly_177 
    poly_231 = poly_2 * poly_102 - poly_180 
    poly_232 = poly_1 * poly_103 
    poly_233 = poly_2 * poly_104 - poly_186 
    poly_234 = poly_9 * poly_38 
    poly_235 = poly_2 * poly_106 - poly_186 
    poly_236 = poly_1 * poly_118 
    poly_237 = poly_3 * poly_104 - poly_184 
    poly_238 = poly_9 * poly_48 
    poly_239 = poly_2 * poly_118 - poly_237 
    poly_240 = poly_1 * poly_107 
    poly_241 = poly_4 * poly_117 
    poly_242 = poly_4 * poly_118 
    poly_243 = poly_1 * poly_110 
    poly_244 = poly_4 * poly_108 
    poly_245 = poly_4 * poly_109 
    poly_246 = poly_1 * poly_113 
    poly_247 = poly_4 * poly_111 
    poly_248 = poly_4 * poly_112 
    poly_249 = poly_1 * poly_119 
    poly_250 = poly_4 * poly_114 
    poly_251 = poly_4 * poly_115 
    poly_252 = poly_1 * poly_116 
    poly_253 = poly_2 * poly_117 - poly_225 
    poly_254 = poly_3 * poly_118 - poly_238 
    poly_255 = poly_4 * poly_119 

#    stack all polynomials 
    poly = jnp.stack([    poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5, 
    poly_6,    poly_7,    poly_8,    poly_9,    poly_10, 
    poly_11,    poly_12,    poly_13,    poly_14,    poly_15, 
    poly_16,    poly_17,    poly_18,    poly_19,    poly_20, 
    poly_21,    poly_22,    poly_23,    poly_24,    poly_25, 
    poly_26,    poly_27,    poly_28,    poly_29,    poly_30, 
    poly_31,    poly_32,    poly_33,    poly_34,    poly_35, 
    poly_36,    poly_37,    poly_38,    poly_39,    poly_40, 
    poly_41,    poly_42,    poly_43,    poly_44,    poly_45, 
    poly_46,    poly_47,    poly_48,    poly_49,    poly_50, 
    poly_51,    poly_52,    poly_53,    poly_54,    poly_55, 
    poly_56,    poly_57,    poly_58,    poly_59,    poly_60, 
    poly_61,    poly_62,    poly_63,    poly_64,    poly_65, 
    poly_66,    poly_67,    poly_68,    poly_69,    poly_70, 
    poly_71,    poly_72,    poly_73,    poly_74,    poly_75, 
    poly_76,    poly_77,    poly_78,    poly_79,    poly_80, 
    poly_81,    poly_82,    poly_83,    poly_84,    poly_85, 
    poly_86,    poly_87,    poly_88,    poly_89,    poly_90, 
    poly_91,    poly_92,    poly_93,    poly_94,    poly_95, 
    poly_96,    poly_97,    poly_98,    poly_99,    poly_100, 
    poly_101,    poly_102,    poly_103,    poly_104,    poly_105, 
    poly_106,    poly_107,    poly_108,    poly_109,    poly_110, 
    poly_111,    poly_112,    poly_113,    poly_114,    poly_115, 
    poly_116,    poly_117,    poly_118,    poly_119,    poly_120, 
    poly_121,    poly_122,    poly_123,    poly_124,    poly_125, 
    poly_126,    poly_127,    poly_128,    poly_129,    poly_130, 
    poly_131,    poly_132,    poly_133,    poly_134,    poly_135, 
    poly_136,    poly_137,    poly_138,    poly_139,    poly_140, 
    poly_141,    poly_142,    poly_143,    poly_144,    poly_145, 
    poly_146,    poly_147,    poly_148,    poly_149,    poly_150, 
    poly_151,    poly_152,    poly_153,    poly_154,    poly_155, 
    poly_156,    poly_157,    poly_158,    poly_159,    poly_160, 
    poly_161,    poly_162,    poly_163,    poly_164,    poly_165, 
    poly_166,    poly_167,    poly_168,    poly_169,    poly_170, 
    poly_171,    poly_172,    poly_173,    poly_174,    poly_175, 
    poly_176,    poly_177,    poly_178,    poly_179,    poly_180, 
    poly_181,    poly_182,    poly_183,    poly_184,    poly_185, 
    poly_186,    poly_187,    poly_188,    poly_189,    poly_190, 
    poly_191,    poly_192,    poly_193,    poly_194,    poly_195, 
    poly_196,    poly_197,    poly_198,    poly_199,    poly_200, 
    poly_201,    poly_202,    poly_203,    poly_204,    poly_205, 
    poly_206,    poly_207,    poly_208,    poly_209,    poly_210, 
    poly_211,    poly_212,    poly_213,    poly_214,    poly_215, 
    poly_216,    poly_217,    poly_218,    poly_219,    poly_220, 
    poly_221,    poly_222,    poly_223,    poly_224,    poly_225, 
    poly_226,    poly_227,    poly_228,    poly_229,    poly_230, 
    poly_231,    poly_232,    poly_233,    poly_234,    poly_235, 
    poly_236,    poly_237,    poly_238,    poly_239,    poly_240, 
    poly_241,    poly_242,    poly_243,    poly_244,    poly_245, 
    poly_246,    poly_247,    poly_248,    poly_249,    poly_250, 
    poly_251,    poly_252,    poly_253,    poly_254,    poly_255, 
    ]) 

    return poly 



