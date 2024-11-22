import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABCDE.monomials_MOL_1_1_1_1_1_3 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_1_1_3.POLY 

N_POLYS = 286

# Total number of monomials = 286 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(286) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = jnp.take(mono,4) 
    poly_5 = jnp.take(mono,5) 
    poly_6 = jnp.take(mono,6) 
    poly_7 = jnp.take(mono,7) 
    poly_8 = jnp.take(mono,8) 
    poly_9 = jnp.take(mono,9) 
    poly_10 = jnp.take(mono,10) 
    poly_11 = poly_1 * poly_2 
    poly_12 = poly_1 * poly_3 
    poly_13 = poly_2 * poly_3 
    poly_14 = poly_1 * poly_4 
    poly_15 = poly_2 * poly_4 
    poly_16 = poly_3 * poly_4 
    poly_17 = poly_1 * poly_5 
    poly_18 = poly_2 * poly_5 
    poly_19 = poly_3 * poly_5 
    poly_20 = poly_4 * poly_5 
    poly_21 = poly_1 * poly_6 
    poly_22 = poly_2 * poly_6 
    poly_23 = poly_3 * poly_6 
    poly_24 = poly_4 * poly_6 
    poly_25 = poly_5 * poly_6 
    poly_26 = poly_1 * poly_7 
    poly_27 = poly_2 * poly_7 
    poly_28 = poly_3 * poly_7 
    poly_29 = poly_4 * poly_7 
    poly_30 = poly_5 * poly_7 
    poly_31 = poly_6 * poly_7 
    poly_32 = poly_1 * poly_8 
    poly_33 = poly_2 * poly_8 
    poly_34 = poly_3 * poly_8 
    poly_35 = poly_4 * poly_8 
    poly_36 = poly_5 * poly_8 
    poly_37 = poly_6 * poly_8 
    poly_38 = poly_7 * poly_8 
    poly_39 = poly_1 * poly_9 
    poly_40 = poly_2 * poly_9 
    poly_41 = poly_3 * poly_9 
    poly_42 = poly_4 * poly_9 
    poly_43 = poly_5 * poly_9 
    poly_44 = poly_6 * poly_9 
    poly_45 = poly_7 * poly_9 
    poly_46 = poly_8 * poly_9 
    poly_47 = poly_1 * poly_10 
    poly_48 = poly_2 * poly_10 
    poly_49 = poly_3 * poly_10 
    poly_50 = poly_4 * poly_10 
    poly_51 = poly_5 * poly_10 
    poly_52 = poly_6 * poly_10 
    poly_53 = poly_7 * poly_10 
    poly_54 = poly_8 * poly_10 
    poly_55 = poly_9 * poly_10 
    poly_56 = poly_1 * poly_1 
    poly_57 = poly_2 * poly_2 
    poly_58 = poly_3 * poly_3 
    poly_59 = poly_4 * poly_4 
    poly_60 = poly_5 * poly_5 
    poly_61 = poly_6 * poly_6 
    poly_62 = poly_7 * poly_7 
    poly_63 = poly_8 * poly_8 
    poly_64 = poly_9 * poly_9 
    poly_65 = poly_10 * poly_10 
    poly_66 = poly_1 * poly_13 
    poly_67 = poly_1 * poly_15 
    poly_68 = poly_1 * poly_16 
    poly_69 = poly_2 * poly_16 
    poly_70 = poly_1 * poly_18 
    poly_71 = poly_1 * poly_19 
    poly_72 = poly_2 * poly_19 
    poly_73 = poly_1 * poly_20 
    poly_74 = poly_2 * poly_20 
    poly_75 = poly_3 * poly_20 
    poly_76 = poly_1 * poly_22 
    poly_77 = poly_1 * poly_23 
    poly_78 = poly_2 * poly_23 
    poly_79 = poly_1 * poly_24 
    poly_80 = poly_2 * poly_24 
    poly_81 = poly_3 * poly_24 
    poly_82 = poly_1 * poly_25 
    poly_83 = poly_2 * poly_25 
    poly_84 = poly_3 * poly_25 
    poly_85 = poly_4 * poly_25 
    poly_86 = poly_1 * poly_27 
    poly_87 = poly_1 * poly_28 
    poly_88 = poly_2 * poly_28 
    poly_89 = poly_1 * poly_29 
    poly_90 = poly_2 * poly_29 
    poly_91 = poly_3 * poly_29 
    poly_92 = poly_1 * poly_30 
    poly_93 = poly_2 * poly_30 
    poly_94 = poly_3 * poly_30 
    poly_95 = poly_4 * poly_30 
    poly_96 = poly_1 * poly_31 
    poly_97 = poly_2 * poly_31 
    poly_98 = poly_3 * poly_31 
    poly_99 = poly_4 * poly_31 
    poly_100 = poly_5 * poly_31 
    poly_101 = poly_1 * poly_33 
    poly_102 = poly_1 * poly_34 
    poly_103 = poly_2 * poly_34 
    poly_104 = poly_1 * poly_35 
    poly_105 = poly_2 * poly_35 
    poly_106 = poly_3 * poly_35 
    poly_107 = poly_1 * poly_36 
    poly_108 = poly_2 * poly_36 
    poly_109 = poly_3 * poly_36 
    poly_110 = poly_4 * poly_36 
    poly_111 = poly_1 * poly_37 
    poly_112 = poly_2 * poly_37 
    poly_113 = poly_3 * poly_37 
    poly_114 = poly_4 * poly_37 
    poly_115 = poly_5 * poly_37 
    poly_116 = poly_1 * poly_38 
    poly_117 = poly_2 * poly_38 
    poly_118 = poly_3 * poly_38 
    poly_119 = poly_4 * poly_38 
    poly_120 = poly_5 * poly_38 
    poly_121 = poly_6 * poly_38 
    poly_122 = poly_1 * poly_40 
    poly_123 = poly_1 * poly_41 
    poly_124 = poly_2 * poly_41 
    poly_125 = poly_1 * poly_42 
    poly_126 = poly_2 * poly_42 
    poly_127 = poly_3 * poly_42 
    poly_128 = poly_1 * poly_43 
    poly_129 = poly_2 * poly_43 
    poly_130 = poly_3 * poly_43 
    poly_131 = poly_4 * poly_43 
    poly_132 = poly_1 * poly_44 
    poly_133 = poly_2 * poly_44 
    poly_134 = poly_3 * poly_44 
    poly_135 = poly_4 * poly_44 
    poly_136 = poly_5 * poly_44 
    poly_137 = poly_1 * poly_45 
    poly_138 = poly_2 * poly_45 
    poly_139 = poly_3 * poly_45 
    poly_140 = poly_4 * poly_45 
    poly_141 = poly_5 * poly_45 
    poly_142 = poly_6 * poly_45 
    poly_143 = poly_1 * poly_46 
    poly_144 = poly_2 * poly_46 
    poly_145 = poly_3 * poly_46 
    poly_146 = poly_4 * poly_46 
    poly_147 = poly_5 * poly_46 
    poly_148 = poly_6 * poly_46 
    poly_149 = poly_7 * poly_46 
    poly_150 = poly_1 * poly_48 
    poly_151 = poly_1 * poly_49 
    poly_152 = poly_2 * poly_49 
    poly_153 = poly_1 * poly_50 
    poly_154 = poly_2 * poly_50 
    poly_155 = poly_3 * poly_50 
    poly_156 = poly_1 * poly_51 
    poly_157 = poly_2 * poly_51 
    poly_158 = poly_3 * poly_51 
    poly_159 = poly_4 * poly_51 
    poly_160 = poly_1 * poly_52 
    poly_161 = poly_2 * poly_52 
    poly_162 = poly_3 * poly_52 
    poly_163 = poly_4 * poly_52 
    poly_164 = poly_5 * poly_52 
    poly_165 = poly_1 * poly_53 
    poly_166 = poly_2 * poly_53 
    poly_167 = poly_3 * poly_53 
    poly_168 = poly_4 * poly_53 
    poly_169 = poly_5 * poly_53 
    poly_170 = poly_6 * poly_53 
    poly_171 = poly_1 * poly_54 
    poly_172 = poly_2 * poly_54 
    poly_173 = poly_3 * poly_54 
    poly_174 = poly_4 * poly_54 
    poly_175 = poly_5 * poly_54 
    poly_176 = poly_6 * poly_54 
    poly_177 = poly_7 * poly_54 
    poly_178 = poly_1 * poly_55 
    poly_179 = poly_2 * poly_55 
    poly_180 = poly_3 * poly_55 
    poly_181 = poly_4 * poly_55 
    poly_182 = poly_5 * poly_55 
    poly_183 = poly_6 * poly_55 
    poly_184 = poly_7 * poly_55 
    poly_185 = poly_8 * poly_55 
    poly_186 = poly_1 * poly_11 
    poly_187 = poly_1 * poly_57 
    poly_188 = poly_1 * poly_12 
    poly_189 = poly_2 * poly_13 
    poly_190 = poly_1 * poly_58 
    poly_191 = poly_2 * poly_58 
    poly_192 = poly_1 * poly_14 
    poly_193 = poly_2 * poly_15 
    poly_194 = poly_3 * poly_16 
    poly_195 = poly_1 * poly_59 
    poly_196 = poly_2 * poly_59 
    poly_197 = poly_3 * poly_59 
    poly_198 = poly_1 * poly_17 
    poly_199 = poly_2 * poly_18 
    poly_200 = poly_3 * poly_19 
    poly_201 = poly_4 * poly_20 
    poly_202 = poly_1 * poly_60 
    poly_203 = poly_2 * poly_60 
    poly_204 = poly_3 * poly_60 
    poly_205 = poly_4 * poly_60 
    poly_206 = poly_1 * poly_21 
    poly_207 = poly_2 * poly_22 
    poly_208 = poly_3 * poly_23 
    poly_209 = poly_4 * poly_24 
    poly_210 = poly_5 * poly_25 
    poly_211 = poly_1 * poly_61 
    poly_212 = poly_2 * poly_61 
    poly_213 = poly_3 * poly_61 
    poly_214 = poly_4 * poly_61 
    poly_215 = poly_5 * poly_61 
    poly_216 = poly_1 * poly_26 
    poly_217 = poly_2 * poly_27 
    poly_218 = poly_3 * poly_28 
    poly_219 = poly_4 * poly_29 
    poly_220 = poly_5 * poly_30 
    poly_221 = poly_6 * poly_31 
    poly_222 = poly_1 * poly_62 
    poly_223 = poly_2 * poly_62 
    poly_224 = poly_3 * poly_62 
    poly_225 = poly_4 * poly_62 
    poly_226 = poly_5 * poly_62 
    poly_227 = poly_6 * poly_62 
    poly_228 = poly_1 * poly_32 
    poly_229 = poly_2 * poly_33 
    poly_230 = poly_3 * poly_34 
    poly_231 = poly_4 * poly_35 
    poly_232 = poly_5 * poly_36 
    poly_233 = poly_6 * poly_37 
    poly_234 = poly_7 * poly_38 
    poly_235 = poly_1 * poly_63 
    poly_236 = poly_2 * poly_63 
    poly_237 = poly_3 * poly_63 
    poly_238 = poly_4 * poly_63 
    poly_239 = poly_5 * poly_63 
    poly_240 = poly_6 * poly_63 
    poly_241 = poly_7 * poly_63 
    poly_242 = poly_1 * poly_39 
    poly_243 = poly_2 * poly_40 
    poly_244 = poly_3 * poly_41 
    poly_245 = poly_4 * poly_42 
    poly_246 = poly_5 * poly_43 
    poly_247 = poly_6 * poly_44 
    poly_248 = poly_7 * poly_45 
    poly_249 = poly_8 * poly_46 
    poly_250 = poly_1 * poly_64 
    poly_251 = poly_2 * poly_64 
    poly_252 = poly_3 * poly_64 
    poly_253 = poly_4 * poly_64 
    poly_254 = poly_5 * poly_64 
    poly_255 = poly_6 * poly_64 
    poly_256 = poly_7 * poly_64 
    poly_257 = poly_8 * poly_64 
    poly_258 = poly_1 * poly_47 
    poly_259 = poly_2 * poly_48 
    poly_260 = poly_3 * poly_49 
    poly_261 = poly_4 * poly_50 
    poly_262 = poly_5 * poly_51 
    poly_263 = poly_6 * poly_52 
    poly_264 = poly_7 * poly_53 
    poly_265 = poly_8 * poly_54 
    poly_266 = poly_9 * poly_55 
    poly_267 = poly_1 * poly_65 
    poly_268 = poly_2 * poly_65 
    poly_269 = poly_3 * poly_65 
    poly_270 = poly_4 * poly_65 
    poly_271 = poly_5 * poly_65 
    poly_272 = poly_6 * poly_65 
    poly_273 = poly_7 * poly_65 
    poly_274 = poly_8 * poly_65 
    poly_275 = poly_9 * poly_65 
    poly_276 = poly_1 * poly_56 
    poly_277 = poly_2 * poly_57 
    poly_278 = poly_3 * poly_58 
    poly_279 = poly_4 * poly_59 
    poly_280 = poly_5 * poly_60 
    poly_281 = poly_6 * poly_61 
    poly_282 = poly_7 * poly_62 
    poly_283 = poly_8 * poly_63 
    poly_284 = poly_9 * poly_64 
    poly_285 = poly_10 * poly_65 

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
    poly_256,    poly_257,    poly_258,    poly_259,    poly_260, 
    poly_261,    poly_262,    poly_263,    poly_264,    poly_265, 
    poly_266,    poly_267,    poly_268,    poly_269,    poly_270, 
    poly_271,    poly_272,    poly_273,    poly_274,    poly_275, 
    poly_276,    poly_277,    poly_278,    poly_279,    poly_280, 
    poly_281,    poly_282,    poly_283,    poly_284,    poly_285, 
    ]) 

    return poly 



