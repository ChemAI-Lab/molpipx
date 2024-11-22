import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2C.monomials_MOL_2_2_1_4 import f_monomials as f_monos

# File created from ./MOL_2_2_1_4.POLY

N_POLYS = 323

# Total number of monomials = 323


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(323)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2)
    poly_2 = jnp.take(mono, 3)
    poly_3 = jnp.take(mono, 4) + jnp.take(mono, 5)
    poly_4 = jnp.take(mono, 6) + jnp.take(mono, 7) + \
        jnp.take(mono, 8) + jnp.take(mono, 9)
    poly_5 = jnp.take(mono, 10)
    poly_6 = jnp.take(mono, 11)
    poly_7 = poly_2 * poly_1
    poly_8 = poly_1 * poly_3
    poly_9 = poly_2 * poly_3
    poly_10 = jnp.take(mono, 12)
    poly_11 = jnp.take(mono, 13) + jnp.take(mono, 14) + \
        jnp.take(mono, 15) + jnp.take(mono, 16)
    poly_12 = poly_1 * poly_4 - poly_11
    poly_13 = poly_2 * poly_4
    poly_14 = jnp.take(mono, 17) + jnp.take(mono, 18) + \
        jnp.take(mono, 19) + jnp.take(mono, 20)
    poly_15 = jnp.take(mono, 21) + jnp.take(mono, 22)
    poly_16 = jnp.take(mono, 23) + jnp.take(mono, 24)
    poly_17 = poly_3 * poly_4 - poly_14
    poly_18 = jnp.take(mono, 25) + jnp.take(mono, 26)
    poly_19 = poly_5 * poly_1
    poly_20 = poly_2 * poly_5
    poly_21 = poly_5 * poly_3
    poly_22 = poly_5 * poly_4
    poly_23 = poly_1 * poly_1 - poly_6 - poly_6
    poly_24 = poly_2 * poly_2
    poly_25 = poly_3 * poly_3 - poly_10 - poly_10
    poly_26 = poly_4 * poly_4 - poly_18 - poly_16 - \
        poly_15 - poly_18 - poly_16 - poly_15
    poly_27 = poly_5 * poly_5
    poly_28 = poly_2 * poly_6
    poly_29 = poly_6 * poly_3
    poly_30 = poly_2 * poly_8
    poly_31 = poly_10 * poly_1
    poly_32 = poly_2 * poly_10
    poly_33 = poly_6 * poly_4
    poly_34 = poly_2 * poly_11
    poly_35 = poly_2 * poly_12
    poly_36 = jnp.take(mono, 27) + jnp.take(mono, 28) + \
        jnp.take(mono, 29) + jnp.take(mono, 30)
    poly_37 = poly_1 * poly_14 - poly_36
    poly_38 = poly_2 * poly_14
    poly_39 = poly_1 * poly_15
    poly_40 = poly_2 * poly_15
    poly_41 = jnp.take(mono, 31) + jnp.take(mono, 32)
    poly_42 = poly_1 * poly_16 - poly_41
    poly_43 = poly_2 * poly_16
    poly_44 = poly_3 * poly_11 - poly_36
    poly_45 = poly_1 * poly_17 - poly_44
    poly_46 = poly_2 * poly_17
    poly_47 = poly_10 * poly_4
    poly_48 = poly_3 * poly_15
    poly_49 = poly_3 * poly_16
    poly_50 = poly_1 * poly_18
    poly_51 = poly_2 * poly_18
    poly_52 = jnp.take(mono, 33) + jnp.take(mono, 34)
    poly_53 = jnp.take(mono, 35) + jnp.take(mono, 36) + \
        jnp.take(mono, 37) + jnp.take(mono, 38)
    poly_54 = poly_3 * poly_18 - poly_52
    poly_55 = poly_5 * poly_6
    poly_56 = poly_2 * poly_19
    poly_57 = poly_5 * poly_8
    poly_58 = poly_2 * poly_21
    poly_59 = poly_5 * poly_10
    poly_60 = poly_5 * poly_11
    poly_61 = poly_5 * poly_12
    poly_62 = poly_2 * poly_22
    poly_63 = poly_5 * poly_14
    poly_64 = poly_5 * poly_15
    poly_65 = poly_5 * poly_16
    poly_66 = poly_5 * poly_17
    poly_67 = poly_5 * poly_18
    poly_68 = poly_6 * poly_1
    poly_69 = poly_2 * poly_23
    poly_70 = poly_2 * poly_7
    poly_71 = poly_3 * poly_23
    poly_72 = poly_2 * poly_9
    poly_73 = poly_1 * poly_25
    poly_74 = poly_2 * poly_25
    poly_75 = poly_10 * poly_3
    poly_76 = poly_1 * poly_11 - poly_33
    poly_77 = poly_1 * poly_12 - poly_33
    poly_78 = poly_2 * poly_13
    poly_79 = poly_3 * poly_14 - poly_47
    poly_80 = poly_3 * poly_17 - poly_47
    poly_81 = poly_4 * poly_11 - poly_50 - poly_41 - poly_39 - poly_41
    poly_82 = poly_1 * poly_26 - poly_81
    poly_83 = poly_2 * poly_26
    poly_84 = poly_4 * poly_14 - poly_52 - poly_49 - poly_48 - poly_52
    poly_85 = poly_4 * poly_15 - poly_53
    poly_86 = poly_4 * poly_16 - poly_53
    poly_87 = poly_3 * poly_26 - poly_84
    poly_88 = poly_4 * poly_18 - poly_53
    poly_89 = poly_5 * poly_23
    poly_90 = poly_2 * poly_20
    poly_91 = poly_5 * poly_25
    poly_92 = poly_5 * poly_26
    poly_93 = poly_5 * poly_19
    poly_94 = poly_2 * poly_27
    poly_95 = poly_5 * poly_21
    poly_96 = poly_5 * poly_22
    poly_97 = poly_1 * poly_23 - poly_68
    poly_98 = poly_2 * poly_24
    poly_99 = poly_3 * poly_25 - poly_75
    poly_100 = poly_4 * poly_26 - poly_88 - poly_86 - poly_85
    poly_101 = poly_5 * poly_27
    poly_102 = poly_2 * poly_29
    poly_103 = poly_6 * poly_10
    poly_104 = poly_2 * poly_31
    poly_105 = poly_2 * poly_33
    poly_106 = poly_6 * poly_14
    poly_107 = poly_2 * poly_36
    poly_108 = poly_2 * poly_37
    poly_109 = poly_6 * poly_15
    poly_110 = poly_2 * poly_39
    poly_111 = poly_6 * poly_16
    poly_112 = poly_2 * poly_41
    poly_113 = poly_2 * poly_42
    poly_114 = poly_6 * poly_17
    poly_115 = poly_2 * poly_44
    poly_116 = poly_2 * poly_45
    poly_117 = poly_10 * poly_11
    poly_118 = poly_10 * poly_12
    poly_119 = poly_2 * poly_47
    poly_120 = jnp.take(mono, 39) + jnp.take(mono, 40) + \
        jnp.take(mono, 41) + jnp.take(mono, 42)
    poly_121 = poly_1 * poly_48 - poly_120
    poly_122 = poly_2 * poly_48
    poly_123 = poly_10 * poly_15
    poly_124 = poly_3 * poly_41
    poly_125 = poly_3 * poly_42
    poly_126 = poly_2 * poly_49
    poly_127 = poly_10 * poly_16
    poly_128 = poly_6 * poly_18
    poly_129 = poly_2 * poly_50
    poly_130 = poly_1 * poly_52
    poly_131 = poly_2 * poly_52
    poly_132 = jnp.take(mono, 43) + jnp.take(mono, 44) + \
        jnp.take(mono, 45) + jnp.take(mono, 46)
    poly_133 = poly_1 * poly_53 - poly_132
    poly_134 = poly_2 * poly_53
    poly_135 = jnp.take(mono, 47) + jnp.take(mono, 48) + \
        jnp.take(mono, 49) + jnp.take(mono, 50)
    poly_136 = jnp.take(mono, 51)
    poly_137 = poly_1 * poly_54
    poly_138 = poly_2 * poly_54
    poly_139 = poly_10 * poly_18
    poly_140 = poly_3 * poly_53 - poly_135
    poly_141 = poly_2 * poly_55
    poly_142 = poly_5 * poly_29
    poly_143 = poly_2 * poly_57
    poly_144 = poly_5 * poly_31
    poly_145 = poly_2 * poly_59
    poly_146 = poly_5 * poly_33
    poly_147 = poly_2 * poly_60
    poly_148 = poly_2 * poly_61
    poly_149 = poly_5 * poly_36
    poly_150 = poly_5 * poly_37
    poly_151 = poly_2 * poly_63
    poly_152 = poly_5 * poly_39
    poly_153 = poly_2 * poly_64
    poly_154 = poly_5 * poly_41
    poly_155 = poly_5 * poly_42
    poly_156 = poly_2 * poly_65
    poly_157 = poly_5 * poly_44
    poly_158 = poly_5 * poly_45
    poly_159 = poly_2 * poly_66
    poly_160 = poly_5 * poly_47
    poly_161 = poly_5 * poly_48
    poly_162 = poly_5 * poly_49
    poly_163 = poly_5 * poly_50
    poly_164 = poly_2 * poly_67
    poly_165 = poly_5 * poly_52
    poly_166 = poly_5 * poly_53
    poly_167 = poly_5 * poly_54
    poly_168 = poly_2 * poly_68
    poly_169 = poly_2 * poly_28
    poly_170 = poly_6 * poly_8
    poly_171 = poly_2 * poly_71
    poly_172 = poly_2 * poly_30
    poly_173 = poly_10 * poly_23
    poly_174 = poly_2 * poly_32
    poly_175 = poly_6 * poly_25
    poly_176 = poly_2 * poly_73
    poly_177 = poly_10 * poly_8
    poly_178 = poly_2 * poly_75
    poly_179 = poly_6 * poly_11
    poly_180 = poly_6 * poly_12
    poly_181 = poly_2 * poly_76
    poly_182 = poly_2 * poly_77
    poly_183 = poly_2 * poly_34
    poly_184 = poly_2 * poly_35
    poly_185 = poly_1 * poly_36 - poly_106
    poly_186 = poly_1 * poly_37 - poly_106
    poly_187 = poly_2 * poly_38
    poly_188 = poly_3 * poly_36 - poly_117
    poly_189 = poly_1 * poly_79 - poly_188
    poly_190 = poly_2 * poly_79
    poly_191 = poly_15 * poly_23
    poly_192 = poly_2 * poly_40
    poly_193 = poly_1 * poly_41 - poly_111
    poly_194 = poly_1 * poly_42 - poly_111
    poly_195 = poly_2 * poly_43
    poly_196 = poly_1 * poly_44 - poly_114
    poly_197 = poly_1 * poly_45 - poly_114
    poly_198 = poly_2 * poly_46
    poly_199 = poly_10 * poly_14
    poly_200 = poly_3 * poly_44 - poly_117
    poly_201 = poly_1 * poly_80 - poly_200
    poly_202 = poly_2 * poly_80
    poly_203 = poly_10 * poly_17
    poly_204 = poly_15 * poly_25
    poly_205 = poly_16 * poly_25
    poly_206 = poly_18 * poly_23
    poly_207 = poly_2 * poly_51
    poly_208 = poly_3 * poly_52 - poly_139
    poly_209 = poly_3 * poly_54 - poly_139
    poly_210 = poly_6 * poly_26
    poly_211 = poly_2 * poly_81
    poly_212 = poly_2 * poly_82
    poly_213 = poly_4 * poly_36 - poly_130 - poly_124 - poly_121
    poly_214 = poly_1 * poly_84 - poly_213
    poly_215 = poly_2 * poly_84
    poly_216 = poly_11 * poly_15 - poly_132
    poly_217 = poly_1 * poly_85 - poly_216
    poly_218 = poly_2 * poly_85
    poly_219 = poly_14 * poly_15 - poly_135
    poly_220 = poly_4 * poly_41 - poly_132
    poly_221 = poly_1 * poly_86 - poly_220
    poly_222 = poly_2 * poly_86
    poly_223 = poly_14 * poly_16 - poly_135
    poly_224 = poly_15 * poly_16
    poly_225 = poly_3 * poly_81 - poly_213
    poly_226 = poly_1 * poly_87 - poly_225
    poly_227 = poly_2 * poly_87
    poly_228 = poly_10 * poly_26
    poly_229 = poly_3 * poly_85 - poly_219
    poly_230 = poly_3 * poly_86 - poly_223
    poly_231 = poly_11 * poly_18 - poly_132
    poly_232 = poly_1 * poly_88 - poly_231
    poly_233 = poly_2 * poly_88
    poly_234 = poly_4 * poly_52 - poly_135
    poly_235 = poly_15 * poly_18
    poly_236 = poly_16 * poly_18
    poly_237 = poly_3 * poly_88 - poly_234
    poly_238 = poly_5 * poly_68
    poly_239 = poly_2 * poly_89
    poly_240 = poly_2 * poly_56
    poly_241 = poly_5 * poly_71
    poly_242 = poly_2 * poly_58
    poly_243 = poly_5 * poly_73
    poly_244 = poly_2 * poly_91
    poly_245 = poly_5 * poly_75
    poly_246 = poly_5 * poly_76
    poly_247 = poly_5 * poly_77
    poly_248 = poly_2 * poly_62
    poly_249 = poly_5 * poly_79
    poly_250 = poly_5 * poly_80
    poly_251 = poly_5 * poly_81
    poly_252 = poly_5 * poly_82
    poly_253 = poly_2 * poly_92
    poly_254 = poly_5 * poly_84
    poly_255 = poly_5 * poly_85
    poly_256 = poly_5 * poly_86
    poly_257 = poly_5 * poly_87
    poly_258 = poly_5 * poly_88
    poly_259 = poly_5 * poly_55
    poly_260 = poly_2 * poly_93
    poly_261 = poly_5 * poly_57
    poly_262 = poly_2 * poly_95
    poly_263 = poly_5 * poly_59
    poly_264 = poly_5 * poly_60
    poly_265 = poly_5 * poly_61
    poly_266 = poly_2 * poly_96
    poly_267 = poly_5 * poly_63
    poly_268 = poly_5 * poly_64
    poly_269 = poly_5 * poly_65
    poly_270 = poly_5 * poly_66
    poly_271 = poly_5 * poly_67
    poly_272 = poly_6 * poly_6
    poly_273 = poly_6 * poly_23
    poly_274 = poly_2 * poly_97
    poly_275 = poly_2 * poly_69
    poly_276 = poly_2 * poly_70
    poly_277 = poly_3 * poly_97
    poly_278 = poly_2 * poly_72
    poly_279 = poly_23 * poly_25
    poly_280 = poly_2 * poly_74
    poly_281 = poly_10 * poly_10
    poly_282 = poly_1 * poly_99
    poly_283 = poly_2 * poly_99
    poly_284 = poly_10 * poly_25
    poly_285 = poly_1 * poly_76 - poly_179
    poly_286 = poly_1 * poly_77 - poly_180
    poly_287 = poly_2 * poly_78
    poly_288 = poly_3 * poly_79 - poly_199
    poly_289 = poly_3 * poly_80 - poly_203
    poly_290 = poly_1 * poly_81 - poly_210
    poly_291 = poly_1 * poly_82 - poly_210
    poly_292 = poly_2 * poly_83
    poly_293 = poly_3 * poly_84 - poly_228
    poly_294 = poly_15 * poly_15 - poly_136 - poly_136
    poly_295 = poly_16 * poly_16 - poly_136 - poly_136
    poly_296 = poly_3 * poly_87 - poly_228
    poly_297 = poly_18 * poly_18 - poly_136 - poly_136
    poly_298 = poly_4 * poly_81 - poly_231 - poly_220 - poly_216
    poly_299 = poly_1 * poly_100 - poly_298
    poly_300 = poly_2 * poly_100
    poly_301 = poly_4 * poly_84 - poly_234 - poly_223 - poly_219
    poly_302 = poly_15 * poly_26 - poly_236
    poly_303 = poly_16 * poly_26 - poly_235
    poly_304 = poly_3 * poly_100 - poly_301
    poly_305 = poly_18 * poly_26 - poly_224
    poly_306 = poly_5 * poly_97
    poly_307 = poly_2 * poly_90
    poly_308 = poly_5 * poly_99
    poly_309 = poly_5 * poly_100
    poly_310 = poly_5 * poly_89
    poly_311 = poly_2 * poly_94
    poly_312 = poly_5 * poly_91
    poly_313 = poly_5 * poly_92
    poly_314 = poly_5 * poly_93
    poly_315 = poly_2 * poly_101
    poly_316 = poly_5 * poly_95
    poly_317 = poly_5 * poly_96
    poly_318 = poly_1 * poly_97 - poly_273
    poly_319 = poly_2 * poly_98
    poly_320 = poly_3 * poly_99 - poly_284
    poly_321 = poly_4 * poly_100 - poly_305 - poly_303 - poly_302
    poly_322 = poly_5 * poly_101

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
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
                      poly_286,    poly_287,    poly_288,    poly_289,    poly_290,
                      poly_291,    poly_292,    poly_293,    poly_294,    poly_295,
                      poly_296,    poly_297,    poly_298,    poly_299,    poly_300,
                      poly_301,    poly_302,    poly_303,    poly_304,    poly_305,
                      poly_306,    poly_307,    poly_308,    poly_309,    poly_310,
                      poly_311,    poly_312,    poly_313,    poly_314,    poly_315,
                      poly_316,    poly_317,    poly_318,    poly_319,    poly_320,
                      poly_321,    poly_322,])

    return poly
