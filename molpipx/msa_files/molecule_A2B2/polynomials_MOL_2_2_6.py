import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2.monomials_MOL_2_2_6 import f_monomials as f_monos

# File created from ./MOL_2_2_6.POLY

N_POLYS = 291

# Total number of monomials = 291


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(291)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = jnp.take(mono, 2) + jnp.take(mono, 3) + \
        jnp.take(mono, 4) + jnp.take(mono, 5)
    poly_3 = jnp.take(mono, 6)
    poly_4 = poly_1 * poly_2
    poly_5 = jnp.take(mono, 7) + jnp.take(mono, 8)
    poly_6 = jnp.take(mono, 9) + jnp.take(mono, 10)
    poly_7 = jnp.take(mono, 11) + jnp.take(mono, 12)
    poly_8 = poly_1 * poly_3
    poly_9 = poly_3 * poly_2
    poly_10 = poly_1 * poly_1
    poly_11 = poly_2 * poly_2 - poly_7 - poly_6 - poly_5 - poly_7 - poly_6 - poly_5
    poly_12 = poly_3 * poly_3
    poly_13 = poly_1 * poly_5
    poly_14 = poly_1 * poly_6
    poly_15 = poly_1 * poly_7
    poly_16 = jnp.take(mono, 13) + jnp.take(mono, 14) + \
        jnp.take(mono, 15) + jnp.take(mono, 16)
    poly_17 = poly_1 * poly_9
    poly_18 = poly_3 * poly_5
    poly_19 = poly_3 * poly_6
    poly_20 = poly_3 * poly_7
    poly_21 = poly_1 * poly_4
    poly_22 = poly_1 * poly_11
    poly_23 = poly_2 * poly_5 - poly_16
    poly_24 = poly_2 * poly_6 - poly_16
    poly_25 = poly_2 * poly_7 - poly_16
    poly_26 = poly_1 * poly_8
    poly_27 = poly_3 * poly_11
    poly_28 = poly_1 * poly_12
    poly_29 = poly_3 * poly_9
    poly_30 = poly_1 * poly_10
    poly_31 = poly_2 * poly_11 - poly_25 - poly_24 - poly_23
    poly_32 = poly_3 * poly_12
    poly_33 = poly_1 * poly_16
    poly_34 = jnp.take(mono, 17)
    poly_35 = poly_1 * poly_18
    poly_36 = poly_1 * poly_19
    poly_37 = poly_1 * poly_20
    poly_38 = poly_3 * poly_16
    poly_39 = poly_1 * poly_13
    poly_40 = poly_1 * poly_14
    poly_41 = poly_1 * poly_15
    poly_42 = poly_1 * poly_23
    poly_43 = poly_1 * poly_24
    poly_44 = poly_5 * poly_6
    poly_45 = poly_1 * poly_25
    poly_46 = poly_5 * poly_7
    poly_47 = poly_6 * poly_7
    poly_48 = poly_1 * poly_17
    poly_49 = poly_1 * poly_27
    poly_50 = poly_3 * poly_23
    poly_51 = poly_3 * poly_24
    poly_52 = poly_3 * poly_25
    poly_53 = poly_1 * poly_29
    poly_54 = poly_3 * poly_18
    poly_55 = poly_3 * poly_19
    poly_56 = poly_3 * poly_20
    poly_57 = poly_1 * poly_21
    poly_58 = poly_1 * poly_22
    poly_59 = poly_5 * poly_5 - poly_34 - poly_34
    poly_60 = poly_6 * poly_6 - poly_34 - poly_34
    poly_61 = poly_7 * poly_7 - poly_34 - poly_34
    poly_62 = poly_1 * poly_31
    poly_63 = poly_5 * poly_11 - poly_47
    poly_64 = poly_6 * poly_11 - poly_46
    poly_65 = poly_7 * poly_11 - poly_44
    poly_66 = poly_1 * poly_26
    poly_67 = poly_3 * poly_31
    poly_68 = poly_1 * poly_28
    poly_69 = poly_3 * poly_27
    poly_70 = poly_1 * poly_32
    poly_71 = poly_3 * poly_29
    poly_72 = poly_1 * poly_30
    poly_73 = poly_2 * poly_31 - poly_65 - poly_64 - poly_63
    poly_74 = poly_3 * poly_32
    poly_75 = poly_1 * poly_34
    poly_76 = poly_1 * poly_38
    poly_77 = poly_3 * poly_34
    poly_78 = poly_1 * poly_33
    poly_79 = poly_1 * poly_44
    poly_80 = poly_1 * poly_46
    poly_81 = poly_1 * poly_47
    poly_82 = poly_34 * poly_2
    poly_83 = poly_1 * poly_35
    poly_84 = poly_1 * poly_36
    poly_85 = poly_1 * poly_37
    poly_86 = poly_1 * poly_50
    poly_87 = poly_1 * poly_51
    poly_88 = poly_3 * poly_44
    poly_89 = poly_1 * poly_52
    poly_90 = poly_3 * poly_46
    poly_91 = poly_3 * poly_47
    poly_92 = poly_1 * poly_54
    poly_93 = poly_1 * poly_55
    poly_94 = poly_1 * poly_56
    poly_95 = poly_3 * poly_38
    poly_96 = poly_1 * poly_39
    poly_97 = poly_1 * poly_40
    poly_98 = poly_1 * poly_41
    poly_99 = poly_1 * poly_42
    poly_100 = poly_1 * poly_59
    poly_101 = poly_1 * poly_43
    poly_102 = poly_1 * poly_60
    poly_103 = poly_1 * poly_45
    poly_104 = poly_5 * poly_16 - poly_82
    poly_105 = poly_6 * poly_16 - poly_82
    poly_106 = poly_1 * poly_61
    poly_107 = poly_7 * poly_16 - poly_82
    poly_108 = poly_1 * poly_63
    poly_109 = poly_1 * poly_64
    poly_110 = poly_5 * poly_24 - poly_105
    poly_111 = poly_1 * poly_65
    poly_112 = poly_5 * poly_25 - poly_107
    poly_113 = poly_6 * poly_25 - poly_107
    poly_114 = poly_1 * poly_48
    poly_115 = poly_1 * poly_49
    poly_116 = poly_3 * poly_59
    poly_117 = poly_3 * poly_60
    poly_118 = poly_3 * poly_61
    poly_119 = poly_1 * poly_67
    poly_120 = poly_3 * poly_63
    poly_121 = poly_3 * poly_64
    poly_122 = poly_3 * poly_65
    poly_123 = poly_1 * poly_53
    poly_124 = poly_1 * poly_69
    poly_125 = poly_3 * poly_50
    poly_126 = poly_3 * poly_51
    poly_127 = poly_3 * poly_52
    poly_128 = poly_1 * poly_71
    poly_129 = poly_3 * poly_54
    poly_130 = poly_3 * poly_55
    poly_131 = poly_3 * poly_56
    poly_132 = poly_1 * poly_57
    poly_133 = poly_1 * poly_58
    poly_134 = poly_1 * poly_62
    poly_135 = poly_2 * poly_59 - poly_104
    poly_136 = poly_2 * poly_60 - poly_105
    poly_137 = poly_2 * poly_61 - poly_107
    poly_138 = poly_1 * poly_73
    poly_139 = poly_5 * poly_31 - poly_113
    poly_140 = poly_6 * poly_31 - poly_112
    poly_141 = poly_7 * poly_31 - poly_110
    poly_142 = poly_1 * poly_66
    poly_143 = poly_3 * poly_73
    poly_144 = poly_1 * poly_68
    poly_145 = poly_3 * poly_67
    poly_146 = poly_1 * poly_70
    poly_147 = poly_3 * poly_69
    poly_148 = poly_1 * poly_74
    poly_149 = poly_3 * poly_71
    poly_150 = poly_1 * poly_72
    poly_151 = poly_2 * poly_73 - poly_141 - poly_140 - poly_139
    poly_152 = poly_3 * poly_74
    poly_153 = poly_1 * poly_77
    poly_154 = poly_1 * poly_75
    poly_155 = poly_1 * poly_82
    poly_156 = poly_1 * poly_76
    poly_157 = poly_1 * poly_88
    poly_158 = poly_1 * poly_90
    poly_159 = poly_1 * poly_91
    poly_160 = poly_3 * poly_82
    poly_161 = poly_1 * poly_95
    poly_162 = poly_3 * poly_77
    poly_163 = poly_1 * poly_78
    poly_164 = poly_1 * poly_79
    poly_165 = poly_1 * poly_80
    poly_166 = poly_1 * poly_104
    poly_167 = poly_1 * poly_81
    poly_168 = poly_34 * poly_5
    poly_169 = poly_1 * poly_105
    poly_170 = poly_34 * poly_6
    poly_171 = poly_1 * poly_107
    poly_172 = poly_34 * poly_7
    poly_173 = poly_1 * poly_110
    poly_174 = poly_1 * poly_112
    poly_175 = poly_1 * poly_113
    poly_176 = poly_34 * poly_11
    poly_177 = poly_1 * poly_83
    poly_178 = poly_1 * poly_84
    poly_179 = poly_1 * poly_85
    poly_180 = poly_1 * poly_86
    poly_181 = poly_1 * poly_116
    poly_182 = poly_1 * poly_87
    poly_183 = poly_1 * poly_117
    poly_184 = poly_1 * poly_89
    poly_185 = poly_3 * poly_104
    poly_186 = poly_3 * poly_105
    poly_187 = poly_1 * poly_118
    poly_188 = poly_3 * poly_107
    poly_189 = poly_1 * poly_120
    poly_190 = poly_1 * poly_121
    poly_191 = poly_3 * poly_110
    poly_192 = poly_1 * poly_122
    poly_193 = poly_3 * poly_112
    poly_194 = poly_3 * poly_113
    poly_195 = poly_1 * poly_92
    poly_196 = poly_1 * poly_93
    poly_197 = poly_1 * poly_94
    poly_198 = poly_1 * poly_125
    poly_199 = poly_1 * poly_126
    poly_200 = poly_3 * poly_88
    poly_201 = poly_1 * poly_127
    poly_202 = poly_3 * poly_90
    poly_203 = poly_3 * poly_91
    poly_204 = poly_1 * poly_129
    poly_205 = poly_1 * poly_130
    poly_206 = poly_1 * poly_131
    poly_207 = poly_3 * poly_95
    poly_208 = poly_1 * poly_96
    poly_209 = poly_1 * poly_97
    poly_210 = poly_1 * poly_98
    poly_211 = poly_1 * poly_99
    poly_212 = poly_1 * poly_100
    poly_213 = poly_1 * poly_101
    poly_214 = poly_1 * poly_102
    poly_215 = poly_1 * poly_103
    poly_216 = poly_1 * poly_106
    poly_217 = poly_5 * poly_47 - poly_176
    poly_218 = poly_1 * poly_108
    poly_219 = poly_1 * poly_135
    poly_220 = poly_1 * poly_109
    poly_221 = poly_6 * poly_59
    poly_222 = poly_1 * poly_136
    poly_223 = poly_5 * poly_60
    poly_224 = poly_1 * poly_111
    poly_225 = poly_7 * poly_59
    poly_226 = poly_7 * poly_60
    poly_227 = poly_1 * poly_137
    poly_228 = poly_5 * poly_61
    poly_229 = poly_6 * poly_61
    poly_230 = poly_1 * poly_139
    poly_231 = poly_1 * poly_140
    poly_232 = poly_5 * poly_64 - poly_226
    poly_233 = poly_1 * poly_141
    poly_234 = poly_5 * poly_65 - poly_229
    poly_235 = poly_6 * poly_65 - poly_228
    poly_236 = poly_1 * poly_114
    poly_237 = poly_1 * poly_115
    poly_238 = poly_1 * poly_119
    poly_239 = poly_3 * poly_135
    poly_240 = poly_3 * poly_136
    poly_241 = poly_3 * poly_137
    poly_242 = poly_1 * poly_143
    poly_243 = poly_3 * poly_139
    poly_244 = poly_3 * poly_140
    poly_245 = poly_3 * poly_141
    poly_246 = poly_1 * poly_123
    poly_247 = poly_1 * poly_124
    poly_248 = poly_3 * poly_116
    poly_249 = poly_3 * poly_117
    poly_250 = poly_3 * poly_118
    poly_251 = poly_1 * poly_145
    poly_252 = poly_3 * poly_120
    poly_253 = poly_3 * poly_121
    poly_254 = poly_3 * poly_122
    poly_255 = poly_1 * poly_128
    poly_256 = poly_1 * poly_147
    poly_257 = poly_3 * poly_125
    poly_258 = poly_3 * poly_126
    poly_259 = poly_3 * poly_127
    poly_260 = poly_1 * poly_149
    poly_261 = poly_3 * poly_129
    poly_262 = poly_3 * poly_130
    poly_263 = poly_3 * poly_131
    poly_264 = poly_1 * poly_132
    poly_265 = poly_1 * poly_133
    poly_266 = poly_1 * poly_134
    poly_267 = poly_5 * poly_59 - poly_168
    poly_268 = poly_6 * poly_60 - poly_170
    poly_269 = poly_7 * poly_61 - poly_172
    poly_270 = poly_1 * poly_138
    poly_271 = poly_5 * poly_63 - poly_176
    poly_272 = poly_6 * poly_64 - poly_176
    poly_273 = poly_7 * poly_65 - poly_176
    poly_274 = poly_1 * poly_151
    poly_275 = poly_5 * poly_73 - poly_235
    poly_276 = poly_6 * poly_73 - poly_234
    poly_277 = poly_7 * poly_73 - poly_232
    poly_278 = poly_1 * poly_142
    poly_279 = poly_3 * poly_151
    poly_280 = poly_1 * poly_144
    poly_281 = poly_3 * poly_143
    poly_282 = poly_1 * poly_146
    poly_283 = poly_3 * poly_145
    poly_284 = poly_1 * poly_148
    poly_285 = poly_3 * poly_147
    poly_286 = poly_1 * poly_152
    poly_287 = poly_3 * poly_149
    poly_288 = poly_1 * poly_150
    poly_289 = poly_2 * poly_151 - poly_277 - poly_276 - poly_275
    poly_290 = poly_3 * poly_152

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
                      ])

    return poly
