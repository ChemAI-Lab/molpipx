import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2C.monomials_MOL_2_2_1_5 import f_monomials as f_monos

# File created from ./MOL_2_2_1_5.POLY

N_POLYS = 904

# Total number of monomials = 904


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(904)

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
    poly_323 = poly_2 * poly_103
    poly_324 = poly_2 * poly_106
    poly_325 = poly_2 * poly_109
    poly_326 = poly_2 * poly_111
    poly_327 = poly_2 * poly_114
    poly_328 = poly_6 * poly_47
    poly_329 = poly_2 * poly_117
    poly_330 = poly_2 * poly_118
    poly_331 = poly_6 * poly_48
    poly_332 = poly_2 * poly_120
    poly_333 = poly_2 * poly_121
    poly_334 = poly_10 * poly_39
    poly_335 = poly_2 * poly_123
    poly_336 = poly_6 * poly_49
    poly_337 = poly_2 * poly_124
    poly_338 = poly_2 * poly_125
    poly_339 = poly_10 * poly_41
    poly_340 = poly_10 * poly_42
    poly_341 = poly_2 * poly_127
    poly_342 = poly_2 * poly_128
    poly_343 = poly_6 * poly_52
    poly_344 = poly_2 * poly_130
    poly_345 = poly_6 * poly_53
    poly_346 = poly_2 * poly_132
    poly_347 = poly_2 * poly_133
    poly_348 = jnp.take(mono, 52) + jnp.take(mono, 53) + \
        jnp.take(mono, 54) + jnp.take(mono, 55)
    poly_349 = poly_1 * poly_135 - poly_348
    poly_350 = poly_2 * poly_135
    poly_351 = poly_136 * poly_1
    poly_352 = poly_2 * poly_136
    poly_353 = poly_6 * poly_54
    poly_354 = poly_2 * poly_137
    poly_355 = poly_10 * poly_50
    poly_356 = poly_2 * poly_139
    poly_357 = poly_3 * poly_132 - poly_348
    poly_358 = poly_1 * poly_140 - poly_357
    poly_359 = poly_2 * poly_140
    poly_360 = poly_10 * poly_53
    poly_361 = poly_136 * poly_3
    poly_362 = poly_2 * poly_142
    poly_363 = poly_5 * poly_103
    poly_364 = poly_2 * poly_144
    poly_365 = poly_2 * poly_146
    poly_366 = poly_5 * poly_106
    poly_367 = poly_2 * poly_149
    poly_368 = poly_2 * poly_150
    poly_369 = poly_5 * poly_109
    poly_370 = poly_2 * poly_152
    poly_371 = poly_5 * poly_111
    poly_372 = poly_2 * poly_154
    poly_373 = poly_2 * poly_155
    poly_374 = poly_5 * poly_114
    poly_375 = poly_2 * poly_157
    poly_376 = poly_2 * poly_158
    poly_377 = poly_5 * poly_117
    poly_378 = poly_5 * poly_118
    poly_379 = poly_2 * poly_160
    poly_380 = poly_5 * poly_120
    poly_381 = poly_5 * poly_121
    poly_382 = poly_2 * poly_161
    poly_383 = poly_5 * poly_123
    poly_384 = poly_5 * poly_124
    poly_385 = poly_5 * poly_125
    poly_386 = poly_2 * poly_162
    poly_387 = poly_5 * poly_127
    poly_388 = poly_5 * poly_128
    poly_389 = poly_2 * poly_163
    poly_390 = poly_5 * poly_130
    poly_391 = poly_2 * poly_165
    poly_392 = poly_5 * poly_132
    poly_393 = poly_5 * poly_133
    poly_394 = poly_2 * poly_166
    poly_395 = poly_5 * poly_135
    poly_396 = poly_5 * poly_136
    poly_397 = poly_5 * poly_137
    poly_398 = poly_2 * poly_167
    poly_399 = poly_5 * poly_139
    poly_400 = poly_5 * poly_140
    poly_401 = poly_2 * poly_170
    poly_402 = poly_2 * poly_102
    poly_403 = poly_6 * poly_31
    poly_404 = poly_2 * poly_173
    poly_405 = poly_2 * poly_104
    poly_406 = poly_2 * poly_175
    poly_407 = poly_6 * poly_75
    poly_408 = poly_2 * poly_177
    poly_409 = poly_2 * poly_179
    poly_410 = poly_2 * poly_180
    poly_411 = poly_2 * poly_105
    poly_412 = poly_6 * poly_36
    poly_413 = poly_6 * poly_37
    poly_414 = poly_2 * poly_185
    poly_415 = poly_2 * poly_186
    poly_416 = poly_2 * poly_107
    poly_417 = poly_2 * poly_108
    poly_418 = poly_6 * poly_79
    poly_419 = poly_2 * poly_188
    poly_420 = poly_2 * poly_189
    poly_421 = poly_6 * poly_39
    poly_422 = poly_2 * poly_191
    poly_423 = poly_2 * poly_110
    poly_424 = poly_6 * poly_41
    poly_425 = poly_6 * poly_42
    poly_426 = poly_2 * poly_193
    poly_427 = poly_2 * poly_194
    poly_428 = poly_2 * poly_112
    poly_429 = poly_2 * poly_113
    poly_430 = poly_6 * poly_44
    poly_431 = poly_6 * poly_45
    poly_432 = poly_2 * poly_196
    poly_433 = poly_2 * poly_197
    poly_434 = poly_2 * poly_115
    poly_435 = poly_2 * poly_116
    poly_436 = poly_10 * poly_76
    poly_437 = poly_10 * poly_77
    poly_438 = poly_2 * poly_119
    poly_439 = poly_10 * poly_36
    poly_440 = poly_10 * poly_37
    poly_441 = poly_2 * poly_199
    poly_442 = poly_1 * poly_120 - poly_331
    poly_443 = poly_1 * poly_121 - poly_331
    poly_444 = poly_2 * poly_122
    poly_445 = poly_3 * poly_193
    poly_446 = poly_3 * poly_194
    poly_447 = poly_2 * poly_126
    poly_448 = poly_6 * poly_80
    poly_449 = poly_2 * poly_200
    poly_450 = poly_2 * poly_201
    poly_451 = poly_10 * poly_44
    poly_452 = poly_10 * poly_45
    poly_453 = poly_2 * poly_203
    poly_454 = poly_3 * poly_120 - poly_334
    poly_455 = poly_1 * poly_204 - poly_454
    poly_456 = poly_2 * poly_204
    poly_457 = poly_10 * poly_48
    poly_458 = poly_25 * poly_41
    poly_459 = poly_25 * poly_42
    poly_460 = poly_2 * poly_205
    poly_461 = poly_10 * poly_49
    poly_462 = poly_6 * poly_50
    poly_463 = poly_2 * poly_206
    poly_464 = poly_2 * poly_129
    poly_465 = poly_23 * poly_52
    poly_466 = poly_2 * poly_131
    poly_467 = poly_1 * poly_208
    poly_468 = poly_2 * poly_208
    poly_469 = poly_1 * poly_132 - poly_345
    poly_470 = poly_1 * poly_133 - poly_345
    poly_471 = poly_2 * poly_134
    poly_472 = poly_3 * poly_135 - poly_360
    poly_473 = poly_23 * poly_54
    poly_474 = poly_2 * poly_138
    poly_475 = poly_10 * poly_52
    poly_476 = poly_1 * poly_209
    poly_477 = poly_2 * poly_209
    poly_478 = poly_10 * poly_54
    poly_479 = poly_3 * poly_140 - poly_360
    poly_480 = poly_2 * poly_210
    poly_481 = poly_6 * poly_84
    poly_482 = poly_2 * poly_213
    poly_483 = poly_2 * poly_214
    poly_484 = poly_6 * poly_85
    poly_485 = poly_2 * poly_216
    poly_486 = poly_2 * poly_217
    poly_487 = poly_15 * poly_36 - poly_348
    poly_488 = poly_1 * poly_219 - poly_487
    poly_489 = poly_2 * poly_219
    poly_490 = poly_6 * poly_86
    poly_491 = poly_2 * poly_220
    poly_492 = poly_2 * poly_221
    poly_493 = poly_14 * poly_41 - poly_348
    poly_494 = poly_1 * poly_223 - poly_493
    poly_495 = poly_2 * poly_223
    poly_496 = poly_15 * poly_41
    poly_497 = poly_15 * poly_42
    poly_498 = poly_2 * poly_224
    poly_499 = jnp.take(mono, 56) + jnp.take(mono, 57) + \
        jnp.take(mono, 58) + jnp.take(mono, 59)
    poly_500 = poly_6 * poly_87
    poly_501 = poly_2 * poly_225
    poly_502 = poly_2 * poly_226
    poly_503 = poly_10 * poly_81
    poly_504 = poly_10 * poly_82
    poly_505 = poly_2 * poly_228
    poly_506 = poly_3 * poly_216 - poly_487
    poly_507 = poly_1 * poly_229 - poly_506
    poly_508 = poly_2 * poly_229
    poly_509 = poly_10 * poly_85
    poly_510 = poly_3 * poly_220 - poly_493
    poly_511 = poly_1 * poly_230 - poly_510
    poly_512 = poly_2 * poly_230
    poly_513 = poly_10 * poly_86
    poly_514 = poly_3 * poly_224 - poly_499
    poly_515 = poly_6 * poly_88
    poly_516 = poly_2 * poly_231
    poly_517 = poly_2 * poly_232
    poly_518 = poly_11 * poly_52 - poly_348
    poly_519 = poly_1 * poly_234 - poly_518
    poly_520 = poly_2 * poly_234
    poly_521 = jnp.take(mono, 60) + jnp.take(mono, 61) + \
        jnp.take(mono, 62) + jnp.take(mono, 63)
    poly_522 = poly_1 * poly_235 - poly_521
    poly_523 = poly_2 * poly_235
    poly_524 = poly_15 * poly_52
    poly_525 = poly_18 * poly_41
    poly_526 = poly_18 * poly_42
    poly_527 = poly_2 * poly_236
    poly_528 = poly_16 * poly_52
    poly_529 = poly_136 * poly_4
    poly_530 = poly_3 * poly_231 - poly_518
    poly_531 = poly_1 * poly_237 - poly_530
    poly_532 = poly_2 * poly_237
    poly_533 = poly_10 * poly_88
    poly_534 = poly_15 * poly_54
    poly_535 = poly_16 * poly_54
    poly_536 = poly_2 * poly_238
    poly_537 = poly_2 * poly_141
    poly_538 = poly_5 * poly_170
    poly_539 = poly_2 * poly_241
    poly_540 = poly_2 * poly_143
    poly_541 = poly_5 * poly_173
    poly_542 = poly_2 * poly_145
    poly_543 = poly_5 * poly_175
    poly_544 = poly_2 * poly_243
    poly_545 = poly_5 * poly_177
    poly_546 = poly_2 * poly_245
    poly_547 = poly_5 * poly_179
    poly_548 = poly_5 * poly_180
    poly_549 = poly_2 * poly_246
    poly_550 = poly_2 * poly_247
    poly_551 = poly_2 * poly_147
    poly_552 = poly_2 * poly_148
    poly_553 = poly_5 * poly_185
    poly_554 = poly_5 * poly_186
    poly_555 = poly_2 * poly_151
    poly_556 = poly_5 * poly_188
    poly_557 = poly_5 * poly_189
    poly_558 = poly_2 * poly_249
    poly_559 = poly_5 * poly_191
    poly_560 = poly_2 * poly_153
    poly_561 = poly_5 * poly_193
    poly_562 = poly_5 * poly_194
    poly_563 = poly_2 * poly_156
    poly_564 = poly_5 * poly_196
    poly_565 = poly_5 * poly_197
    poly_566 = poly_2 * poly_159
    poly_567 = poly_5 * poly_199
    poly_568 = poly_5 * poly_200
    poly_569 = poly_5 * poly_201
    poly_570 = poly_2 * poly_250
    poly_571 = poly_5 * poly_203
    poly_572 = poly_5 * poly_204
    poly_573 = poly_5 * poly_205
    poly_574 = poly_5 * poly_206
    poly_575 = poly_2 * poly_164
    poly_576 = poly_5 * poly_208
    poly_577 = poly_5 * poly_209
    poly_578 = poly_5 * poly_210
    poly_579 = poly_2 * poly_251
    poly_580 = poly_2 * poly_252
    poly_581 = poly_5 * poly_213
    poly_582 = poly_5 * poly_214
    poly_583 = poly_2 * poly_254
    poly_584 = poly_5 * poly_216
    poly_585 = poly_5 * poly_217
    poly_586 = poly_2 * poly_255
    poly_587 = poly_5 * poly_219
    poly_588 = poly_5 * poly_220
    poly_589 = poly_5 * poly_221
    poly_590 = poly_2 * poly_256
    poly_591 = poly_5 * poly_223
    poly_592 = poly_5 * poly_224
    poly_593 = poly_5 * poly_225
    poly_594 = poly_5 * poly_226
    poly_595 = poly_2 * poly_257
    poly_596 = poly_5 * poly_228
    poly_597 = poly_5 * poly_229
    poly_598 = poly_5 * poly_230
    poly_599 = poly_5 * poly_231
    poly_600 = poly_5 * poly_232
    poly_601 = poly_2 * poly_258
    poly_602 = poly_5 * poly_234
    poly_603 = poly_5 * poly_235
    poly_604 = poly_5 * poly_236
    poly_605 = poly_5 * poly_237
    poly_606 = poly_2 * poly_259
    poly_607 = poly_5 * poly_142
    poly_608 = poly_2 * poly_261
    poly_609 = poly_5 * poly_144
    poly_610 = poly_2 * poly_263
    poly_611 = poly_5 * poly_146
    poly_612 = poly_2 * poly_264
    poly_613 = poly_2 * poly_265
    poly_614 = poly_5 * poly_149
    poly_615 = poly_5 * poly_150
    poly_616 = poly_2 * poly_267
    poly_617 = poly_5 * poly_152
    poly_618 = poly_2 * poly_268
    poly_619 = poly_5 * poly_154
    poly_620 = poly_5 * poly_155
    poly_621 = poly_2 * poly_269
    poly_622 = poly_5 * poly_157
    poly_623 = poly_5 * poly_158
    poly_624 = poly_2 * poly_270
    poly_625 = poly_5 * poly_160
    poly_626 = poly_5 * poly_161
    poly_627 = poly_5 * poly_162
    poly_628 = poly_5 * poly_163
    poly_629 = poly_2 * poly_271
    poly_630 = poly_5 * poly_165
    poly_631 = poly_5 * poly_166
    poly_632 = poly_5 * poly_167
    poly_633 = poly_2 * poly_272
    poly_634 = poly_2 * poly_273
    poly_635 = poly_2 * poly_168
    poly_636 = poly_2 * poly_169
    poly_637 = poly_6 * poly_29
    poly_638 = poly_6 * poly_71
    poly_639 = poly_2 * poly_277
    poly_640 = poly_2 * poly_171
    poly_641 = poly_2 * poly_172
    poly_642 = poly_10 * poly_97
    poly_643 = poly_2 * poly_174
    poly_644 = poly_6 * poly_73
    poly_645 = poly_2 * poly_279
    poly_646 = poly_2 * poly_176
    poly_647 = poly_10 * poly_71
    poly_648 = poly_2 * poly_178
    poly_649 = poly_10 * poly_31
    poly_650 = poly_2 * poly_281
    poly_651 = poly_6 * poly_99
    poly_652 = poly_2 * poly_282
    poly_653 = poly_10 * poly_73
    poly_654 = poly_2 * poly_284
    poly_655 = poly_6 * poly_76
    poly_656 = poly_6 * poly_33
    poly_657 = poly_6 * poly_77
    poly_658 = poly_2 * poly_285
    poly_659 = poly_2 * poly_286
    poly_660 = poly_2 * poly_181
    poly_661 = poly_2 * poly_182
    poly_662 = poly_2 * poly_183
    poly_663 = poly_2 * poly_184
    poly_664 = poly_1 * poly_185 - poly_412
    poly_665 = poly_1 * poly_186 - poly_413
    poly_666 = poly_2 * poly_187
    poly_667 = poly_1 * poly_188 - poly_418
    poly_668 = poly_1 * poly_189 - poly_418
    poly_669 = poly_2 * poly_190
    poly_670 = poly_3 * poly_188 - poly_439
    poly_671 = poly_1 * poly_288 - poly_670
    poly_672 = poly_2 * poly_288
    poly_673 = poly_15 * poly_97
    poly_674 = poly_2 * poly_192
    poly_675 = poly_1 * poly_193 - poly_424
    poly_676 = poly_1 * poly_194 - poly_425
    poly_677 = poly_2 * poly_195
    poly_678 = poly_1 * poly_196 - poly_430
    poly_679 = poly_1 * poly_197 - poly_431
    poly_680 = poly_2 * poly_198
    poly_681 = poly_10 * poly_79
    poly_682 = poly_1 * poly_200 - poly_448
    poly_683 = poly_1 * poly_201 - poly_448
    poly_684 = poly_2 * poly_202
    poly_685 = poly_10 * poly_47
    poly_686 = poly_3 * poly_200 - poly_451
    poly_687 = poly_1 * poly_289 - poly_686
    poly_688 = poly_2 * poly_289
    poly_689 = poly_10 * poly_80
    poly_690 = poly_15 * poly_99
    poly_691 = poly_16 * poly_99
    poly_692 = poly_18 * poly_97
    poly_693 = poly_2 * poly_207
    poly_694 = poly_3 * poly_208 - poly_475
    poly_695 = poly_3 * poly_209 - poly_478
    poly_696 = poly_6 * poly_81
    poly_697 = poly_6 * poly_82
    poly_698 = poly_2 * poly_290
    poly_699 = poly_2 * poly_291
    poly_700 = poly_2 * poly_211
    poly_701 = poly_2 * poly_212
    poly_702 = poly_1 * poly_213 - poly_481
    poly_703 = poly_1 * poly_214 - poly_481
    poly_704 = poly_2 * poly_215
    poly_705 = poly_3 * poly_213 - poly_503
    poly_706 = poly_1 * poly_293 - poly_705
    poly_707 = poly_2 * poly_293
    poly_708 = poly_1 * poly_216 - poly_484
    poly_709 = poly_1 * poly_217 - poly_484
    poly_710 = poly_2 * poly_218
    poly_711 = poly_3 * poly_219 - poly_509
    poly_712 = poly_1 * poly_294
    poly_713 = poly_2 * poly_294
    poly_714 = poly_1 * poly_220 - poly_490
    poly_715 = poly_1 * poly_221 - poly_490
    poly_716 = poly_2 * poly_222
    poly_717 = poly_3 * poly_223 - poly_513
    poly_718 = poly_16 * poly_41 - poly_351
    poly_719 = poly_1 * poly_295 - poly_718
    poly_720 = poly_2 * poly_295
    poly_721 = poly_1 * poly_225 - poly_500
    poly_722 = poly_1 * poly_226 - poly_500
    poly_723 = poly_2 * poly_227
    poly_724 = poly_10 * poly_84
    poly_725 = poly_3 * poly_294
    poly_726 = poly_3 * poly_295
    poly_727 = poly_3 * poly_225 - poly_503
    poly_728 = poly_1 * poly_296 - poly_727
    poly_729 = poly_2 * poly_296
    poly_730 = poly_10 * poly_87
    poly_731 = poly_3 * poly_229 - poly_509
    poly_732 = poly_3 * poly_230 - poly_513
    poly_733 = poly_1 * poly_231 - poly_515
    poly_734 = poly_1 * poly_232 - poly_515
    poly_735 = poly_2 * poly_233
    poly_736 = poly_3 * poly_234 - poly_533
    poly_737 = poly_15 * poly_53 - poly_529
    poly_738 = poly_16 * poly_53 - poly_529
    poly_739 = poly_3 * poly_237 - poly_533
    poly_740 = poly_1 * poly_297
    poly_741 = poly_2 * poly_297
    poly_742 = poly_18 * poly_52 - poly_361
    poly_743 = poly_18 * poly_53 - poly_529
    poly_744 = poly_3 * poly_297 - poly_742
    poly_745 = poly_6 * poly_100
    poly_746 = poly_2 * poly_298
    poly_747 = poly_2 * poly_299
    poly_748 = poly_4 * poly_213 - poly_518 - poly_493 - poly_487
    poly_749 = poly_1 * poly_301 - poly_748
    poly_750 = poly_2 * poly_301
    poly_751 = poly_15 * poly_81 - poly_525
    poly_752 = poly_1 * poly_302 - poly_751
    poly_753 = poly_2 * poly_302
    poly_754 = poly_15 * poly_84 - poly_528
    poly_755 = poly_16 * poly_81 - poly_521
    poly_756 = poly_1 * poly_303 - poly_755
    poly_757 = poly_2 * poly_303
    poly_758 = poly_16 * poly_84 - poly_524
    poly_759 = poly_15 * poly_86 - poly_738
    poly_760 = poly_3 * poly_298 - poly_748
    poly_761 = poly_1 * poly_304 - poly_760
    poly_762 = poly_2 * poly_304
    poly_763 = poly_10 * poly_100
    poly_764 = poly_3 * poly_302 - poly_754
    poly_765 = poly_3 * poly_303 - poly_758
    poly_766 = poly_18 * poly_81 - poly_496
    poly_767 = poly_1 * poly_305 - poly_766
    poly_768 = poly_2 * poly_305
    poly_769 = poly_18 * poly_84 - poly_499
    poly_770 = poly_15 * poly_88 - poly_743
    poly_771 = poly_16 * poly_88 - poly_743
    poly_772 = poly_3 * poly_305 - poly_769
    poly_773 = poly_5 * poly_272
    poly_774 = poly_5 * poly_273
    poly_775 = poly_2 * poly_306
    poly_776 = poly_2 * poly_239
    poly_777 = poly_2 * poly_240
    poly_778 = poly_5 * poly_277
    poly_779 = poly_2 * poly_242
    poly_780 = poly_5 * poly_279
    poly_781 = poly_2 * poly_244
    poly_782 = poly_5 * poly_281
    poly_783 = poly_5 * poly_282
    poly_784 = poly_2 * poly_308
    poly_785 = poly_5 * poly_284
    poly_786 = poly_5 * poly_285
    poly_787 = poly_5 * poly_286
    poly_788 = poly_2 * poly_248
    poly_789 = poly_5 * poly_288
    poly_790 = poly_5 * poly_289
    poly_791 = poly_5 * poly_290
    poly_792 = poly_5 * poly_291
    poly_793 = poly_2 * poly_253
    poly_794 = poly_5 * poly_293
    poly_795 = poly_5 * poly_294
    poly_796 = poly_5 * poly_295
    poly_797 = poly_5 * poly_296
    poly_798 = poly_5 * poly_297
    poly_799 = poly_5 * poly_298
    poly_800 = poly_5 * poly_299
    poly_801 = poly_2 * poly_309
    poly_802 = poly_5 * poly_301
    poly_803 = poly_5 * poly_302
    poly_804 = poly_5 * poly_303
    poly_805 = poly_5 * poly_304
    poly_806 = poly_5 * poly_305
    poly_807 = poly_5 * poly_238
    poly_808 = poly_2 * poly_310
    poly_809 = poly_2 * poly_260
    poly_810 = poly_5 * poly_241
    poly_811 = poly_2 * poly_262
    poly_812 = poly_5 * poly_243
    poly_813 = poly_2 * poly_312
    poly_814 = poly_5 * poly_245
    poly_815 = poly_5 * poly_246
    poly_816 = poly_5 * poly_247
    poly_817 = poly_2 * poly_266
    poly_818 = poly_5 * poly_249
    poly_819 = poly_5 * poly_250
    poly_820 = poly_5 * poly_251
    poly_821 = poly_5 * poly_252
    poly_822 = poly_2 * poly_313
    poly_823 = poly_5 * poly_254
    poly_824 = poly_5 * poly_255
    poly_825 = poly_5 * poly_256
    poly_826 = poly_5 * poly_257
    poly_827 = poly_5 * poly_258
    poly_828 = poly_5 * poly_259
    poly_829 = poly_2 * poly_314
    poly_830 = poly_5 * poly_261
    poly_831 = poly_2 * poly_316
    poly_832 = poly_5 * poly_263
    poly_833 = poly_5 * poly_264
    poly_834 = poly_5 * poly_265
    poly_835 = poly_2 * poly_317
    poly_836 = poly_5 * poly_267
    poly_837 = poly_5 * poly_268
    poly_838 = poly_5 * poly_269
    poly_839 = poly_5 * poly_270
    poly_840 = poly_5 * poly_271
    poly_841 = poly_6 * poly_68
    poly_842 = poly_6 * poly_97
    poly_843 = poly_2 * poly_318
    poly_844 = poly_2 * poly_274
    poly_845 = poly_2 * poly_275
    poly_846 = poly_2 * poly_276
    poly_847 = poly_3 * poly_318
    poly_848 = poly_2 * poly_278
    poly_849 = poly_25 * poly_97
    poly_850 = poly_2 * poly_280
    poly_851 = poly_23 * poly_99
    poly_852 = poly_2 * poly_283
    poly_853 = poly_10 * poly_75
    poly_854 = poly_1 * poly_320
    poly_855 = poly_2 * poly_320
    poly_856 = poly_10 * poly_99
    poly_857 = poly_1 * poly_285 - poly_655
    poly_858 = poly_1 * poly_286 - poly_657
    poly_859 = poly_2 * poly_287
    poly_860 = poly_3 * poly_288 - poly_681
    poly_861 = poly_3 * poly_289 - poly_689
    poly_862 = poly_1 * poly_290 - poly_696
    poly_863 = poly_1 * poly_291 - poly_697
    poly_864 = poly_2 * poly_292
    poly_865 = poly_3 * poly_293 - poly_724
    poly_866 = poly_3 * poly_296 - poly_730
    poly_867 = poly_1 * poly_298 - poly_745
    poly_868 = poly_1 * poly_299 - poly_745
    poly_869 = poly_2 * poly_300
    poly_870 = poly_3 * poly_301 - poly_763
    poly_871 = poly_4 * poly_294 - poly_737
    poly_872 = poly_4 * poly_295 - poly_738
    poly_873 = poly_3 * poly_304 - poly_763
    poly_874 = poly_4 * poly_297 - poly_743
    poly_875 = poly_4 * poly_298 - poly_766 - poly_755 - poly_751
    poly_876 = poly_1 * poly_321 - poly_875
    poly_877 = poly_2 * poly_321
    poly_878 = poly_4 * poly_301 - poly_769 - poly_758 - poly_754
    poly_879 = poly_15 * poly_100 - poly_771
    poly_880 = poly_16 * poly_100 - poly_770
    poly_881 = poly_3 * poly_321 - poly_878
    poly_882 = poly_18 * poly_100 - poly_759
    poly_883 = poly_5 * poly_318
    poly_884 = poly_2 * poly_307
    poly_885 = poly_5 * poly_320
    poly_886 = poly_5 * poly_321
    poly_887 = poly_5 * poly_306
    poly_888 = poly_2 * poly_311
    poly_889 = poly_5 * poly_308
    poly_890 = poly_5 * poly_309
    poly_891 = poly_5 * poly_310
    poly_892 = poly_2 * poly_315
    poly_893 = poly_5 * poly_312
    poly_894 = poly_5 * poly_313
    poly_895 = poly_5 * poly_314
    poly_896 = poly_2 * poly_322
    poly_897 = poly_5 * poly_316
    poly_898 = poly_5 * poly_317
    poly_899 = poly_1 * poly_318 - poly_842
    poly_900 = poly_2 * poly_319
    poly_901 = poly_3 * poly_320 - poly_856
    poly_902 = poly_4 * poly_321 - poly_882 - poly_880 - poly_879
    poly_903 = poly_5 * poly_322

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
                      poly_321,    poly_322,    poly_323,    poly_324,    poly_325,
                      poly_326,    poly_327,    poly_328,    poly_329,    poly_330,
                      poly_331,    poly_332,    poly_333,    poly_334,    poly_335,
                      poly_336,    poly_337,    poly_338,    poly_339,    poly_340,
                      poly_341,    poly_342,    poly_343,    poly_344,    poly_345,
                      poly_346,    poly_347,    poly_348,    poly_349,    poly_350,
                      poly_351,    poly_352,    poly_353,    poly_354,    poly_355,
                      poly_356,    poly_357,    poly_358,    poly_359,    poly_360,
                      poly_361,    poly_362,    poly_363,    poly_364,    poly_365,
                      poly_366,    poly_367,    poly_368,    poly_369,    poly_370,
                      poly_371,    poly_372,    poly_373,    poly_374,    poly_375,
                      poly_376,    poly_377,    poly_378,    poly_379,    poly_380,
                      poly_381,    poly_382,    poly_383,    poly_384,    poly_385,
                      poly_386,    poly_387,    poly_388,    poly_389,    poly_390,
                      poly_391,    poly_392,    poly_393,    poly_394,    poly_395,
                      poly_396,    poly_397,    poly_398,    poly_399,    poly_400,
                      poly_401,    poly_402,    poly_403,    poly_404,    poly_405,
                      poly_406,    poly_407,    poly_408,    poly_409,    poly_410,
                      poly_411,    poly_412,    poly_413,    poly_414,    poly_415,
                      poly_416,    poly_417,    poly_418,    poly_419,    poly_420,
                      poly_421,    poly_422,    poly_423,    poly_424,    poly_425,
                      poly_426,    poly_427,    poly_428,    poly_429,    poly_430,
                      poly_431,    poly_432,    poly_433,    poly_434,    poly_435,
                      poly_436,    poly_437,    poly_438,    poly_439,    poly_440,
                      poly_441,    poly_442,    poly_443,    poly_444,    poly_445,
                      poly_446,    poly_447,    poly_448,    poly_449,    poly_450,
                      poly_451,    poly_452,    poly_453,    poly_454,    poly_455,
                      poly_456,    poly_457,    poly_458,    poly_459,    poly_460,
                      poly_461,    poly_462,    poly_463,    poly_464,    poly_465,
                      poly_466,    poly_467,    poly_468,    poly_469,    poly_470,
                      poly_471,    poly_472,    poly_473,    poly_474,    poly_475,
                      poly_476,    poly_477,    poly_478,    poly_479,    poly_480,
                      poly_481,    poly_482,    poly_483,    poly_484,    poly_485,
                      poly_486,    poly_487,    poly_488,    poly_489,    poly_490,
                      poly_491,    poly_492,    poly_493,    poly_494,    poly_495,
                      poly_496,    poly_497,    poly_498,    poly_499,    poly_500,
                      poly_501,    poly_502,    poly_503,    poly_504,    poly_505,
                      poly_506,    poly_507,    poly_508,    poly_509,    poly_510,
                      poly_511,    poly_512,    poly_513,    poly_514,    poly_515,
                      poly_516,    poly_517,    poly_518,    poly_519,    poly_520,
                      poly_521,    poly_522,    poly_523,    poly_524,    poly_525,
                      poly_526,    poly_527,    poly_528,    poly_529,    poly_530,
                      poly_531,    poly_532,    poly_533,    poly_534,    poly_535,
                      poly_536,    poly_537,    poly_538,    poly_539,    poly_540,
                      poly_541,    poly_542,    poly_543,    poly_544,    poly_545,
                      poly_546,    poly_547,    poly_548,    poly_549,    poly_550,
                      poly_551,    poly_552,    poly_553,    poly_554,    poly_555,
                      poly_556,    poly_557,    poly_558,    poly_559,    poly_560,
                      poly_561,    poly_562,    poly_563,    poly_564,    poly_565,
                      poly_566,    poly_567,    poly_568,    poly_569,    poly_570,
                      poly_571,    poly_572,    poly_573,    poly_574,    poly_575,
                      poly_576,    poly_577,    poly_578,    poly_579,    poly_580,
                      poly_581,    poly_582,    poly_583,    poly_584,    poly_585,
                      poly_586,    poly_587,    poly_588,    poly_589,    poly_590,
                      poly_591,    poly_592,    poly_593,    poly_594,    poly_595,
                      poly_596,    poly_597,    poly_598,    poly_599,    poly_600,
                      poly_601,    poly_602,    poly_603,    poly_604,    poly_605,
                      poly_606,    poly_607,    poly_608,    poly_609,    poly_610,
                      poly_611,    poly_612,    poly_613,    poly_614,    poly_615,
                      poly_616,    poly_617,    poly_618,    poly_619,    poly_620,
                      poly_621,    poly_622,    poly_623,    poly_624,    poly_625,
                      poly_626,    poly_627,    poly_628,    poly_629,    poly_630,
                      poly_631,    poly_632,    poly_633,    poly_634,    poly_635,
                      poly_636,    poly_637,    poly_638,    poly_639,    poly_640,
                      poly_641,    poly_642,    poly_643,    poly_644,    poly_645,
                      poly_646,    poly_647,    poly_648,    poly_649,    poly_650,
                      poly_651,    poly_652,    poly_653,    poly_654,    poly_655,
                      poly_656,    poly_657,    poly_658,    poly_659,    poly_660,
                      poly_661,    poly_662,    poly_663,    poly_664,    poly_665,
                      poly_666,    poly_667,    poly_668,    poly_669,    poly_670,
                      poly_671,    poly_672,    poly_673,    poly_674,    poly_675,
                      poly_676,    poly_677,    poly_678,    poly_679,    poly_680,
                      poly_681,    poly_682,    poly_683,    poly_684,    poly_685,
                      poly_686,    poly_687,    poly_688,    poly_689,    poly_690,
                      poly_691,    poly_692,    poly_693,    poly_694,    poly_695,
                      poly_696,    poly_697,    poly_698,    poly_699,    poly_700,
                      poly_701,    poly_702,    poly_703,    poly_704,    poly_705,
                      poly_706,    poly_707,    poly_708,    poly_709,    poly_710,
                      poly_711,    poly_712,    poly_713,    poly_714,    poly_715,
                      poly_716,    poly_717,    poly_718,    poly_719,    poly_720,
                      poly_721,    poly_722,    poly_723,    poly_724,    poly_725,
                      poly_726,    poly_727,    poly_728,    poly_729,    poly_730,
                      poly_731,    poly_732,    poly_733,    poly_734,    poly_735,
                      poly_736,    poly_737,    poly_738,    poly_739,    poly_740,
                      poly_741,    poly_742,    poly_743,    poly_744,    poly_745,
                      poly_746,    poly_747,    poly_748,    poly_749,    poly_750,
                      poly_751,    poly_752,    poly_753,    poly_754,    poly_755,
                      poly_756,    poly_757,    poly_758,    poly_759,    poly_760,
                      poly_761,    poly_762,    poly_763,    poly_764,    poly_765,
                      poly_766,    poly_767,    poly_768,    poly_769,    poly_770,
                      poly_771,    poly_772,    poly_773,    poly_774,    poly_775,
                      poly_776,    poly_777,    poly_778,    poly_779,    poly_780,
                      poly_781,    poly_782,    poly_783,    poly_784,    poly_785,
                      poly_786,    poly_787,    poly_788,    poly_789,    poly_790,
                      poly_791,    poly_792,    poly_793,    poly_794,    poly_795,
                      poly_796,    poly_797,    poly_798,    poly_799,    poly_800,
                      poly_801,    poly_802,    poly_803,    poly_804,    poly_805,
                      poly_806,    poly_807,    poly_808,    poly_809,    poly_810,
                      poly_811,    poly_812,    poly_813,    poly_814,    poly_815,
                      poly_816,    poly_817,    poly_818,    poly_819,    poly_820,
                      poly_821,    poly_822,    poly_823,    poly_824,    poly_825,
                      poly_826,    poly_827,    poly_828,    poly_829,    poly_830,
                      poly_831,    poly_832,    poly_833,    poly_834,    poly_835,
                      poly_836,    poly_837,    poly_838,    poly_839,    poly_840,
                      poly_841,    poly_842,    poly_843,    poly_844,    poly_845,
                      poly_846,    poly_847,    poly_848,    poly_849,    poly_850,
                      poly_851,    poly_852,    poly_853,    poly_854,    poly_855,
                      poly_856,    poly_857,    poly_858,    poly_859,    poly_860,
                      poly_861,    poly_862,    poly_863,    poly_864,    poly_865,
                      poly_866,    poly_867,    poly_868,    poly_869,    poly_870,
                      poly_871,    poly_872,    poly_873,    poly_874,    poly_875,
                      poly_876,    poly_877,    poly_878,    poly_879,    poly_880,
                      poly_881,    poly_882,    poly_883,    poly_884,    poly_885,
                      poly_886,    poly_887,    poly_888,    poly_889,    poly_890,
                      poly_891,    poly_892,    poly_893,    poly_894,    poly_895,
                      poly_896,    poly_897,    poly_898,    poly_899,    poly_900,
                      poly_901,    poly_902,    poly_903,])

    return poly
