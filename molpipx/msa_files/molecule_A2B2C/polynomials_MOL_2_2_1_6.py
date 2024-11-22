import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2C.monomials_MOL_2_2_1_6 import f_monomials as f_monos

# File created from ./MOL_2_2_1_6.POLY

N_POLYS = 2304

# Total number of monomials = 2304


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(2304)

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
    poly_904 = poly_2 * poly_328
    poly_905 = poly_2 * poly_331
    poly_906 = poly_6 * poly_123
    poly_907 = poly_2 * poly_334
    poly_908 = poly_2 * poly_336
    poly_909 = poly_6 * poly_127
    poly_910 = poly_2 * poly_339
    poly_911 = poly_2 * poly_340
    poly_912 = poly_2 * poly_343
    poly_913 = poly_2 * poly_345
    poly_914 = poly_6 * poly_135
    poly_915 = poly_2 * poly_348
    poly_916 = poly_2 * poly_349
    poly_917 = poly_6 * poly_136
    poly_918 = poly_2 * poly_351
    poly_919 = poly_2 * poly_353
    poly_920 = poly_6 * poly_139
    poly_921 = poly_2 * poly_355
    poly_922 = poly_6 * poly_140
    poly_923 = poly_2 * poly_357
    poly_924 = poly_2 * poly_358
    poly_925 = poly_10 * poly_132
    poly_926 = poly_10 * poly_133
    poly_927 = poly_2 * poly_360
    poly_928 = poly_136 * poly_8
    poly_929 = poly_2 * poly_361
    poly_930 = poly_10 * poly_136
    poly_931 = poly_2 * poly_363
    poly_932 = poly_2 * poly_366
    poly_933 = poly_2 * poly_369
    poly_934 = poly_2 * poly_371
    poly_935 = poly_2 * poly_374
    poly_936 = poly_5 * poly_328
    poly_937 = poly_2 * poly_377
    poly_938 = poly_2 * poly_378
    poly_939 = poly_5 * poly_331
    poly_940 = poly_2 * poly_380
    poly_941 = poly_2 * poly_381
    poly_942 = poly_5 * poly_334
    poly_943 = poly_2 * poly_383
    poly_944 = poly_5 * poly_336
    poly_945 = poly_2 * poly_384
    poly_946 = poly_2 * poly_385
    poly_947 = poly_5 * poly_339
    poly_948 = poly_5 * poly_340
    poly_949 = poly_2 * poly_387
    poly_950 = poly_2 * poly_388
    poly_951 = poly_5 * poly_343
    poly_952 = poly_2 * poly_390
    poly_953 = poly_5 * poly_345
    poly_954 = poly_2 * poly_392
    poly_955 = poly_2 * poly_393
    poly_956 = poly_5 * poly_348
    poly_957 = poly_5 * poly_349
    poly_958 = poly_2 * poly_395
    poly_959 = poly_5 * poly_351
    poly_960 = poly_2 * poly_396
    poly_961 = poly_5 * poly_353
    poly_962 = poly_2 * poly_397
    poly_963 = poly_5 * poly_355
    poly_964 = poly_2 * poly_399
    poly_965 = poly_5 * poly_357
    poly_966 = poly_5 * poly_358
    poly_967 = poly_2 * poly_400
    poly_968 = poly_5 * poly_360
    poly_969 = poly_5 * poly_361
    poly_970 = poly_2 * poly_403
    poly_971 = poly_2 * poly_323
    poly_972 = poly_2 * poly_407
    poly_973 = poly_2 * poly_412
    poly_974 = poly_2 * poly_413
    poly_975 = poly_2 * poly_324
    poly_976 = poly_2 * poly_418
    poly_977 = poly_2 * poly_421
    poly_978 = poly_2 * poly_325
    poly_979 = poly_2 * poly_424
    poly_980 = poly_2 * poly_425
    poly_981 = poly_2 * poly_326
    poly_982 = poly_2 * poly_430
    poly_983 = poly_2 * poly_431
    poly_984 = poly_2 * poly_327
    poly_985 = poly_6 * poly_117
    poly_986 = poly_6 * poly_118
    poly_987 = poly_2 * poly_436
    poly_988 = poly_2 * poly_437
    poly_989 = poly_2 * poly_329
    poly_990 = poly_2 * poly_330
    poly_991 = poly_6 * poly_199
    poly_992 = poly_2 * poly_439
    poly_993 = poly_2 * poly_440
    poly_994 = poly_6 * poly_120
    poly_995 = poly_6 * poly_121
    poly_996 = poly_2 * poly_442
    poly_997 = poly_2 * poly_443
    poly_998 = poly_2 * poly_332
    poly_999 = poly_2 * poly_333
    poly_1000 = poly_10 * poly_191
    poly_1001 = poly_2 * poly_335
    poly_1002 = poly_6 * poly_124
    poly_1003 = poly_6 * poly_125
    poly_1004 = poly_2 * poly_445
    poly_1005 = poly_2 * poly_446
    poly_1006 = poly_2 * poly_337
    poly_1007 = poly_2 * poly_338
    poly_1008 = poly_10 * poly_193
    poly_1009 = poly_10 * poly_194
    poly_1010 = poly_2 * poly_341
    poly_1011 = poly_2 * poly_448
    poly_1012 = poly_6 * poly_203
    poly_1013 = poly_2 * poly_451
    poly_1014 = poly_2 * poly_452
    poly_1015 = poly_6 * poly_204
    poly_1016 = poly_2 * poly_454
    poly_1017 = poly_2 * poly_455
    poly_1018 = poly_10 * poly_120
    poly_1019 = poly_10 * poly_121
    poly_1020 = poly_2 * poly_457
    poly_1021 = poly_6 * poly_205
    poly_1022 = poly_2 * poly_458
    poly_1023 = poly_2 * poly_459
    poly_1024 = poly_10 * poly_124
    poly_1025 = poly_10 * poly_125
    poly_1026 = poly_2 * poly_461
    poly_1027 = poly_2 * poly_462
    poly_1028 = poly_2 * poly_342
    poly_1029 = poly_6 * poly_130
    poly_1030 = poly_2 * poly_465
    poly_1031 = poly_2 * poly_344
    poly_1032 = poly_6 * poly_208
    poly_1033 = poly_2 * poly_467
    poly_1034 = poly_6 * poly_132
    poly_1035 = poly_6 * poly_133
    poly_1036 = poly_2 * poly_469
    poly_1037 = poly_2 * poly_470
    poly_1038 = poly_2 * poly_346
    poly_1039 = poly_2 * poly_347
    poly_1040 = poly_1 * poly_348 - poly_914
    poly_1041 = poly_1 * poly_349 - poly_914
    poly_1042 = poly_2 * poly_350
    poly_1043 = poly_3 * poly_348 - poly_925
    poly_1044 = poly_1 * poly_472 - poly_1043
    poly_1045 = poly_2 * poly_472
    poly_1046 = poly_136 * poly_23
    poly_1047 = poly_2 * poly_352
    poly_1048 = poly_6 * poly_137
    poly_1049 = poly_2 * poly_473
    poly_1050 = poly_2 * poly_354
    poly_1051 = poly_10 * poly_206
    poly_1052 = poly_2 * poly_356
    poly_1053 = poly_10 * poly_130
    poly_1054 = poly_2 * poly_475
    poly_1055 = poly_1 * poly_357 - poly_922
    poly_1056 = poly_1 * poly_358 - poly_922
    poly_1057 = poly_2 * poly_359
    poly_1058 = poly_10 * poly_135
    poly_1059 = poly_6 * poly_209
    poly_1060 = poly_2 * poly_476
    poly_1061 = poly_10 * poly_137
    poly_1062 = poly_2 * poly_478
    poly_1063 = poly_3 * poly_357 - poly_925
    poly_1064 = poly_1 * poly_479 - poly_1063
    poly_1065 = poly_2 * poly_479
    poly_1066 = poly_10 * poly_140
    poly_1067 = poly_136 * poly_25
    poly_1068 = poly_2 * poly_481
    poly_1069 = poly_2 * poly_484
    poly_1070 = poly_6 * poly_219
    poly_1071 = poly_2 * poly_487
    poly_1072 = poly_2 * poly_488
    poly_1073 = poly_2 * poly_490
    poly_1074 = poly_6 * poly_223
    poly_1075 = poly_2 * poly_493
    poly_1076 = poly_2 * poly_494
    poly_1077 = poly_6 * poly_224
    poly_1078 = poly_2 * poly_496
    poly_1079 = poly_2 * poly_497
    poly_1080 = jnp.take(mono, 64) + jnp.take(mono, 65) + \
        jnp.take(mono, 66) + jnp.take(mono, 67)
    poly_1081 = poly_1 * poly_499 - poly_1080
    poly_1082 = poly_2 * poly_499
    poly_1083 = poly_2 * poly_500
    poly_1084 = poly_6 * poly_228
    poly_1085 = poly_2 * poly_503
    poly_1086 = poly_2 * poly_504
    poly_1087 = poly_6 * poly_229
    poly_1088 = poly_2 * poly_506
    poly_1089 = poly_2 * poly_507
    poly_1090 = poly_10 * poly_216
    poly_1091 = poly_10 * poly_217
    poly_1092 = poly_2 * poly_509
    poly_1093 = poly_6 * poly_230
    poly_1094 = poly_2 * poly_510
    poly_1095 = poly_2 * poly_511
    poly_1096 = poly_10 * poly_220
    poly_1097 = poly_10 * poly_221
    poly_1098 = poly_2 * poly_513
    poly_1099 = poly_3 * poly_496 - poly_1080
    poly_1100 = poly_1 * poly_514 - poly_1099
    poly_1101 = poly_2 * poly_514
    poly_1102 = poly_10 * poly_224
    poly_1103 = poly_2 * poly_515
    poly_1104 = poly_6 * poly_234
    poly_1105 = poly_2 * poly_518
    poly_1106 = poly_2 * poly_519
    poly_1107 = poly_6 * poly_235
    poly_1108 = poly_2 * poly_521
    poly_1109 = poly_2 * poly_522
    poly_1110 = jnp.take(mono, 68) + jnp.take(mono, 69) + \
        jnp.take(mono, 70) + jnp.take(mono, 71)
    poly_1111 = poly_1 * poly_524 - poly_1110
    poly_1112 = poly_2 * poly_524
    poly_1113 = poly_6 * poly_236
    poly_1114 = poly_2 * poly_525
    poly_1115 = poly_2 * poly_526
    poly_1116 = poly_41 * poly_52
    poly_1117 = poly_42 * poly_52
    poly_1118 = poly_2 * poly_528
    poly_1119 = poly_136 * poly_11
    poly_1120 = poly_136 * poly_12
    poly_1121 = poly_2 * poly_529
    poly_1122 = poly_136 * poly_14
    poly_1123 = poly_6 * poly_237
    poly_1124 = poly_2 * poly_530
    poly_1125 = poly_2 * poly_531
    poly_1126 = poly_10 * poly_231
    poly_1127 = poly_10 * poly_232
    poly_1128 = poly_2 * poly_533
    poly_1129 = poly_3 * poly_521 - poly_1110
    poly_1130 = poly_1 * poly_534 - poly_1129
    poly_1131 = poly_2 * poly_534
    poly_1132 = poly_10 * poly_235
    poly_1133 = poly_41 * poly_54
    poly_1134 = poly_42 * poly_54
    poly_1135 = poly_2 * poly_535
    poly_1136 = poly_10 * poly_236
    poly_1137 = poly_136 * poly_17
    poly_1138 = poly_2 * poly_538
    poly_1139 = poly_2 * poly_362
    poly_1140 = poly_5 * poly_403
    poly_1141 = poly_2 * poly_541
    poly_1142 = poly_2 * poly_364
    poly_1143 = poly_2 * poly_543
    poly_1144 = poly_5 * poly_407
    poly_1145 = poly_2 * poly_545
    poly_1146 = poly_2 * poly_547
    poly_1147 = poly_2 * poly_548
    poly_1148 = poly_2 * poly_365
    poly_1149 = poly_5 * poly_412
    poly_1150 = poly_5 * poly_413
    poly_1151 = poly_2 * poly_553
    poly_1152 = poly_2 * poly_554
    poly_1153 = poly_2 * poly_367
    poly_1154 = poly_2 * poly_368
    poly_1155 = poly_5 * poly_418
    poly_1156 = poly_2 * poly_556
    poly_1157 = poly_2 * poly_557
    poly_1158 = poly_5 * poly_421
    poly_1159 = poly_2 * poly_559
    poly_1160 = poly_2 * poly_370
    poly_1161 = poly_5 * poly_424
    poly_1162 = poly_5 * poly_425
    poly_1163 = poly_2 * poly_561
    poly_1164 = poly_2 * poly_562
    poly_1165 = poly_2 * poly_372
    poly_1166 = poly_2 * poly_373
    poly_1167 = poly_5 * poly_430
    poly_1168 = poly_5 * poly_431
    poly_1169 = poly_2 * poly_564
    poly_1170 = poly_2 * poly_565
    poly_1171 = poly_2 * poly_375
    poly_1172 = poly_2 * poly_376
    poly_1173 = poly_5 * poly_436
    poly_1174 = poly_5 * poly_437
    poly_1175 = poly_2 * poly_379
    poly_1176 = poly_5 * poly_439
    poly_1177 = poly_5 * poly_440
    poly_1178 = poly_2 * poly_567
    poly_1179 = poly_5 * poly_442
    poly_1180 = poly_5 * poly_443
    poly_1181 = poly_2 * poly_382
    poly_1182 = poly_5 * poly_445
    poly_1183 = poly_5 * poly_446
    poly_1184 = poly_2 * poly_386
    poly_1185 = poly_5 * poly_448
    poly_1186 = poly_2 * poly_568
    poly_1187 = poly_2 * poly_569
    poly_1188 = poly_5 * poly_451
    poly_1189 = poly_5 * poly_452
    poly_1190 = poly_2 * poly_571
    poly_1191 = poly_5 * poly_454
    poly_1192 = poly_5 * poly_455
    poly_1193 = poly_2 * poly_572
    poly_1194 = poly_5 * poly_457
    poly_1195 = poly_5 * poly_458
    poly_1196 = poly_5 * poly_459
    poly_1197 = poly_2 * poly_573
    poly_1198 = poly_5 * poly_461
    poly_1199 = poly_5 * poly_462
    poly_1200 = poly_2 * poly_574
    poly_1201 = poly_2 * poly_389
    poly_1202 = poly_5 * poly_465
    poly_1203 = poly_2 * poly_391
    poly_1204 = poly_5 * poly_467
    poly_1205 = poly_2 * poly_576
    poly_1206 = poly_5 * poly_469
    poly_1207 = poly_5 * poly_470
    poly_1208 = poly_2 * poly_394
    poly_1209 = poly_5 * poly_472
    poly_1210 = poly_5 * poly_473
    poly_1211 = poly_2 * poly_398
    poly_1212 = poly_5 * poly_475
    poly_1213 = poly_5 * poly_476
    poly_1214 = poly_2 * poly_577
    poly_1215 = poly_5 * poly_478
    poly_1216 = poly_5 * poly_479
    poly_1217 = poly_2 * poly_578
    poly_1218 = poly_5 * poly_481
    poly_1219 = poly_2 * poly_581
    poly_1220 = poly_2 * poly_582
    poly_1221 = poly_5 * poly_484
    poly_1222 = poly_2 * poly_584
    poly_1223 = poly_2 * poly_585
    poly_1224 = poly_5 * poly_487
    poly_1225 = poly_5 * poly_488
    poly_1226 = poly_2 * poly_587
    poly_1227 = poly_5 * poly_490
    poly_1228 = poly_2 * poly_588
    poly_1229 = poly_2 * poly_589
    poly_1230 = poly_5 * poly_493
    poly_1231 = poly_5 * poly_494
    poly_1232 = poly_2 * poly_591
    poly_1233 = poly_5 * poly_496
    poly_1234 = poly_5 * poly_497
    poly_1235 = poly_2 * poly_592
    poly_1236 = poly_5 * poly_499
    poly_1237 = poly_5 * poly_500
    poly_1238 = poly_2 * poly_593
    poly_1239 = poly_2 * poly_594
    poly_1240 = poly_5 * poly_503
    poly_1241 = poly_5 * poly_504
    poly_1242 = poly_2 * poly_596
    poly_1243 = poly_5 * poly_506
    poly_1244 = poly_5 * poly_507
    poly_1245 = poly_2 * poly_597
    poly_1246 = poly_5 * poly_509
    poly_1247 = poly_5 * poly_510
    poly_1248 = poly_5 * poly_511
    poly_1249 = poly_2 * poly_598
    poly_1250 = poly_5 * poly_513
    poly_1251 = poly_5 * poly_514
    poly_1252 = poly_5 * poly_515
    poly_1253 = poly_2 * poly_599
    poly_1254 = poly_2 * poly_600
    poly_1255 = poly_5 * poly_518
    poly_1256 = poly_5 * poly_519
    poly_1257 = poly_2 * poly_602
    poly_1258 = poly_5 * poly_521
    poly_1259 = poly_5 * poly_522
    poly_1260 = poly_2 * poly_603
    poly_1261 = poly_5 * poly_524
    poly_1262 = poly_5 * poly_525
    poly_1263 = poly_5 * poly_526
    poly_1264 = poly_2 * poly_604
    poly_1265 = poly_5 * poly_528
    poly_1266 = poly_5 * poly_529
    poly_1267 = poly_5 * poly_530
    poly_1268 = poly_5 * poly_531
    poly_1269 = poly_2 * poly_605
    poly_1270 = poly_5 * poly_533
    poly_1271 = poly_5 * poly_534
    poly_1272 = poly_5 * poly_535
    poly_1273 = poly_2 * poly_607
    poly_1274 = poly_5 * poly_363
    poly_1275 = poly_2 * poly_609
    poly_1276 = poly_2 * poly_611
    poly_1277 = poly_5 * poly_366
    poly_1278 = poly_2 * poly_614
    poly_1279 = poly_2 * poly_615
    poly_1280 = poly_5 * poly_369
    poly_1281 = poly_2 * poly_617
    poly_1282 = poly_5 * poly_371
    poly_1283 = poly_2 * poly_619
    poly_1284 = poly_2 * poly_620
    poly_1285 = poly_5 * poly_374
    poly_1286 = poly_2 * poly_622
    poly_1287 = poly_2 * poly_623
    poly_1288 = poly_5 * poly_377
    poly_1289 = poly_5 * poly_378
    poly_1290 = poly_2 * poly_625
    poly_1291 = poly_5 * poly_380
    poly_1292 = poly_5 * poly_381
    poly_1293 = poly_2 * poly_626
    poly_1294 = poly_5 * poly_383
    poly_1295 = poly_5 * poly_384
    poly_1296 = poly_5 * poly_385
    poly_1297 = poly_2 * poly_627
    poly_1298 = poly_5 * poly_387
    poly_1299 = poly_5 * poly_388
    poly_1300 = poly_2 * poly_628
    poly_1301 = poly_5 * poly_390
    poly_1302 = poly_2 * poly_630
    poly_1303 = poly_5 * poly_392
    poly_1304 = poly_5 * poly_393
    poly_1305 = poly_2 * poly_631
    poly_1306 = poly_5 * poly_395
    poly_1307 = poly_5 * poly_396
    poly_1308 = poly_5 * poly_397
    poly_1309 = poly_2 * poly_632
    poly_1310 = poly_5 * poly_399
    poly_1311 = poly_5 * poly_400
    poly_1312 = poly_2 * poly_637
    poly_1313 = poly_2 * poly_638
    poly_1314 = poly_2 * poly_401
    poly_1315 = poly_2 * poly_402
    poly_1316 = poly_6 * poly_103
    poly_1317 = poly_6 * poly_173
    poly_1318 = poly_2 * poly_642
    poly_1319 = poly_2 * poly_404
    poly_1320 = poly_2 * poly_405
    poly_1321 = poly_2 * poly_644
    poly_1322 = poly_2 * poly_406
    poly_1323 = poly_6 * poly_177
    poly_1324 = poly_2 * poly_647
    poly_1325 = poly_2 * poly_408
    poly_1326 = poly_6 * poly_281
    poly_1327 = poly_2 * poly_649
    poly_1328 = poly_2 * poly_651
    poly_1329 = poly_6 * poly_284
    poly_1330 = poly_2 * poly_653
    poly_1331 = poly_2 * poly_655
    poly_1332 = poly_2 * poly_656
    poly_1333 = poly_2 * poly_657
    poly_1334 = poly_2 * poly_409
    poly_1335 = poly_2 * poly_410
    poly_1336 = poly_2 * poly_411
    poly_1337 = poly_6 * poly_185
    poly_1338 = poly_6 * poly_106
    poly_1339 = poly_6 * poly_186
    poly_1340 = poly_2 * poly_664
    poly_1341 = poly_2 * poly_665
    poly_1342 = poly_2 * poly_414
    poly_1343 = poly_2 * poly_415
    poly_1344 = poly_2 * poly_416
    poly_1345 = poly_2 * poly_417
    poly_1346 = poly_6 * poly_188
    poly_1347 = poly_6 * poly_189
    poly_1348 = poly_2 * poly_667
    poly_1349 = poly_2 * poly_668
    poly_1350 = poly_2 * poly_419
    poly_1351 = poly_2 * poly_420
    poly_1352 = poly_6 * poly_288
    poly_1353 = poly_2 * poly_670
    poly_1354 = poly_2 * poly_671
    poly_1355 = poly_6 * poly_109
    poly_1356 = poly_6 * poly_191
    poly_1357 = poly_2 * poly_673
    poly_1358 = poly_2 * poly_422
    poly_1359 = poly_2 * poly_423
    poly_1360 = poly_6 * poly_193
    poly_1361 = poly_6 * poly_111
    poly_1362 = poly_6 * poly_194
    poly_1363 = poly_2 * poly_675
    poly_1364 = poly_2 * poly_676
    poly_1365 = poly_2 * poly_426
    poly_1366 = poly_2 * poly_427
    poly_1367 = poly_2 * poly_428
    poly_1368 = poly_2 * poly_429
    poly_1369 = poly_6 * poly_196
    poly_1370 = poly_6 * poly_114
    poly_1371 = poly_6 * poly_197
    poly_1372 = poly_2 * poly_678
    poly_1373 = poly_2 * poly_679
    poly_1374 = poly_2 * poly_432
    poly_1375 = poly_2 * poly_433
    poly_1376 = poly_2 * poly_434
    poly_1377 = poly_2 * poly_435
    poly_1378 = poly_10 * poly_285
    poly_1379 = poly_10 * poly_286
    poly_1380 = poly_2 * poly_438
    poly_1381 = poly_10 * poly_185
    poly_1382 = poly_10 * poly_186
    poly_1383 = poly_2 * poly_441
    poly_1384 = poly_10 * poly_188
    poly_1385 = poly_10 * poly_189
    poly_1386 = poly_2 * poly_681
    poly_1387 = poly_1 * poly_442 - poly_994
    poly_1388 = poly_1 * poly_443 - poly_995
    poly_1389 = poly_2 * poly_444
    poly_1390 = poly_3 * poly_675
    poly_1391 = poly_3 * poly_676
    poly_1392 = poly_2 * poly_447
    poly_1393 = poly_6 * poly_200
    poly_1394 = poly_6 * poly_201
    poly_1395 = poly_2 * poly_682
    poly_1396 = poly_2 * poly_683
    poly_1397 = poly_2 * poly_449
    poly_1398 = poly_2 * poly_450
    poly_1399 = poly_10 * poly_196
    poly_1400 = poly_10 * poly_197
    poly_1401 = poly_2 * poly_453
    poly_1402 = poly_10 * poly_117
    poly_1403 = poly_10 * poly_118
    poly_1404 = poly_2 * poly_685
    poly_1405 = poly_1 * poly_454 - poly_1015
    poly_1406 = poly_1 * poly_455 - poly_1015
    poly_1407 = poly_2 * poly_456
    poly_1408 = poly_10 * poly_123
    poly_1409 = poly_25 * poly_193
    poly_1410 = poly_25 * poly_194
    poly_1411 = poly_2 * poly_460
    poly_1412 = poly_10 * poly_127
    poly_1413 = poly_6 * poly_289
    poly_1414 = poly_2 * poly_686
    poly_1415 = poly_2 * poly_687
    poly_1416 = poly_10 * poly_200
    poly_1417 = poly_10 * poly_201
    poly_1418 = poly_2 * poly_689
    poly_1419 = poly_3 * poly_454 - poly_1018
    poly_1420 = poly_1 * poly_690 - poly_1419
    poly_1421 = poly_2 * poly_690
    poly_1422 = poly_10 * poly_204
    poly_1423 = poly_41 * poly_99
    poly_1424 = poly_42 * poly_99
    poly_1425 = poly_2 * poly_691
    poly_1426 = poly_10 * poly_205
    poly_1427 = poly_6 * poly_128
    poly_1428 = poly_6 * poly_206
    poly_1429 = poly_2 * poly_692
    poly_1430 = poly_2 * poly_463
    poly_1431 = poly_2 * poly_464
    poly_1432 = poly_52 * poly_97
    poly_1433 = poly_2 * poly_466
    poly_1434 = poly_23 * poly_208
    poly_1435 = poly_2 * poly_468
    poly_1436 = poly_1 * poly_694
    poly_1437 = poly_2 * poly_694
    poly_1438 = poly_1 * poly_469 - poly_1034
    poly_1439 = poly_1 * poly_470 - poly_1035
    poly_1440 = poly_2 * poly_471
    poly_1441 = poly_3 * poly_472 - poly_1058
    poly_1442 = poly_54 * poly_97
    poly_1443 = poly_2 * poly_474
    poly_1444 = poly_10 * poly_208
    poly_1445 = poly_23 * poly_209
    poly_1446 = poly_2 * poly_477
    poly_1447 = poly_10 * poly_139
    poly_1448 = poly_1 * poly_695
    poly_1449 = poly_2 * poly_695
    poly_1450 = poly_10 * poly_209
    poly_1451 = poly_3 * poly_479 - poly_1066
    poly_1452 = poly_2 * poly_696
    poly_1453 = poly_2 * poly_697
    poly_1454 = poly_2 * poly_480
    poly_1455 = poly_6 * poly_213
    poly_1456 = poly_6 * poly_214
    poly_1457 = poly_2 * poly_702
    poly_1458 = poly_2 * poly_703
    poly_1459 = poly_2 * poly_482
    poly_1460 = poly_2 * poly_483
    poly_1461 = poly_6 * poly_293
    poly_1462 = poly_2 * poly_705
    poly_1463 = poly_2 * poly_706
    poly_1464 = poly_6 * poly_216
    poly_1465 = poly_6 * poly_217
    poly_1466 = poly_2 * poly_708
    poly_1467 = poly_2 * poly_709
    poly_1468 = poly_2 * poly_485
    poly_1469 = poly_2 * poly_486
    poly_1470 = poly_1 * poly_487 - poly_1070
    poly_1471 = poly_1 * poly_488 - poly_1070
    poly_1472 = poly_2 * poly_489
    poly_1473 = poly_3 * poly_487 - poly_1090
    poly_1474 = poly_1 * poly_711 - poly_1473
    poly_1475 = poly_2 * poly_711
    poly_1476 = poly_6 * poly_294
    poly_1477 = poly_2 * poly_712
    poly_1478 = poly_6 * poly_220
    poly_1479 = poly_6 * poly_221
    poly_1480 = poly_2 * poly_714
    poly_1481 = poly_2 * poly_715
    poly_1482 = poly_2 * poly_491
    poly_1483 = poly_2 * poly_492
    poly_1484 = poly_1 * poly_493 - poly_1074
    poly_1485 = poly_1 * poly_494 - poly_1074
    poly_1486 = poly_2 * poly_495
    poly_1487 = poly_3 * poly_493 - poly_1096
    poly_1488 = poly_1 * poly_717 - poly_1487
    poly_1489 = poly_2 * poly_717
    poly_1490 = poly_15 * poly_193
    poly_1491 = poly_15 * poly_194
    poly_1492 = poly_2 * poly_498
    poly_1493 = poly_3 * poly_499 - poly_1102
    poly_1494 = poly_6 * poly_295
    poly_1495 = poly_2 * poly_718
    poly_1496 = poly_2 * poly_719
    poly_1497 = poly_6 * poly_225
    poly_1498 = poly_6 * poly_226
    poly_1499 = poly_2 * poly_721
    poly_1500 = poly_2 * poly_722
    poly_1501 = poly_2 * poly_501
    poly_1502 = poly_2 * poly_502
    poly_1503 = poly_10 * poly_290
    poly_1504 = poly_10 * poly_291
    poly_1505 = poly_2 * poly_505
    poly_1506 = poly_10 * poly_213
    poly_1507 = poly_10 * poly_214
    poly_1508 = poly_2 * poly_724
    poly_1509 = poly_1 * poly_506 - poly_1087
    poly_1510 = poly_1 * poly_507 - poly_1087
    poly_1511 = poly_2 * poly_508
    poly_1512 = poly_10 * poly_219
    poly_1513 = poly_15 * poly_120 - poly_928
    poly_1514 = poly_1 * poly_725 - poly_1513
    poly_1515 = poly_2 * poly_725
    poly_1516 = poly_10 * poly_294
    poly_1517 = poly_1 * poly_510 - poly_1093
    poly_1518 = poly_1 * poly_511 - poly_1093
    poly_1519 = poly_2 * poly_512
    poly_1520 = poly_10 * poly_223
    poly_1521 = poly_3 * poly_718
    poly_1522 = poly_3 * poly_719
    poly_1523 = poly_2 * poly_726
    poly_1524 = poly_10 * poly_295
    poly_1525 = poly_6 * poly_296
    poly_1526 = poly_2 * poly_727
    poly_1527 = poly_2 * poly_728
    poly_1528 = poly_10 * poly_225
    poly_1529 = poly_10 * poly_226
    poly_1530 = poly_2 * poly_730
    poly_1531 = poly_3 * poly_506 - poly_1090
    poly_1532 = poly_1 * poly_731 - poly_1531
    poly_1533 = poly_2 * poly_731
    poly_1534 = poly_10 * poly_229
    poly_1535 = poly_3 * poly_510 - poly_1096
    poly_1536 = poly_1 * poly_732 - poly_1535
    poly_1537 = poly_2 * poly_732
    poly_1538 = poly_10 * poly_230
    poly_1539 = poly_3 * poly_514 - poly_1102
    poly_1540 = poly_6 * poly_231
    poly_1541 = poly_6 * poly_232
    poly_1542 = poly_2 * poly_733
    poly_1543 = poly_2 * poly_734
    poly_1544 = poly_2 * poly_516
    poly_1545 = poly_2 * poly_517
    poly_1546 = poly_1 * poly_518 - poly_1104
    poly_1547 = poly_1 * poly_519 - poly_1104
    poly_1548 = poly_2 * poly_520
    poly_1549 = poly_3 * poly_518 - poly_1126
    poly_1550 = poly_1 * poly_736 - poly_1549
    poly_1551 = poly_2 * poly_736
    poly_1552 = poly_1 * poly_521 - poly_1107
    poly_1553 = poly_1 * poly_522 - poly_1107
    poly_1554 = poly_2 * poly_523
    poly_1555 = poly_15 * poly_208
    poly_1556 = poly_15 * poly_133 - poly_1120
    poly_1557 = poly_1 * poly_737 - poly_1556
    poly_1558 = poly_2 * poly_737
    poly_1559 = poly_15 * poly_135 - poly_1122
    poly_1560 = poly_18 * poly_193
    poly_1561 = poly_18 * poly_194
    poly_1562 = poly_2 * poly_527
    poly_1563 = poly_16 * poly_208
    poly_1564 = poly_136 * poly_15
    poly_1565 = poly_16 * poly_132 - poly_1120
    poly_1566 = poly_1 * poly_738 - poly_1565
    poly_1567 = poly_2 * poly_738
    poly_1568 = poly_16 * poly_135 - poly_1122
    poly_1569 = poly_136 * poly_16
    poly_1570 = poly_1 * poly_530 - poly_1123
    poly_1571 = poly_1 * poly_531 - poly_1123
    poly_1572 = poly_2 * poly_532
    poly_1573 = poly_10 * poly_234
    poly_1574 = poly_3 * poly_737 - poly_1559
    poly_1575 = poly_3 * poly_738 - poly_1568
    poly_1576 = poly_3 * poly_530 - poly_1126
    poly_1577 = poly_1 * poly_739 - poly_1576
    poly_1578 = poly_2 * poly_739
    poly_1579 = poly_10 * poly_237
    poly_1580 = poly_15 * poly_209
    poly_1581 = poly_16 * poly_209
    poly_1582 = poly_6 * poly_297
    poly_1583 = poly_2 * poly_740
    poly_1584 = poly_1 * poly_742
    poly_1585 = poly_2 * poly_742
    poly_1586 = poly_18 * poly_132 - poly_1119
    poly_1587 = poly_1 * poly_743 - poly_1586
    poly_1588 = poly_2 * poly_743
    poly_1589 = poly_18 * poly_135 - poly_1137
    poly_1590 = poly_136 * poly_18
    poly_1591 = poly_1 * poly_744
    poly_1592 = poly_2 * poly_744
    poly_1593 = poly_10 * poly_297
    poly_1594 = poly_3 * poly_743 - poly_1589
    poly_1595 = poly_2 * poly_745
    poly_1596 = poly_6 * poly_301
    poly_1597 = poly_2 * poly_748
    poly_1598 = poly_2 * poly_749
    poly_1599 = poly_6 * poly_302
    poly_1600 = poly_2 * poly_751
    poly_1601 = poly_2 * poly_752
    poly_1602 = poly_15 * poly_213 - poly_1116
    poly_1603 = poly_1 * poly_754 - poly_1602
    poly_1604 = poly_2 * poly_754
    poly_1605 = poly_6 * poly_303
    poly_1606 = poly_2 * poly_755
    poly_1607 = poly_2 * poly_756
    poly_1608 = poly_16 * poly_213 - poly_1110
    poly_1609 = poly_1 * poly_758 - poly_1608
    poly_1610 = poly_2 * poly_758
    poly_1611 = poly_15 * poly_220 - poly_1565
    poly_1612 = poly_1 * poly_759 - poly_1611
    poly_1613 = poly_2 * poly_759
    poly_1614 = poly_15 * poly_223 - poly_1568
    poly_1615 = poly_6 * poly_304
    poly_1616 = poly_2 * poly_760
    poly_1617 = poly_2 * poly_761
    poly_1618 = poly_10 * poly_298
    poly_1619 = poly_10 * poly_299
    poly_1620 = poly_2 * poly_763
    poly_1621 = poly_3 * poly_751 - poly_1602
    poly_1622 = poly_1 * poly_764 - poly_1621
    poly_1623 = poly_2 * poly_764
    poly_1624 = poly_10 * poly_302
    poly_1625 = poly_3 * poly_755 - poly_1608
    poly_1626 = poly_1 * poly_765 - poly_1625
    poly_1627 = poly_2 * poly_765
    poly_1628 = poly_10 * poly_303
    poly_1629 = poly_3 * poly_759 - poly_1614
    poly_1630 = poly_6 * poly_305
    poly_1631 = poly_2 * poly_766
    poly_1632 = poly_2 * poly_767
    poly_1633 = poly_18 * poly_213 - poly_1080
    poly_1634 = poly_1 * poly_769 - poly_1633
    poly_1635 = poly_2 * poly_769
    poly_1636 = poly_15 * poly_231 - poly_1586
    poly_1637 = poly_1 * poly_770 - poly_1636
    poly_1638 = poly_2 * poly_770
    poly_1639 = poly_15 * poly_234 - poly_1589
    poly_1640 = poly_16 * poly_231 - poly_1587
    poly_1641 = poly_1 * poly_771 - poly_1640
    poly_1642 = poly_2 * poly_771
    poly_1643 = poly_16 * poly_234 - poly_1589
    poly_1644 = poly_136 * poly_26
    poly_1645 = poly_3 * poly_766 - poly_1633
    poly_1646 = poly_1 * poly_772 - poly_1645
    poly_1647 = poly_2 * poly_772
    poly_1648 = poly_10 * poly_305
    poly_1649 = poly_3 * poly_770 - poly_1639
    poly_1650 = poly_3 * poly_771 - poly_1643
    poly_1651 = poly_2 * poly_773
    poly_1652 = poly_2 * poly_774
    poly_1653 = poly_2 * poly_536
    poly_1654 = poly_2 * poly_537
    poly_1655 = poly_5 * poly_637
    poly_1656 = poly_5 * poly_638
    poly_1657 = poly_2 * poly_778
    poly_1658 = poly_2 * poly_539
    poly_1659 = poly_2 * poly_540
    poly_1660 = poly_5 * poly_642
    poly_1661 = poly_2 * poly_542
    poly_1662 = poly_5 * poly_644
    poly_1663 = poly_2 * poly_780
    poly_1664 = poly_2 * poly_544
    poly_1665 = poly_5 * poly_647
    poly_1666 = poly_2 * poly_546
    poly_1667 = poly_5 * poly_649
    poly_1668 = poly_2 * poly_782
    poly_1669 = poly_5 * poly_651
    poly_1670 = poly_2 * poly_783
    poly_1671 = poly_5 * poly_653
    poly_1672 = poly_2 * poly_785
    poly_1673 = poly_5 * poly_655
    poly_1674 = poly_5 * poly_656
    poly_1675 = poly_5 * poly_657
    poly_1676 = poly_2 * poly_786
    poly_1677 = poly_2 * poly_787
    poly_1678 = poly_2 * poly_549
    poly_1679 = poly_2 * poly_550
    poly_1680 = poly_2 * poly_551
    poly_1681 = poly_2 * poly_552
    poly_1682 = poly_5 * poly_664
    poly_1683 = poly_5 * poly_665
    poly_1684 = poly_2 * poly_555
    poly_1685 = poly_5 * poly_667
    poly_1686 = poly_5 * poly_668
    poly_1687 = poly_2 * poly_558
    poly_1688 = poly_5 * poly_670
    poly_1689 = poly_5 * poly_671
    poly_1690 = poly_2 * poly_789
    poly_1691 = poly_5 * poly_673
    poly_1692 = poly_2 * poly_560
    poly_1693 = poly_5 * poly_675
    poly_1694 = poly_5 * poly_676
    poly_1695 = poly_2 * poly_563
    poly_1696 = poly_5 * poly_678
    poly_1697 = poly_5 * poly_679
    poly_1698 = poly_2 * poly_566
    poly_1699 = poly_5 * poly_681
    poly_1700 = poly_5 * poly_682
    poly_1701 = poly_5 * poly_683
    poly_1702 = poly_2 * poly_570
    poly_1703 = poly_5 * poly_685
    poly_1704 = poly_5 * poly_686
    poly_1705 = poly_5 * poly_687
    poly_1706 = poly_2 * poly_790
    poly_1707 = poly_5 * poly_689
    poly_1708 = poly_5 * poly_690
    poly_1709 = poly_5 * poly_691
    poly_1710 = poly_5 * poly_692
    poly_1711 = poly_2 * poly_575
    poly_1712 = poly_5 * poly_694
    poly_1713 = poly_5 * poly_695
    poly_1714 = poly_5 * poly_696
    poly_1715 = poly_5 * poly_697
    poly_1716 = poly_2 * poly_791
    poly_1717 = poly_2 * poly_792
    poly_1718 = poly_2 * poly_579
    poly_1719 = poly_2 * poly_580
    poly_1720 = poly_5 * poly_702
    poly_1721 = poly_5 * poly_703
    poly_1722 = poly_2 * poly_583
    poly_1723 = poly_5 * poly_705
    poly_1724 = poly_5 * poly_706
    poly_1725 = poly_2 * poly_794
    poly_1726 = poly_5 * poly_708
    poly_1727 = poly_5 * poly_709
    poly_1728 = poly_2 * poly_586
    poly_1729 = poly_5 * poly_711
    poly_1730 = poly_5 * poly_712
    poly_1731 = poly_2 * poly_795
    poly_1732 = poly_5 * poly_714
    poly_1733 = poly_5 * poly_715
    poly_1734 = poly_2 * poly_590
    poly_1735 = poly_5 * poly_717
    poly_1736 = poly_5 * poly_718
    poly_1737 = poly_5 * poly_719
    poly_1738 = poly_2 * poly_796
    poly_1739 = poly_5 * poly_721
    poly_1740 = poly_5 * poly_722
    poly_1741 = poly_2 * poly_595
    poly_1742 = poly_5 * poly_724
    poly_1743 = poly_5 * poly_725
    poly_1744 = poly_5 * poly_726
    poly_1745 = poly_5 * poly_727
    poly_1746 = poly_5 * poly_728
    poly_1747 = poly_2 * poly_797
    poly_1748 = poly_5 * poly_730
    poly_1749 = poly_5 * poly_731
    poly_1750 = poly_5 * poly_732
    poly_1751 = poly_5 * poly_733
    poly_1752 = poly_5 * poly_734
    poly_1753 = poly_2 * poly_601
    poly_1754 = poly_5 * poly_736
    poly_1755 = poly_5 * poly_737
    poly_1756 = poly_5 * poly_738
    poly_1757 = poly_5 * poly_739
    poly_1758 = poly_5 * poly_740
    poly_1759 = poly_2 * poly_798
    poly_1760 = poly_5 * poly_742
    poly_1761 = poly_5 * poly_743
    poly_1762 = poly_5 * poly_744
    poly_1763 = poly_5 * poly_745
    poly_1764 = poly_2 * poly_799
    poly_1765 = poly_2 * poly_800
    poly_1766 = poly_5 * poly_748
    poly_1767 = poly_5 * poly_749
    poly_1768 = poly_2 * poly_802
    poly_1769 = poly_5 * poly_751
    poly_1770 = poly_5 * poly_752
    poly_1771 = poly_2 * poly_803
    poly_1772 = poly_5 * poly_754
    poly_1773 = poly_5 * poly_755
    poly_1774 = poly_5 * poly_756
    poly_1775 = poly_2 * poly_804
    poly_1776 = poly_5 * poly_758
    poly_1777 = poly_5 * poly_759
    poly_1778 = poly_5 * poly_760
    poly_1779 = poly_5 * poly_761
    poly_1780 = poly_2 * poly_805
    poly_1781 = poly_5 * poly_763
    poly_1782 = poly_5 * poly_764
    poly_1783 = poly_5 * poly_765
    poly_1784 = poly_5 * poly_766
    poly_1785 = poly_5 * poly_767
    poly_1786 = poly_2 * poly_806
    poly_1787 = poly_5 * poly_769
    poly_1788 = poly_5 * poly_770
    poly_1789 = poly_5 * poly_771
    poly_1790 = poly_5 * poly_772
    poly_1791 = poly_2 * poly_807
    poly_1792 = poly_2 * poly_606
    poly_1793 = poly_5 * poly_538
    poly_1794 = poly_2 * poly_810
    poly_1795 = poly_2 * poly_608
    poly_1796 = poly_5 * poly_541
    poly_1797 = poly_2 * poly_610
    poly_1798 = poly_5 * poly_543
    poly_1799 = poly_2 * poly_812
    poly_1800 = poly_5 * poly_545
    poly_1801 = poly_2 * poly_814
    poly_1802 = poly_5 * poly_547
    poly_1803 = poly_5 * poly_548
    poly_1804 = poly_2 * poly_815
    poly_1805 = poly_2 * poly_816
    poly_1806 = poly_2 * poly_612
    poly_1807 = poly_2 * poly_613
    poly_1808 = poly_5 * poly_553
    poly_1809 = poly_5 * poly_554
    poly_1810 = poly_2 * poly_616
    poly_1811 = poly_5 * poly_556
    poly_1812 = poly_5 * poly_557
    poly_1813 = poly_2 * poly_818
    poly_1814 = poly_5 * poly_559
    poly_1815 = poly_2 * poly_618
    poly_1816 = poly_5 * poly_561
    poly_1817 = poly_5 * poly_562
    poly_1818 = poly_2 * poly_621
    poly_1819 = poly_5 * poly_564
    poly_1820 = poly_5 * poly_565
    poly_1821 = poly_2 * poly_624
    poly_1822 = poly_5 * poly_567
    poly_1823 = poly_5 * poly_568
    poly_1824 = poly_5 * poly_569
    poly_1825 = poly_2 * poly_819
    poly_1826 = poly_5 * poly_571
    poly_1827 = poly_5 * poly_572
    poly_1828 = poly_5 * poly_573
    poly_1829 = poly_5 * poly_574
    poly_1830 = poly_2 * poly_629
    poly_1831 = poly_5 * poly_576
    poly_1832 = poly_5 * poly_577
    poly_1833 = poly_5 * poly_578
    poly_1834 = poly_2 * poly_820
    poly_1835 = poly_2 * poly_821
    poly_1836 = poly_5 * poly_581
    poly_1837 = poly_5 * poly_582
    poly_1838 = poly_2 * poly_823
    poly_1839 = poly_5 * poly_584
    poly_1840 = poly_5 * poly_585
    poly_1841 = poly_2 * poly_824
    poly_1842 = poly_5 * poly_587
    poly_1843 = poly_5 * poly_588
    poly_1844 = poly_5 * poly_589
    poly_1845 = poly_2 * poly_825
    poly_1846 = poly_5 * poly_591
    poly_1847 = poly_5 * poly_592
    poly_1848 = poly_5 * poly_593
    poly_1849 = poly_5 * poly_594
    poly_1850 = poly_2 * poly_826
    poly_1851 = poly_5 * poly_596
    poly_1852 = poly_5 * poly_597
    poly_1853 = poly_5 * poly_598
    poly_1854 = poly_5 * poly_599
    poly_1855 = poly_5 * poly_600
    poly_1856 = poly_2 * poly_827
    poly_1857 = poly_5 * poly_602
    poly_1858 = poly_5 * poly_603
    poly_1859 = poly_5 * poly_604
    poly_1860 = poly_5 * poly_605
    poly_1861 = poly_2 * poly_828
    poly_1862 = poly_5 * poly_607
    poly_1863 = poly_2 * poly_830
    poly_1864 = poly_5 * poly_609
    poly_1865 = poly_2 * poly_832
    poly_1866 = poly_5 * poly_611
    poly_1867 = poly_2 * poly_833
    poly_1868 = poly_2 * poly_834
    poly_1869 = poly_5 * poly_614
    poly_1870 = poly_5 * poly_615
    poly_1871 = poly_2 * poly_836
    poly_1872 = poly_5 * poly_617
    poly_1873 = poly_2 * poly_837
    poly_1874 = poly_5 * poly_619
    poly_1875 = poly_5 * poly_620
    poly_1876 = poly_2 * poly_838
    poly_1877 = poly_5 * poly_622
    poly_1878 = poly_5 * poly_623
    poly_1879 = poly_2 * poly_839
    poly_1880 = poly_5 * poly_625
    poly_1881 = poly_5 * poly_626
    poly_1882 = poly_5 * poly_627
    poly_1883 = poly_5 * poly_628
    poly_1884 = poly_2 * poly_840
    poly_1885 = poly_5 * poly_630
    poly_1886 = poly_5 * poly_631
    poly_1887 = poly_5 * poly_632
    poly_1888 = poly_2 * poly_841
    poly_1889 = poly_2 * poly_842
    poly_1890 = poly_2 * poly_633
    poly_1891 = poly_2 * poly_634
    poly_1892 = poly_2 * poly_635
    poly_1893 = poly_2 * poly_636
    poly_1894 = poly_6 * poly_170
    poly_1895 = poly_6 * poly_277
    poly_1896 = poly_2 * poly_847
    poly_1897 = poly_2 * poly_639
    poly_1898 = poly_2 * poly_640
    poly_1899 = poly_2 * poly_641
    poly_1900 = poly_10 * poly_318
    poly_1901 = poly_2 * poly_643
    poly_1902 = poly_6 * poly_175
    poly_1903 = poly_6 * poly_279
    poly_1904 = poly_2 * poly_849
    poly_1905 = poly_2 * poly_645
    poly_1906 = poly_2 * poly_646
    poly_1907 = poly_10 * poly_277
    poly_1908 = poly_2 * poly_648
    poly_1909 = poly_10 * poly_173
    poly_1910 = poly_2 * poly_650
    poly_1911 = poly_6 * poly_282
    poly_1912 = poly_2 * poly_851
    poly_1913 = poly_2 * poly_652
    poly_1914 = poly_10 * poly_279
    poly_1915 = poly_2 * poly_654
    poly_1916 = poly_10 * poly_177
    poly_1917 = poly_2 * poly_853
    poly_1918 = poly_6 * poly_320
    poly_1919 = poly_2 * poly_854
    poly_1920 = poly_10 * poly_282
    poly_1921 = poly_2 * poly_856
    poly_1922 = poly_6 * poly_285
    poly_1923 = poly_6 * poly_179
    poly_1924 = poly_6 * poly_180
    poly_1925 = poly_6 * poly_286
    poly_1926 = poly_2 * poly_857
    poly_1927 = poly_2 * poly_858
    poly_1928 = poly_2 * poly_658
    poly_1929 = poly_2 * poly_659
    poly_1930 = poly_2 * poly_660
    poly_1931 = poly_2 * poly_661
    poly_1932 = poly_2 * poly_662
    poly_1933 = poly_2 * poly_663
    poly_1934 = poly_1 * poly_664 - poly_1337
    poly_1935 = poly_1 * poly_665 - poly_1339
    poly_1936 = poly_2 * poly_666
    poly_1937 = poly_1 * poly_667 - poly_1346
    poly_1938 = poly_1 * poly_668 - poly_1347
    poly_1939 = poly_2 * poly_669
    poly_1940 = poly_1 * poly_670 - poly_1352
    poly_1941 = poly_1 * poly_671 - poly_1352
    poly_1942 = poly_2 * poly_672
    poly_1943 = poly_3 * poly_670 - poly_1384
    poly_1944 = poly_1 * poly_860 - poly_1943
    poly_1945 = poly_2 * poly_860
    poly_1946 = poly_15 * poly_318
    poly_1947 = poly_2 * poly_674
    poly_1948 = poly_1 * poly_675 - poly_1360
    poly_1949 = poly_1 * poly_676 - poly_1362
    poly_1950 = poly_2 * poly_677
    poly_1951 = poly_1 * poly_678 - poly_1369
    poly_1952 = poly_1 * poly_679 - poly_1371
    poly_1953 = poly_2 * poly_680
    poly_1954 = poly_10 * poly_288
    poly_1955 = poly_1 * poly_682 - poly_1393
    poly_1956 = poly_1 * poly_683 - poly_1394
    poly_1957 = poly_2 * poly_684
    poly_1958 = poly_10 * poly_199
    poly_1959 = poly_1 * poly_686 - poly_1413
    poly_1960 = poly_1 * poly_687 - poly_1413
    poly_1961 = poly_2 * poly_688
    poly_1962 = poly_10 * poly_203
    poly_1963 = poly_3 * poly_686 - poly_1416
    poly_1964 = poly_1 * poly_861 - poly_1963
    poly_1965 = poly_2 * poly_861
    poly_1966 = poly_10 * poly_289
    poly_1967 = poly_15 * poly_320
    poly_1968 = poly_16 * poly_320
    poly_1969 = poly_18 * poly_318
    poly_1970 = poly_2 * poly_693
    poly_1971 = poly_3 * poly_694 - poly_1444
    poly_1972 = poly_3 * poly_695 - poly_1450
    poly_1973 = poly_6 * poly_290
    poly_1974 = poly_6 * poly_210
    poly_1975 = poly_6 * poly_291
    poly_1976 = poly_2 * poly_862
    poly_1977 = poly_2 * poly_863
    poly_1978 = poly_2 * poly_698
    poly_1979 = poly_2 * poly_699
    poly_1980 = poly_2 * poly_700
    poly_1981 = poly_2 * poly_701
    poly_1982 = poly_1 * poly_702 - poly_1455
    poly_1983 = poly_1 * poly_703 - poly_1456
    poly_1984 = poly_2 * poly_704
    poly_1985 = poly_1 * poly_705 - poly_1461
    poly_1986 = poly_1 * poly_706 - poly_1461
    poly_1987 = poly_2 * poly_707
    poly_1988 = poly_3 * poly_705 - poly_1506
    poly_1989 = poly_1 * poly_865 - poly_1988
    poly_1990 = poly_2 * poly_865
    poly_1991 = poly_1 * poly_708 - poly_1464
    poly_1992 = poly_1 * poly_709 - poly_1465
    poly_1993 = poly_2 * poly_710
    poly_1994 = poly_3 * poly_711 - poly_1512
    poly_1995 = poly_23 * poly_294
    poly_1996 = poly_2 * poly_713
    poly_1997 = poly_1 * poly_714 - poly_1478
    poly_1998 = poly_1 * poly_715 - poly_1479
    poly_1999 = poly_2 * poly_716
    poly_2000 = poly_3 * poly_717 - poly_1520
    poly_2001 = poly_1 * poly_718 - poly_1494
    poly_2002 = poly_1 * poly_719 - poly_1494
    poly_2003 = poly_2 * poly_720
    poly_2004 = poly_1 * poly_721 - poly_1497
    poly_2005 = poly_1 * poly_722 - poly_1498
    poly_2006 = poly_2 * poly_723
    poly_2007 = poly_10 * poly_293
    poly_2008 = poly_1 * poly_727 - poly_1525
    poly_2009 = poly_1 * poly_728 - poly_1525
    poly_2010 = poly_2 * poly_729
    poly_2011 = poly_10 * poly_228
    poly_2012 = poly_25 * poly_294
    poly_2013 = poly_25 * poly_295
    poly_2014 = poly_3 * poly_727 - poly_1528
    poly_2015 = poly_1 * poly_866 - poly_2014
    poly_2016 = poly_2 * poly_866
    poly_2017 = poly_10 * poly_296
    poly_2018 = poly_3 * poly_731 - poly_1534
    poly_2019 = poly_3 * poly_732 - poly_1538
    poly_2020 = poly_1 * poly_733 - poly_1540
    poly_2021 = poly_1 * poly_734 - poly_1541
    poly_2022 = poly_2 * poly_735
    poly_2023 = poly_3 * poly_736 - poly_1573
    poly_2024 = poly_3 * poly_739 - poly_1579
    poly_2025 = poly_23 * poly_297
    poly_2026 = poly_2 * poly_741
    poly_2027 = poly_3 * poly_742 - poly_1593
    poly_2028 = poly_15 * poly_236 - poly_1644
    poly_2029 = poly_3 * poly_744 - poly_1593
    poly_2030 = poly_6 * poly_298
    poly_2031 = poly_6 * poly_299
    poly_2032 = poly_2 * poly_867
    poly_2033 = poly_2 * poly_868
    poly_2034 = poly_2 * poly_746
    poly_2035 = poly_2 * poly_747
    poly_2036 = poly_1 * poly_748 - poly_1596
    poly_2037 = poly_1 * poly_749 - poly_1596
    poly_2038 = poly_2 * poly_750
    poly_2039 = poly_3 * poly_748 - poly_1618
    poly_2040 = poly_1 * poly_870 - poly_2039
    poly_2041 = poly_2 * poly_870
    poly_2042 = poly_1 * poly_751 - poly_1599
    poly_2043 = poly_1 * poly_752 - poly_1599
    poly_2044 = poly_2 * poly_753
    poly_2045 = poly_3 * poly_754 - poly_1624
    poly_2046 = poly_11 * poly_294 - poly_1557
    poly_2047 = poly_1 * poly_871 - poly_2046
    poly_2048 = poly_2 * poly_871
    poly_2049 = poly_14 * poly_294 - poly_1559
    poly_2050 = poly_1 * poly_755 - poly_1605
    poly_2051 = poly_1 * poly_756 - poly_1605
    poly_2052 = poly_2 * poly_757
    poly_2053 = poly_3 * poly_758 - poly_1628
    poly_2054 = poly_16 * poly_294
    poly_2055 = poly_4 * poly_718 - poly_1565
    poly_2056 = poly_1 * poly_872 - poly_2055
    poly_2057 = poly_2 * poly_872
    poly_2058 = poly_14 * poly_295 - poly_1568
    poly_2059 = poly_15 * poly_295
    poly_2060 = poly_1 * poly_760 - poly_1615
    poly_2061 = poly_1 * poly_761 - poly_1615
    poly_2062 = poly_2 * poly_762
    poly_2063 = poly_10 * poly_301
    poly_2064 = poly_3 * poly_871 - poly_2049
    poly_2065 = poly_3 * poly_872 - poly_2058
    poly_2066 = poly_3 * poly_760 - poly_1618
    poly_2067 = poly_1 * poly_873 - poly_2066
    poly_2068 = poly_2 * poly_873
    poly_2069 = poly_10 * poly_304
    poly_2070 = poly_3 * poly_764 - poly_1624
    poly_2071 = poly_3 * poly_765 - poly_1628
    poly_2072 = poly_1 * poly_766 - poly_1630
    poly_2073 = poly_1 * poly_767 - poly_1630
    poly_2074 = poly_2 * poly_768
    poly_2075 = poly_3 * poly_769 - poly_1648
    poly_2076 = poly_18 * poly_294
    poly_2077 = poly_18 * poly_295
    poly_2078 = poly_3 * poly_772 - poly_1648
    poly_2079 = poly_11 * poly_297 - poly_1586
    poly_2080 = poly_1 * poly_874 - poly_2079
    poly_2081 = poly_2 * poly_874
    poly_2082 = poly_4 * poly_742 - poly_1589
    poly_2083 = poly_15 * poly_297
    poly_2084 = poly_16 * poly_297
    poly_2085 = poly_3 * poly_874 - poly_2082
    poly_2086 = poly_6 * poly_321
    poly_2087 = poly_2 * poly_875
    poly_2088 = poly_2 * poly_876
    poly_2089 = poly_4 * poly_748 - poly_1633 - poly_1608 - poly_1602
    poly_2090 = poly_1 * poly_878 - poly_2089
    poly_2091 = poly_2 * poly_878
    poly_2092 = poly_15 * poly_298 - poly_1640
    poly_2093 = poly_1 * poly_879 - poly_2092
    poly_2094 = poly_2 * poly_879
    poly_2095 = poly_15 * poly_301 - poly_1643
    poly_2096 = poly_16 * poly_298 - poly_1636
    poly_2097 = poly_1 * poly_880 - poly_2096
    poly_2098 = poly_2 * poly_880
    poly_2099 = poly_16 * poly_301 - poly_1639
    poly_2100 = poly_15 * poly_303 - poly_2077
    poly_2101 = poly_3 * poly_875 - poly_2089
    poly_2102 = poly_1 * poly_881 - poly_2101
    poly_2103 = poly_2 * poly_881
    poly_2104 = poly_10 * poly_321
    poly_2105 = poly_3 * poly_879 - poly_2095
    poly_2106 = poly_3 * poly_880 - poly_2099
    poly_2107 = poly_18 * poly_298 - poly_1611
    poly_2108 = poly_1 * poly_882 - poly_2107
    poly_2109 = poly_2 * poly_882
    poly_2110 = poly_18 * poly_301 - poly_1614
    poly_2111 = poly_15 * poly_305 - poly_2084
    poly_2112 = poly_16 * poly_305 - poly_2083
    poly_2113 = poly_3 * poly_882 - poly_2110
    poly_2114 = poly_5 * poly_841
    poly_2115 = poly_5 * poly_842
    poly_2116 = poly_2 * poly_883
    poly_2117 = poly_2 * poly_775
    poly_2118 = poly_2 * poly_776
    poly_2119 = poly_2 * poly_777
    poly_2120 = poly_5 * poly_847
    poly_2121 = poly_2 * poly_779
    poly_2122 = poly_5 * poly_849
    poly_2123 = poly_2 * poly_781
    poly_2124 = poly_5 * poly_851
    poly_2125 = poly_2 * poly_784
    poly_2126 = poly_5 * poly_853
    poly_2127 = poly_5 * poly_854
    poly_2128 = poly_2 * poly_885
    poly_2129 = poly_5 * poly_856
    poly_2130 = poly_5 * poly_857
    poly_2131 = poly_5 * poly_858
    poly_2132 = poly_2 * poly_788
    poly_2133 = poly_5 * poly_860
    poly_2134 = poly_5 * poly_861
    poly_2135 = poly_5 * poly_862
    poly_2136 = poly_5 * poly_863
    poly_2137 = poly_2 * poly_793
    poly_2138 = poly_5 * poly_865
    poly_2139 = poly_5 * poly_866
    poly_2140 = poly_5 * poly_867
    poly_2141 = poly_5 * poly_868
    poly_2142 = poly_2 * poly_801
    poly_2143 = poly_5 * poly_870
    poly_2144 = poly_5 * poly_871
    poly_2145 = poly_5 * poly_872
    poly_2146 = poly_5 * poly_873
    poly_2147 = poly_5 * poly_874
    poly_2148 = poly_5 * poly_875
    poly_2149 = poly_5 * poly_876
    poly_2150 = poly_2 * poly_886
    poly_2151 = poly_5 * poly_878
    poly_2152 = poly_5 * poly_879
    poly_2153 = poly_5 * poly_880
    poly_2154 = poly_5 * poly_881
    poly_2155 = poly_5 * poly_882
    poly_2156 = poly_5 * poly_773
    poly_2157 = poly_5 * poly_774
    poly_2158 = poly_2 * poly_887
    poly_2159 = poly_2 * poly_808
    poly_2160 = poly_2 * poly_809
    poly_2161 = poly_5 * poly_778
    poly_2162 = poly_2 * poly_811
    poly_2163 = poly_5 * poly_780
    poly_2164 = poly_2 * poly_813
    poly_2165 = poly_5 * poly_782
    poly_2166 = poly_5 * poly_783
    poly_2167 = poly_2 * poly_889
    poly_2168 = poly_5 * poly_785
    poly_2169 = poly_5 * poly_786
    poly_2170 = poly_5 * poly_787
    poly_2171 = poly_2 * poly_817
    poly_2172 = poly_5 * poly_789
    poly_2173 = poly_5 * poly_790
    poly_2174 = poly_5 * poly_791
    poly_2175 = poly_5 * poly_792
    poly_2176 = poly_2 * poly_822
    poly_2177 = poly_5 * poly_794
    poly_2178 = poly_5 * poly_795
    poly_2179 = poly_5 * poly_796
    poly_2180 = poly_5 * poly_797
    poly_2181 = poly_5 * poly_798
    poly_2182 = poly_5 * poly_799
    poly_2183 = poly_5 * poly_800
    poly_2184 = poly_2 * poly_890
    poly_2185 = poly_5 * poly_802
    poly_2186 = poly_5 * poly_803
    poly_2187 = poly_5 * poly_804
    poly_2188 = poly_5 * poly_805
    poly_2189 = poly_5 * poly_806
    poly_2190 = poly_5 * poly_807
    poly_2191 = poly_2 * poly_891
    poly_2192 = poly_2 * poly_829
    poly_2193 = poly_5 * poly_810
    poly_2194 = poly_2 * poly_831
    poly_2195 = poly_5 * poly_812
    poly_2196 = poly_2 * poly_893
    poly_2197 = poly_5 * poly_814
    poly_2198 = poly_5 * poly_815
    poly_2199 = poly_5 * poly_816
    poly_2200 = poly_2 * poly_835
    poly_2201 = poly_5 * poly_818
    poly_2202 = poly_5 * poly_819
    poly_2203 = poly_5 * poly_820
    poly_2204 = poly_5 * poly_821
    poly_2205 = poly_2 * poly_894
    poly_2206 = poly_5 * poly_823
    poly_2207 = poly_5 * poly_824
    poly_2208 = poly_5 * poly_825
    poly_2209 = poly_5 * poly_826
    poly_2210 = poly_5 * poly_827
    poly_2211 = poly_5 * poly_828
    poly_2212 = poly_2 * poly_895
    poly_2213 = poly_5 * poly_830
    poly_2214 = poly_2 * poly_897
    poly_2215 = poly_5 * poly_832
    poly_2216 = poly_5 * poly_833
    poly_2217 = poly_5 * poly_834
    poly_2218 = poly_2 * poly_898
    poly_2219 = poly_5 * poly_836
    poly_2220 = poly_5 * poly_837
    poly_2221 = poly_5 * poly_838
    poly_2222 = poly_5 * poly_839
    poly_2223 = poly_5 * poly_840
    poly_2224 = poly_6 * poly_272
    poly_2225 = poly_6 * poly_273
    poly_2226 = poly_6 * poly_318
    poly_2227 = poly_2 * poly_899
    poly_2228 = poly_2 * poly_843
    poly_2229 = poly_2 * poly_844
    poly_2230 = poly_2 * poly_845
    poly_2231 = poly_2 * poly_846
    poly_2232 = poly_3 * poly_899
    poly_2233 = poly_2 * poly_848
    poly_2234 = poly_25 * poly_318
    poly_2235 = poly_2 * poly_850
    poly_2236 = poly_97 * poly_99
    poly_2237 = poly_2 * poly_852
    poly_2238 = poly_10 * poly_281
    poly_2239 = poly_23 * poly_320
    poly_2240 = poly_2 * poly_855
    poly_2241 = poly_10 * poly_284
    poly_2242 = poly_1 * poly_901
    poly_2243 = poly_2 * poly_901
    poly_2244 = poly_10 * poly_320
    poly_2245 = poly_1 * poly_857 - poly_1922
    poly_2246 = poly_1 * poly_858 - poly_1925
    poly_2247 = poly_2 * poly_859
    poly_2248 = poly_3 * poly_860 - poly_1954
    poly_2249 = poly_3 * poly_861 - poly_1966
    poly_2250 = poly_1 * poly_862 - poly_1973
    poly_2251 = poly_1 * poly_863 - poly_1975
    poly_2252 = poly_2 * poly_864
    poly_2253 = poly_3 * poly_865 - poly_2007
    poly_2254 = poly_3 * poly_866 - poly_2017
    poly_2255 = poly_1 * poly_867 - poly_2030
    poly_2256 = poly_1 * poly_868 - poly_2031
    poly_2257 = poly_2 * poly_869
    poly_2258 = poly_3 * poly_870 - poly_2063
    poly_2259 = poly_15 * poly_294 - poly_1564
    poly_2260 = poly_16 * poly_295 - poly_1569
    poly_2261 = poly_3 * poly_873 - poly_2069
    poly_2262 = poly_18 * poly_297 - poly_1590
    poly_2263 = poly_1 * poly_875 - poly_2086
    poly_2264 = poly_1 * poly_876 - poly_2086
    poly_2265 = poly_2 * poly_877
    poly_2266 = poly_3 * poly_878 - poly_2104
    poly_2267 = poly_15 * poly_302 - poly_1644
    poly_2268 = poly_16 * poly_303 - poly_1644
    poly_2269 = poly_3 * poly_881 - poly_2104
    poly_2270 = poly_18 * poly_305 - poly_1644
    poly_2271 = poly_4 * poly_875 - poly_2107 - poly_2096 - poly_2092
    poly_2272 = poly_1 * poly_902 - poly_2271
    poly_2273 = poly_2 * poly_902
    poly_2274 = poly_4 * poly_878 - poly_2110 - poly_2099 - poly_2095
    poly_2275 = poly_15 * poly_321 - poly_2112
    poly_2276 = poly_16 * poly_321 - poly_2111
    poly_2277 = poly_3 * poly_902 - poly_2274
    poly_2278 = poly_18 * poly_321 - poly_2100
    poly_2279 = poly_5 * poly_899
    poly_2280 = poly_2 * poly_884
    poly_2281 = poly_5 * poly_901
    poly_2282 = poly_5 * poly_902
    poly_2283 = poly_5 * poly_883
    poly_2284 = poly_2 * poly_888
    poly_2285 = poly_5 * poly_885
    poly_2286 = poly_5 * poly_886
    poly_2287 = poly_5 * poly_887
    poly_2288 = poly_2 * poly_892
    poly_2289 = poly_5 * poly_889
    poly_2290 = poly_5 * poly_890
    poly_2291 = poly_5 * poly_891
    poly_2292 = poly_2 * poly_896
    poly_2293 = poly_5 * poly_893
    poly_2294 = poly_5 * poly_894
    poly_2295 = poly_5 * poly_895
    poly_2296 = poly_2 * poly_903
    poly_2297 = poly_5 * poly_897
    poly_2298 = poly_5 * poly_898
    poly_2299 = poly_1 * poly_899 - poly_2226
    poly_2300 = poly_2 * poly_900
    poly_2301 = poly_3 * poly_901 - poly_2244
    poly_2302 = poly_4 * poly_902 - poly_2278 - poly_2276 - poly_2275
    poly_2303 = poly_5 * poly_903

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
                      poly_901,    poly_902,    poly_903,    poly_904,    poly_905,
                      poly_906,    poly_907,    poly_908,    poly_909,    poly_910,
                      poly_911,    poly_912,    poly_913,    poly_914,    poly_915,
                      poly_916,    poly_917,    poly_918,    poly_919,    poly_920,
                      poly_921,    poly_922,    poly_923,    poly_924,    poly_925,
                      poly_926,    poly_927,    poly_928,    poly_929,    poly_930,
                      poly_931,    poly_932,    poly_933,    poly_934,    poly_935,
                      poly_936,    poly_937,    poly_938,    poly_939,    poly_940,
                      poly_941,    poly_942,    poly_943,    poly_944,    poly_945,
                      poly_946,    poly_947,    poly_948,    poly_949,    poly_950,
                      poly_951,    poly_952,    poly_953,    poly_954,    poly_955,
                      poly_956,    poly_957,    poly_958,    poly_959,    poly_960,
                      poly_961,    poly_962,    poly_963,    poly_964,    poly_965,
                      poly_966,    poly_967,    poly_968,    poly_969,    poly_970,
                      poly_971,    poly_972,    poly_973,    poly_974,    poly_975,
                      poly_976,    poly_977,    poly_978,    poly_979,    poly_980,
                      poly_981,    poly_982,    poly_983,    poly_984,    poly_985,
                      poly_986,    poly_987,    poly_988,    poly_989,    poly_990,
                      poly_991,    poly_992,    poly_993,    poly_994,    poly_995,
                      poly_996,    poly_997,    poly_998,    poly_999,    poly_1000,
                      poly_1001,    poly_1002,    poly_1003,    poly_1004,    poly_1005,
                      poly_1006,    poly_1007,    poly_1008,    poly_1009,    poly_1010,
                      poly_1011,    poly_1012,    poly_1013,    poly_1014,    poly_1015,
                      poly_1016,    poly_1017,    poly_1018,    poly_1019,    poly_1020,
                      poly_1021,    poly_1022,    poly_1023,    poly_1024,    poly_1025,
                      poly_1026,    poly_1027,    poly_1028,    poly_1029,    poly_1030,
                      poly_1031,    poly_1032,    poly_1033,    poly_1034,    poly_1035,
                      poly_1036,    poly_1037,    poly_1038,    poly_1039,    poly_1040,
                      poly_1041,    poly_1042,    poly_1043,    poly_1044,    poly_1045,
                      poly_1046,    poly_1047,    poly_1048,    poly_1049,    poly_1050,
                      poly_1051,    poly_1052,    poly_1053,    poly_1054,    poly_1055,
                      poly_1056,    poly_1057,    poly_1058,    poly_1059,    poly_1060,
                      poly_1061,    poly_1062,    poly_1063,    poly_1064,    poly_1065,
                      poly_1066,    poly_1067,    poly_1068,    poly_1069,    poly_1070,
                      poly_1071,    poly_1072,    poly_1073,    poly_1074,    poly_1075,
                      poly_1076,    poly_1077,    poly_1078,    poly_1079,    poly_1080,
                      poly_1081,    poly_1082,    poly_1083,    poly_1084,    poly_1085,
                      poly_1086,    poly_1087,    poly_1088,    poly_1089,    poly_1090,
                      poly_1091,    poly_1092,    poly_1093,    poly_1094,    poly_1095,
                      poly_1096,    poly_1097,    poly_1098,    poly_1099,    poly_1100,
                      poly_1101,    poly_1102,    poly_1103,    poly_1104,    poly_1105,
                      poly_1106,    poly_1107,    poly_1108,    poly_1109,    poly_1110,
                      poly_1111,    poly_1112,    poly_1113,    poly_1114,    poly_1115,
                      poly_1116,    poly_1117,    poly_1118,    poly_1119,    poly_1120,
                      poly_1121,    poly_1122,    poly_1123,    poly_1124,    poly_1125,
                      poly_1126,    poly_1127,    poly_1128,    poly_1129,    poly_1130,
                      poly_1131,    poly_1132,    poly_1133,    poly_1134,    poly_1135,
                      poly_1136,    poly_1137,    poly_1138,    poly_1139,    poly_1140,
                      poly_1141,    poly_1142,    poly_1143,    poly_1144,    poly_1145,
                      poly_1146,    poly_1147,    poly_1148,    poly_1149,    poly_1150,
                      poly_1151,    poly_1152,    poly_1153,    poly_1154,    poly_1155,
                      poly_1156,    poly_1157,    poly_1158,    poly_1159,    poly_1160,
                      poly_1161,    poly_1162,    poly_1163,    poly_1164,    poly_1165,
                      poly_1166,    poly_1167,    poly_1168,    poly_1169,    poly_1170,
                      poly_1171,    poly_1172,    poly_1173,    poly_1174,    poly_1175,
                      poly_1176,    poly_1177,    poly_1178,    poly_1179,    poly_1180,
                      poly_1181,    poly_1182,    poly_1183,    poly_1184,    poly_1185,
                      poly_1186,    poly_1187,    poly_1188,    poly_1189,    poly_1190,
                      poly_1191,    poly_1192,    poly_1193,    poly_1194,    poly_1195,
                      poly_1196,    poly_1197,    poly_1198,    poly_1199,    poly_1200,
                      poly_1201,    poly_1202,    poly_1203,    poly_1204,    poly_1205,
                      poly_1206,    poly_1207,    poly_1208,    poly_1209,    poly_1210,
                      poly_1211,    poly_1212,    poly_1213,    poly_1214,    poly_1215,
                      poly_1216,    poly_1217,    poly_1218,    poly_1219,    poly_1220,
                      poly_1221,    poly_1222,    poly_1223,    poly_1224,    poly_1225,
                      poly_1226,    poly_1227,    poly_1228,    poly_1229,    poly_1230,
                      poly_1231,    poly_1232,    poly_1233,    poly_1234,    poly_1235,
                      poly_1236,    poly_1237,    poly_1238,    poly_1239,    poly_1240,
                      poly_1241,    poly_1242,    poly_1243,    poly_1244,    poly_1245,
                      poly_1246,    poly_1247,    poly_1248,    poly_1249,    poly_1250,
                      poly_1251,    poly_1252,    poly_1253,    poly_1254,    poly_1255,
                      poly_1256,    poly_1257,    poly_1258,    poly_1259,    poly_1260,
                      poly_1261,    poly_1262,    poly_1263,    poly_1264,    poly_1265,
                      poly_1266,    poly_1267,    poly_1268,    poly_1269,    poly_1270,
                      poly_1271,    poly_1272,    poly_1273,    poly_1274,    poly_1275,
                      poly_1276,    poly_1277,    poly_1278,    poly_1279,    poly_1280,
                      poly_1281,    poly_1282,    poly_1283,    poly_1284,    poly_1285,
                      poly_1286,    poly_1287,    poly_1288,    poly_1289,    poly_1290,
                      poly_1291,    poly_1292,    poly_1293,    poly_1294,    poly_1295,
                      poly_1296,    poly_1297,    poly_1298,    poly_1299,    poly_1300,
                      poly_1301,    poly_1302,    poly_1303,    poly_1304,    poly_1305,
                      poly_1306,    poly_1307,    poly_1308,    poly_1309,    poly_1310,
                      poly_1311,    poly_1312,    poly_1313,    poly_1314,    poly_1315,
                      poly_1316,    poly_1317,    poly_1318,    poly_1319,    poly_1320,
                      poly_1321,    poly_1322,    poly_1323,    poly_1324,    poly_1325,
                      poly_1326,    poly_1327,    poly_1328,    poly_1329,    poly_1330,
                      poly_1331,    poly_1332,    poly_1333,    poly_1334,    poly_1335,
                      poly_1336,    poly_1337,    poly_1338,    poly_1339,    poly_1340,
                      poly_1341,    poly_1342,    poly_1343,    poly_1344,    poly_1345,
                      poly_1346,    poly_1347,    poly_1348,    poly_1349,    poly_1350,
                      poly_1351,    poly_1352,    poly_1353,    poly_1354,    poly_1355,
                      poly_1356,    poly_1357,    poly_1358,    poly_1359,    poly_1360,
                      poly_1361,    poly_1362,    poly_1363,    poly_1364,    poly_1365,
                      poly_1366,    poly_1367,    poly_1368,    poly_1369,    poly_1370,
                      poly_1371,    poly_1372,    poly_1373,    poly_1374,    poly_1375,
                      poly_1376,    poly_1377,    poly_1378,    poly_1379,    poly_1380,
                      poly_1381,    poly_1382,    poly_1383,    poly_1384,    poly_1385,
                      poly_1386,    poly_1387,    poly_1388,    poly_1389,    poly_1390,
                      poly_1391,    poly_1392,    poly_1393,    poly_1394,    poly_1395,
                      poly_1396,    poly_1397,    poly_1398,    poly_1399,    poly_1400,
                      poly_1401,    poly_1402,    poly_1403,    poly_1404,    poly_1405,
                      poly_1406,    poly_1407,    poly_1408,    poly_1409,    poly_1410,
                      poly_1411,    poly_1412,    poly_1413,    poly_1414,    poly_1415,
                      poly_1416,    poly_1417,    poly_1418,    poly_1419,    poly_1420,
                      poly_1421,    poly_1422,    poly_1423,    poly_1424,    poly_1425,
                      poly_1426,    poly_1427,    poly_1428,    poly_1429,    poly_1430,
                      poly_1431,    poly_1432,    poly_1433,    poly_1434,    poly_1435,
                      poly_1436,    poly_1437,    poly_1438,    poly_1439,    poly_1440,
                      poly_1441,    poly_1442,    poly_1443,    poly_1444,    poly_1445,
                      poly_1446,    poly_1447,    poly_1448,    poly_1449,    poly_1450,
                      poly_1451,    poly_1452,    poly_1453,    poly_1454,    poly_1455,
                      poly_1456,    poly_1457,    poly_1458,    poly_1459,    poly_1460,
                      poly_1461,    poly_1462,    poly_1463,    poly_1464,    poly_1465,
                      poly_1466,    poly_1467,    poly_1468,    poly_1469,    poly_1470,
                      poly_1471,    poly_1472,    poly_1473,    poly_1474,    poly_1475,
                      poly_1476,    poly_1477,    poly_1478,    poly_1479,    poly_1480,
                      poly_1481,    poly_1482,    poly_1483,    poly_1484,    poly_1485,
                      poly_1486,    poly_1487,    poly_1488,    poly_1489,    poly_1490,
                      poly_1491,    poly_1492,    poly_1493,    poly_1494,    poly_1495,
                      poly_1496,    poly_1497,    poly_1498,    poly_1499,    poly_1500,
                      poly_1501,    poly_1502,    poly_1503,    poly_1504,    poly_1505,
                      poly_1506,    poly_1507,    poly_1508,    poly_1509,    poly_1510,
                      poly_1511,    poly_1512,    poly_1513,    poly_1514,    poly_1515,
                      poly_1516,    poly_1517,    poly_1518,    poly_1519,    poly_1520,
                      poly_1521,    poly_1522,    poly_1523,    poly_1524,    poly_1525,
                      poly_1526,    poly_1527,    poly_1528,    poly_1529,    poly_1530,
                      poly_1531,    poly_1532,    poly_1533,    poly_1534,    poly_1535,
                      poly_1536,    poly_1537,    poly_1538,    poly_1539,    poly_1540,
                      poly_1541,    poly_1542,    poly_1543,    poly_1544,    poly_1545,
                      poly_1546,    poly_1547,    poly_1548,    poly_1549,    poly_1550,
                      poly_1551,    poly_1552,    poly_1553,    poly_1554,    poly_1555,
                      poly_1556,    poly_1557,    poly_1558,    poly_1559,    poly_1560,
                      poly_1561,    poly_1562,    poly_1563,    poly_1564,    poly_1565,
                      poly_1566,    poly_1567,    poly_1568,    poly_1569,    poly_1570,
                      poly_1571,    poly_1572,    poly_1573,    poly_1574,    poly_1575,
                      poly_1576,    poly_1577,    poly_1578,    poly_1579,    poly_1580,
                      poly_1581,    poly_1582,    poly_1583,    poly_1584,    poly_1585,
                      poly_1586,    poly_1587,    poly_1588,    poly_1589,    poly_1590,
                      poly_1591,    poly_1592,    poly_1593,    poly_1594,    poly_1595,
                      poly_1596,    poly_1597,    poly_1598,    poly_1599,    poly_1600,
                      poly_1601,    poly_1602,    poly_1603,    poly_1604,    poly_1605,
                      poly_1606,    poly_1607,    poly_1608,    poly_1609,    poly_1610,
                      poly_1611,    poly_1612,    poly_1613,    poly_1614,    poly_1615,
                      poly_1616,    poly_1617,    poly_1618,    poly_1619,    poly_1620,
                      poly_1621,    poly_1622,    poly_1623,    poly_1624,    poly_1625,
                      poly_1626,    poly_1627,    poly_1628,    poly_1629,    poly_1630,
                      poly_1631,    poly_1632,    poly_1633,    poly_1634,    poly_1635,
                      poly_1636,    poly_1637,    poly_1638,    poly_1639,    poly_1640,
                      poly_1641,    poly_1642,    poly_1643,    poly_1644,    poly_1645,
                      poly_1646,    poly_1647,    poly_1648,    poly_1649,    poly_1650,
                      poly_1651,    poly_1652,    poly_1653,    poly_1654,    poly_1655,
                      poly_1656,    poly_1657,    poly_1658,    poly_1659,    poly_1660,
                      poly_1661,    poly_1662,    poly_1663,    poly_1664,    poly_1665,
                      poly_1666,    poly_1667,    poly_1668,    poly_1669,    poly_1670,
                      poly_1671,    poly_1672,    poly_1673,    poly_1674,    poly_1675,
                      poly_1676,    poly_1677,    poly_1678,    poly_1679,    poly_1680,
                      poly_1681,    poly_1682,    poly_1683,    poly_1684,    poly_1685,
                      poly_1686,    poly_1687,    poly_1688,    poly_1689,    poly_1690,
                      poly_1691,    poly_1692,    poly_1693,    poly_1694,    poly_1695,
                      poly_1696,    poly_1697,    poly_1698,    poly_1699,    poly_1700,
                      poly_1701,    poly_1702,    poly_1703,    poly_1704,    poly_1705,
                      poly_1706,    poly_1707,    poly_1708,    poly_1709,    poly_1710,
                      poly_1711,    poly_1712,    poly_1713,    poly_1714,    poly_1715,
                      poly_1716,    poly_1717,    poly_1718,    poly_1719,    poly_1720,
                      poly_1721,    poly_1722,    poly_1723,    poly_1724,    poly_1725,
                      poly_1726,    poly_1727,    poly_1728,    poly_1729,    poly_1730,
                      poly_1731,    poly_1732,    poly_1733,    poly_1734,    poly_1735,
                      poly_1736,    poly_1737,    poly_1738,    poly_1739,    poly_1740,
                      poly_1741,    poly_1742,    poly_1743,    poly_1744,    poly_1745,
                      poly_1746,    poly_1747,    poly_1748,    poly_1749,    poly_1750,
                      poly_1751,    poly_1752,    poly_1753,    poly_1754,    poly_1755,
                      poly_1756,    poly_1757,    poly_1758,    poly_1759,    poly_1760,
                      poly_1761,    poly_1762,    poly_1763,    poly_1764,    poly_1765,
                      poly_1766,    poly_1767,    poly_1768,    poly_1769,    poly_1770,
                      poly_1771,    poly_1772,    poly_1773,    poly_1774,    poly_1775,
                      poly_1776,    poly_1777,    poly_1778,    poly_1779,    poly_1780,
                      poly_1781,    poly_1782,    poly_1783,    poly_1784,    poly_1785,
                      poly_1786,    poly_1787,    poly_1788,    poly_1789,    poly_1790,
                      poly_1791,    poly_1792,    poly_1793,    poly_1794,    poly_1795,
                      poly_1796,    poly_1797,    poly_1798,    poly_1799,    poly_1800,
                      poly_1801,    poly_1802,    poly_1803,    poly_1804,    poly_1805,
                      poly_1806,    poly_1807,    poly_1808,    poly_1809,    poly_1810,
                      poly_1811,    poly_1812,    poly_1813,    poly_1814,    poly_1815,
                      poly_1816,    poly_1817,    poly_1818,    poly_1819,    poly_1820,
                      poly_1821,    poly_1822,    poly_1823,    poly_1824,    poly_1825,
                      poly_1826,    poly_1827,    poly_1828,    poly_1829,    poly_1830,
                      poly_1831,    poly_1832,    poly_1833,    poly_1834,    poly_1835,
                      poly_1836,    poly_1837,    poly_1838,    poly_1839,    poly_1840,
                      poly_1841,    poly_1842,    poly_1843,    poly_1844,    poly_1845,
                      poly_1846,    poly_1847,    poly_1848,    poly_1849,    poly_1850,
                      poly_1851,    poly_1852,    poly_1853,    poly_1854,    poly_1855,
                      poly_1856,    poly_1857,    poly_1858,    poly_1859,    poly_1860,
                      poly_1861,    poly_1862,    poly_1863,    poly_1864,    poly_1865,
                      poly_1866,    poly_1867,    poly_1868,    poly_1869,    poly_1870,
                      poly_1871,    poly_1872,    poly_1873,    poly_1874,    poly_1875,
                      poly_1876,    poly_1877,    poly_1878,    poly_1879,    poly_1880,
                      poly_1881,    poly_1882,    poly_1883,    poly_1884,    poly_1885,
                      poly_1886,    poly_1887,    poly_1888,    poly_1889,    poly_1890,
                      poly_1891,    poly_1892,    poly_1893,    poly_1894,    poly_1895,
                      poly_1896,    poly_1897,    poly_1898,    poly_1899,    poly_1900,
                      poly_1901,    poly_1902,    poly_1903,    poly_1904,    poly_1905,
                      poly_1906,    poly_1907,    poly_1908,    poly_1909,    poly_1910,
                      poly_1911,    poly_1912,    poly_1913,    poly_1914,    poly_1915,
                      poly_1916,    poly_1917,    poly_1918,    poly_1919,    poly_1920,
                      poly_1921,    poly_1922,    poly_1923,    poly_1924,    poly_1925,
                      poly_1926,    poly_1927,    poly_1928,    poly_1929,    poly_1930,
                      poly_1931,    poly_1932,    poly_1933,    poly_1934,    poly_1935,
                      poly_1936,    poly_1937,    poly_1938,    poly_1939,    poly_1940,
                      poly_1941,    poly_1942,    poly_1943,    poly_1944,    poly_1945,
                      poly_1946,    poly_1947,    poly_1948,    poly_1949,    poly_1950,
                      poly_1951,    poly_1952,    poly_1953,    poly_1954,    poly_1955,
                      poly_1956,    poly_1957,    poly_1958,    poly_1959,    poly_1960,
                      poly_1961,    poly_1962,    poly_1963,    poly_1964,    poly_1965,
                      poly_1966,    poly_1967,    poly_1968,    poly_1969,    poly_1970,
                      poly_1971,    poly_1972,    poly_1973,    poly_1974,    poly_1975,
                      poly_1976,    poly_1977,    poly_1978,    poly_1979,    poly_1980,
                      poly_1981,    poly_1982,    poly_1983,    poly_1984,    poly_1985,
                      poly_1986,    poly_1987,    poly_1988,    poly_1989,    poly_1990,
                      poly_1991,    poly_1992,    poly_1993,    poly_1994,    poly_1995,
                      poly_1996,    poly_1997,    poly_1998,    poly_1999,    poly_2000,
                      poly_2001,    poly_2002,    poly_2003,    poly_2004,    poly_2005,
                      poly_2006,    poly_2007,    poly_2008,    poly_2009,    poly_2010,
                      poly_2011,    poly_2012,    poly_2013,    poly_2014,    poly_2015,
                      poly_2016,    poly_2017,    poly_2018,    poly_2019,    poly_2020,
                      poly_2021,    poly_2022,    poly_2023,    poly_2024,    poly_2025,
                      poly_2026,    poly_2027,    poly_2028,    poly_2029,    poly_2030,
                      poly_2031,    poly_2032,    poly_2033,    poly_2034,    poly_2035,
                      poly_2036,    poly_2037,    poly_2038,    poly_2039,    poly_2040,
                      poly_2041,    poly_2042,    poly_2043,    poly_2044,    poly_2045,
                      poly_2046,    poly_2047,    poly_2048,    poly_2049,    poly_2050,
                      poly_2051,    poly_2052,    poly_2053,    poly_2054,    poly_2055,
                      poly_2056,    poly_2057,    poly_2058,    poly_2059,    poly_2060,
                      poly_2061,    poly_2062,    poly_2063,    poly_2064,    poly_2065,
                      poly_2066,    poly_2067,    poly_2068,    poly_2069,    poly_2070,
                      poly_2071,    poly_2072,    poly_2073,    poly_2074,    poly_2075,
                      poly_2076,    poly_2077,    poly_2078,    poly_2079,    poly_2080,
                      poly_2081,    poly_2082,    poly_2083,    poly_2084,    poly_2085,
                      poly_2086,    poly_2087,    poly_2088,    poly_2089,    poly_2090,
                      poly_2091,    poly_2092,    poly_2093,    poly_2094,    poly_2095,
                      poly_2096,    poly_2097,    poly_2098,    poly_2099,    poly_2100,
                      poly_2101,    poly_2102,    poly_2103,    poly_2104,    poly_2105,
                      poly_2106,    poly_2107,    poly_2108,    poly_2109,    poly_2110,
                      poly_2111,    poly_2112,    poly_2113,    poly_2114,    poly_2115,
                      poly_2116,    poly_2117,    poly_2118,    poly_2119,    poly_2120,
                      poly_2121,    poly_2122,    poly_2123,    poly_2124,    poly_2125,
                      poly_2126,    poly_2127,    poly_2128,    poly_2129,    poly_2130,
                      poly_2131,    poly_2132,    poly_2133,    poly_2134,    poly_2135,
                      poly_2136,    poly_2137,    poly_2138,    poly_2139,    poly_2140,
                      poly_2141,    poly_2142,    poly_2143,    poly_2144,    poly_2145,
                      poly_2146,    poly_2147,    poly_2148,    poly_2149,    poly_2150,
                      poly_2151,    poly_2152,    poly_2153,    poly_2154,    poly_2155,
                      poly_2156,    poly_2157,    poly_2158,    poly_2159,    poly_2160,
                      poly_2161,    poly_2162,    poly_2163,    poly_2164,    poly_2165,
                      poly_2166,    poly_2167,    poly_2168,    poly_2169,    poly_2170,
                      poly_2171,    poly_2172,    poly_2173,    poly_2174,    poly_2175,
                      poly_2176,    poly_2177,    poly_2178,    poly_2179,    poly_2180,
                      poly_2181,    poly_2182,    poly_2183,    poly_2184,    poly_2185,
                      poly_2186,    poly_2187,    poly_2188,    poly_2189,    poly_2190,
                      poly_2191,    poly_2192,    poly_2193,    poly_2194,    poly_2195,
                      poly_2196,    poly_2197,    poly_2198,    poly_2199,    poly_2200,
                      poly_2201,    poly_2202,    poly_2203,    poly_2204,    poly_2205,
                      poly_2206,    poly_2207,    poly_2208,    poly_2209,    poly_2210,
                      poly_2211,    poly_2212,    poly_2213,    poly_2214,    poly_2215,
                      poly_2216,    poly_2217,    poly_2218,    poly_2219,    poly_2220,
                      poly_2221,    poly_2222,    poly_2223,    poly_2224,    poly_2225,
                      poly_2226,    poly_2227,    poly_2228,    poly_2229,    poly_2230,
                      poly_2231,    poly_2232,    poly_2233,    poly_2234,    poly_2235,
                      poly_2236,    poly_2237,    poly_2238,    poly_2239,    poly_2240,
                      poly_2241,    poly_2242,    poly_2243,    poly_2244,    poly_2245,
                      poly_2246,    poly_2247,    poly_2248,    poly_2249,    poly_2250,
                      poly_2251,    poly_2252,    poly_2253,    poly_2254,    poly_2255,
                      poly_2256,    poly_2257,    poly_2258,    poly_2259,    poly_2260,
                      poly_2261,    poly_2262,    poly_2263,    poly_2264,    poly_2265,
                      poly_2266,    poly_2267,    poly_2268,    poly_2269,    poly_2270,
                      poly_2271,    poly_2272,    poly_2273,    poly_2274,    poly_2275,
                      poly_2276,    poly_2277,    poly_2278,    poly_2279,    poly_2280,
                      poly_2281,    poly_2282,    poly_2283,    poly_2284,    poly_2285,
                      poly_2286,    poly_2287,    poly_2288,    poly_2289,    poly_2290,
                      poly_2291,    poly_2292,    poly_2293,    poly_2294,    poly_2295,
                      poly_2296,    poly_2297,    poly_2298,    poly_2299,    poly_2300,
                      poly_2301,    poly_2302,    poly_2303,])

    return poly
