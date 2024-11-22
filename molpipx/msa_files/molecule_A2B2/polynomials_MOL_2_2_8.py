import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2.monomials_MOL_2_2_8 import f_monomials as f_monos

# File created from ./MOL_2_2_8.POLY

N_POLYS = 882

# Total number of monomials = 882


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(882)

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
    poly_291 = poly_1 * poly_153
    poly_292 = poly_1 * poly_160
    poly_293 = poly_1 * poly_162
    poly_294 = poly_1 * poly_154
    poly_295 = poly_1 * poly_155
    poly_296 = poly_1 * poly_168
    poly_297 = poly_1 * poly_170
    poly_298 = poly_1 * poly_172
    poly_299 = poly_1 * poly_176
    poly_300 = poly_1 * poly_156
    poly_301 = poly_1 * poly_157
    poly_302 = poly_1 * poly_158
    poly_303 = poly_1 * poly_185
    poly_304 = poly_1 * poly_159
    poly_305 = poly_3 * poly_168
    poly_306 = poly_1 * poly_186
    poly_307 = poly_3 * poly_170
    poly_308 = poly_1 * poly_188
    poly_309 = poly_3 * poly_172
    poly_310 = poly_1 * poly_191
    poly_311 = poly_1 * poly_193
    poly_312 = poly_1 * poly_194
    poly_313 = poly_3 * poly_176
    poly_314 = poly_1 * poly_161
    poly_315 = poly_1 * poly_200
    poly_316 = poly_1 * poly_202
    poly_317 = poly_1 * poly_203
    poly_318 = poly_3 * poly_160
    poly_319 = poly_1 * poly_207
    poly_320 = poly_3 * poly_162
    poly_321 = poly_1 * poly_163
    poly_322 = poly_1 * poly_164
    poly_323 = poly_1 * poly_165
    poly_324 = poly_1 * poly_166
    poly_325 = poly_1 * poly_167
    poly_326 = poly_1 * poly_169
    poly_327 = poly_1 * poly_171
    poly_328 = poly_1 * poly_217
    poly_329 = poly_34 * poly_16
    poly_330 = poly_1 * poly_173
    poly_331 = poly_1 * poly_221
    poly_332 = poly_1 * poly_223
    poly_333 = poly_1 * poly_174
    poly_334 = poly_1 * poly_225
    poly_335 = poly_1 * poly_175
    poly_336 = poly_34 * poly_23
    poly_337 = poly_1 * poly_226
    poly_338 = poly_34 * poly_24
    poly_339 = poly_1 * poly_228
    poly_340 = poly_1 * poly_229
    poly_341 = poly_34 * poly_25
    poly_342 = poly_1 * poly_232
    poly_343 = poly_1 * poly_234
    poly_344 = poly_1 * poly_235
    poly_345 = poly_34 * poly_31
    poly_346 = poly_1 * poly_177
    poly_347 = poly_1 * poly_178
    poly_348 = poly_1 * poly_179
    poly_349 = poly_1 * poly_180
    poly_350 = poly_1 * poly_181
    poly_351 = poly_1 * poly_182
    poly_352 = poly_1 * poly_183
    poly_353 = poly_1 * poly_184
    poly_354 = poly_1 * poly_187
    poly_355 = poly_3 * poly_217
    poly_356 = poly_1 * poly_189
    poly_357 = poly_1 * poly_239
    poly_358 = poly_1 * poly_190
    poly_359 = poly_3 * poly_221
    poly_360 = poly_1 * poly_240
    poly_361 = poly_3 * poly_223
    poly_362 = poly_1 * poly_192
    poly_363 = poly_3 * poly_225
    poly_364 = poly_3 * poly_226
    poly_365 = poly_1 * poly_241
    poly_366 = poly_3 * poly_228
    poly_367 = poly_3 * poly_229
    poly_368 = poly_1 * poly_243
    poly_369 = poly_1 * poly_244
    poly_370 = poly_3 * poly_232
    poly_371 = poly_1 * poly_245
    poly_372 = poly_3 * poly_234
    poly_373 = poly_3 * poly_235
    poly_374 = poly_1 * poly_195
    poly_375 = poly_1 * poly_196
    poly_376 = poly_1 * poly_197
    poly_377 = poly_1 * poly_198
    poly_378 = poly_1 * poly_248
    poly_379 = poly_1 * poly_199
    poly_380 = poly_1 * poly_249
    poly_381 = poly_1 * poly_201
    poly_382 = poly_3 * poly_185
    poly_383 = poly_3 * poly_186
    poly_384 = poly_1 * poly_250
    poly_385 = poly_3 * poly_188
    poly_386 = poly_1 * poly_252
    poly_387 = poly_1 * poly_253
    poly_388 = poly_3 * poly_191
    poly_389 = poly_1 * poly_254
    poly_390 = poly_3 * poly_193
    poly_391 = poly_3 * poly_194
    poly_392 = poly_1 * poly_204
    poly_393 = poly_1 * poly_205
    poly_394 = poly_1 * poly_206
    poly_395 = poly_1 * poly_257
    poly_396 = poly_1 * poly_258
    poly_397 = poly_3 * poly_200
    poly_398 = poly_1 * poly_259
    poly_399 = poly_3 * poly_202
    poly_400 = poly_3 * poly_203
    poly_401 = poly_1 * poly_261
    poly_402 = poly_1 * poly_262
    poly_403 = poly_1 * poly_263
    poly_404 = poly_3 * poly_207
    poly_405 = poly_1 * poly_208
    poly_406 = poly_1 * poly_209
    poly_407 = poly_1 * poly_210
    poly_408 = poly_1 * poly_211
    poly_409 = poly_1 * poly_212
    poly_410 = poly_1 * poly_213
    poly_411 = poly_1 * poly_214
    poly_412 = poly_1 * poly_215
    poly_413 = poly_1 * poly_216
    poly_414 = poly_1 * poly_218
    poly_415 = poly_1 * poly_219
    poly_416 = poly_1 * poly_267
    poly_417 = poly_1 * poly_220
    poly_418 = poly_1 * poly_222
    poly_419 = poly_5 * poly_105 - poly_338
    poly_420 = poly_1 * poly_268
    poly_421 = poly_1 * poly_224
    poly_422 = poly_5 * poly_104 - poly_329
    poly_423 = poly_6 * poly_105 - poly_329
    poly_424 = poly_1 * poly_227
    poly_425 = poly_5 * poly_107 - poly_341
    poly_426 = poly_5 * poly_113 - poly_345
    poly_427 = poly_1 * poly_269
    poly_428 = poly_7 * poly_107 - poly_329
    poly_429 = poly_1 * poly_230
    poly_430 = poly_1 * poly_271
    poly_431 = poly_1 * poly_231
    poly_432 = poly_5 * poly_110 - poly_338
    poly_433 = poly_1 * poly_272
    poly_434 = poly_5 * poly_136 - poly_423
    poly_435 = poly_1 * poly_233
    poly_436 = poly_5 * poly_112 - poly_341
    poly_437 = poly_6 * poly_113 - poly_341
    poly_438 = poly_1 * poly_273
    poly_439 = poly_5 * poly_137 - poly_428
    poly_440 = poly_6 * poly_137 - poly_428
    poly_441 = poly_1 * poly_275
    poly_442 = poly_1 * poly_276
    poly_443 = poly_5 * poly_140 - poly_437
    poly_444 = poly_1 * poly_277
    poly_445 = poly_5 * poly_141 - poly_440
    poly_446 = poly_6 * poly_141 - poly_439
    poly_447 = poly_1 * poly_236
    poly_448 = poly_1 * poly_237
    poly_449 = poly_1 * poly_238
    poly_450 = poly_3 * poly_267
    poly_451 = poly_3 * poly_268
    poly_452 = poly_3 * poly_269
    poly_453 = poly_1 * poly_242
    poly_454 = poly_3 * poly_271
    poly_455 = poly_3 * poly_272
    poly_456 = poly_3 * poly_273
    poly_457 = poly_1 * poly_279
    poly_458 = poly_3 * poly_275
    poly_459 = poly_3 * poly_276
    poly_460 = poly_3 * poly_277
    poly_461 = poly_1 * poly_246
    poly_462 = poly_1 * poly_247
    poly_463 = poly_1 * poly_251
    poly_464 = poly_3 * poly_239
    poly_465 = poly_3 * poly_240
    poly_466 = poly_3 * poly_241
    poly_467 = poly_1 * poly_281
    poly_468 = poly_3 * poly_243
    poly_469 = poly_3 * poly_244
    poly_470 = poly_3 * poly_245
    poly_471 = poly_1 * poly_255
    poly_472 = poly_1 * poly_256
    poly_473 = poly_3 * poly_248
    poly_474 = poly_3 * poly_249
    poly_475 = poly_3 * poly_250
    poly_476 = poly_1 * poly_283
    poly_477 = poly_3 * poly_252
    poly_478 = poly_3 * poly_253
    poly_479 = poly_3 * poly_254
    poly_480 = poly_1 * poly_260
    poly_481 = poly_1 * poly_285
    poly_482 = poly_3 * poly_257
    poly_483 = poly_3 * poly_258
    poly_484 = poly_3 * poly_259
    poly_485 = poly_1 * poly_287
    poly_486 = poly_3 * poly_261
    poly_487 = poly_3 * poly_262
    poly_488 = poly_3 * poly_263
    poly_489 = poly_1 * poly_264
    poly_490 = poly_1 * poly_265
    poly_491 = poly_1 * poly_266
    poly_492 = poly_1 * poly_270
    poly_493 = poly_2 * poly_267 - poly_422
    poly_494 = poly_2 * poly_268 - poly_423
    poly_495 = poly_2 * poly_269 - poly_428
    poly_496 = poly_1 * poly_274
    poly_497 = poly_5 * poly_139 - poly_345
    poly_498 = poly_6 * poly_140 - poly_345
    poly_499 = poly_7 * poly_141 - poly_345
    poly_500 = poly_1 * poly_289
    poly_501 = poly_5 * poly_151 - poly_446
    poly_502 = poly_6 * poly_151 - poly_445
    poly_503 = poly_7 * poly_151 - poly_443
    poly_504 = poly_1 * poly_278
    poly_505 = poly_3 * poly_289
    poly_506 = poly_1 * poly_280
    poly_507 = poly_3 * poly_279
    poly_508 = poly_1 * poly_282
    poly_509 = poly_3 * poly_281
    poly_510 = poly_1 * poly_284
    poly_511 = poly_3 * poly_283
    poly_512 = poly_1 * poly_286
    poly_513 = poly_3 * poly_285
    poly_514 = poly_1 * poly_290
    poly_515 = poly_3 * poly_287
    poly_516 = poly_1 * poly_288
    poly_517 = poly_2 * poly_289 - poly_503 - poly_502 - poly_501
    poly_518 = poly_3 * poly_290
    poly_519 = poly_1 * poly_291
    poly_520 = poly_1 * poly_292
    poly_521 = poly_1 * poly_305
    poly_522 = poly_1 * poly_307
    poly_523 = poly_1 * poly_309
    poly_524 = poly_1 * poly_313
    poly_525 = poly_1 * poly_293
    poly_526 = poly_1 * poly_318
    poly_527 = poly_1 * poly_320
    poly_528 = poly_1 * poly_294
    poly_529 = poly_1 * poly_295
    poly_530 = poly_1 * poly_296
    poly_531 = poly_1 * poly_297
    poly_532 = poly_1 * poly_298
    poly_533 = poly_1 * poly_329
    poly_534 = poly_1 * poly_299
    poly_535 = poly_1 * poly_336
    poly_536 = poly_1 * poly_338
    poly_537 = poly_1 * poly_341
    poly_538 = poly_1 * poly_345
    poly_539 = poly_1 * poly_300
    poly_540 = poly_1 * poly_301
    poly_541 = poly_1 * poly_302
    poly_542 = poly_1 * poly_303
    poly_543 = poly_1 * poly_304
    poly_544 = poly_1 * poly_306
    poly_545 = poly_1 * poly_308
    poly_546 = poly_1 * poly_355
    poly_547 = poly_3 * poly_329
    poly_548 = poly_1 * poly_310
    poly_549 = poly_1 * poly_359
    poly_550 = poly_1 * poly_361
    poly_551 = poly_1 * poly_311
    poly_552 = poly_1 * poly_363
    poly_553 = poly_1 * poly_312
    poly_554 = poly_3 * poly_336
    poly_555 = poly_1 * poly_364
    poly_556 = poly_3 * poly_338
    poly_557 = poly_1 * poly_366
    poly_558 = poly_1 * poly_367
    poly_559 = poly_3 * poly_341
    poly_560 = poly_1 * poly_370
    poly_561 = poly_1 * poly_372
    poly_562 = poly_1 * poly_373
    poly_563 = poly_3 * poly_345
    poly_564 = poly_1 * poly_314
    poly_565 = poly_1 * poly_315
    poly_566 = poly_1 * poly_316
    poly_567 = poly_1 * poly_382
    poly_568 = poly_1 * poly_317
    poly_569 = poly_3 * poly_305
    poly_570 = poly_1 * poly_383
    poly_571 = poly_3 * poly_307
    poly_572 = poly_1 * poly_385
    poly_573 = poly_3 * poly_309
    poly_574 = poly_1 * poly_388
    poly_575 = poly_1 * poly_390
    poly_576 = poly_1 * poly_391
    poly_577 = poly_3 * poly_313
    poly_578 = poly_1 * poly_319
    poly_579 = poly_1 * poly_397
    poly_580 = poly_1 * poly_399
    poly_581 = poly_1 * poly_400
    poly_582 = poly_3 * poly_318
    poly_583 = poly_1 * poly_404
    poly_584 = poly_3 * poly_320
    poly_585 = poly_1 * poly_321
    poly_586 = poly_1 * poly_322
    poly_587 = poly_1 * poly_323
    poly_588 = poly_1 * poly_324
    poly_589 = poly_1 * poly_325
    poly_590 = poly_1 * poly_326
    poly_591 = poly_1 * poly_327
    poly_592 = poly_1 * poly_328
    poly_593 = poly_34 * poly_34
    poly_594 = poly_1 * poly_330
    poly_595 = poly_1 * poly_331
    poly_596 = poly_1 * poly_332
    poly_597 = poly_1 * poly_419
    poly_598 = poly_1 * poly_333
    poly_599 = poly_1 * poly_334
    poly_600 = poly_1 * poly_422
    poly_601 = poly_1 * poly_335
    poly_602 = poly_34 * poly_59
    poly_603 = poly_1 * poly_337
    poly_604 = poly_34 * poly_44
    poly_605 = poly_1 * poly_423
    poly_606 = poly_34 * poly_60
    poly_607 = poly_1 * poly_339
    poly_608 = poly_1 * poly_425
    poly_609 = poly_1 * poly_340
    poly_610 = poly_34 * poly_46
    poly_611 = poly_1 * poly_426
    poly_612 = poly_34 * poly_47
    poly_613 = poly_1 * poly_428
    poly_614 = poly_34 * poly_61
    poly_615 = poly_1 * poly_342
    poly_616 = poly_1 * poly_432
    poly_617 = poly_1 * poly_434
    poly_618 = poly_1 * poly_343
    poly_619 = poly_1 * poly_436
    poly_620 = poly_1 * poly_344
    poly_621 = poly_34 * poly_63
    poly_622 = poly_1 * poly_437
    poly_623 = poly_34 * poly_64
    poly_624 = poly_1 * poly_439
    poly_625 = poly_1 * poly_440
    poly_626 = poly_34 * poly_65
    poly_627 = poly_1 * poly_443
    poly_628 = poly_1 * poly_445
    poly_629 = poly_1 * poly_446
    poly_630 = poly_34 * poly_73
    poly_631 = poly_1 * poly_346
    poly_632 = poly_1 * poly_347
    poly_633 = poly_1 * poly_348
    poly_634 = poly_1 * poly_349
    poly_635 = poly_1 * poly_350
    poly_636 = poly_1 * poly_351
    poly_637 = poly_1 * poly_352
    poly_638 = poly_1 * poly_353
    poly_639 = poly_1 * poly_354
    poly_640 = poly_1 * poly_356
    poly_641 = poly_1 * poly_357
    poly_642 = poly_1 * poly_450
    poly_643 = poly_1 * poly_358
    poly_644 = poly_1 * poly_360
    poly_645 = poly_3 * poly_419
    poly_646 = poly_1 * poly_451
    poly_647 = poly_1 * poly_362
    poly_648 = poly_3 * poly_422
    poly_649 = poly_3 * poly_423
    poly_650 = poly_1 * poly_365
    poly_651 = poly_3 * poly_425
    poly_652 = poly_3 * poly_426
    poly_653 = poly_1 * poly_452
    poly_654 = poly_3 * poly_428
    poly_655 = poly_1 * poly_368
    poly_656 = poly_1 * poly_454
    poly_657 = poly_1 * poly_369
    poly_658 = poly_3 * poly_432
    poly_659 = poly_1 * poly_455
    poly_660 = poly_3 * poly_434
    poly_661 = poly_1 * poly_371
    poly_662 = poly_3 * poly_436
    poly_663 = poly_3 * poly_437
    poly_664 = poly_1 * poly_456
    poly_665 = poly_3 * poly_439
    poly_666 = poly_3 * poly_440
    poly_667 = poly_1 * poly_458
    poly_668 = poly_1 * poly_459
    poly_669 = poly_3 * poly_443
    poly_670 = poly_1 * poly_460
    poly_671 = poly_3 * poly_445
    poly_672 = poly_3 * poly_446
    poly_673 = poly_1 * poly_374
    poly_674 = poly_1 * poly_375
    poly_675 = poly_1 * poly_376
    poly_676 = poly_1 * poly_377
    poly_677 = poly_1 * poly_378
    poly_678 = poly_1 * poly_379
    poly_679 = poly_1 * poly_380
    poly_680 = poly_1 * poly_381
    poly_681 = poly_1 * poly_384
    poly_682 = poly_3 * poly_355
    poly_683 = poly_1 * poly_386
    poly_684 = poly_1 * poly_464
    poly_685 = poly_1 * poly_387
    poly_686 = poly_3 * poly_359
    poly_687 = poly_1 * poly_465
    poly_688 = poly_3 * poly_361
    poly_689 = poly_1 * poly_389
    poly_690 = poly_3 * poly_363
    poly_691 = poly_3 * poly_364
    poly_692 = poly_1 * poly_466
    poly_693 = poly_3 * poly_366
    poly_694 = poly_3 * poly_367
    poly_695 = poly_1 * poly_468
    poly_696 = poly_1 * poly_469
    poly_697 = poly_3 * poly_370
    poly_698 = poly_1 * poly_470
    poly_699 = poly_3 * poly_372
    poly_700 = poly_3 * poly_373
    poly_701 = poly_1 * poly_392
    poly_702 = poly_1 * poly_393
    poly_703 = poly_1 * poly_394
    poly_704 = poly_1 * poly_395
    poly_705 = poly_1 * poly_473
    poly_706 = poly_1 * poly_396
    poly_707 = poly_1 * poly_474
    poly_708 = poly_1 * poly_398
    poly_709 = poly_3 * poly_382
    poly_710 = poly_3 * poly_383
    poly_711 = poly_1 * poly_475
    poly_712 = poly_3 * poly_385
    poly_713 = poly_1 * poly_477
    poly_714 = poly_1 * poly_478
    poly_715 = poly_3 * poly_388
    poly_716 = poly_1 * poly_479
    poly_717 = poly_3 * poly_390
    poly_718 = poly_3 * poly_391
    poly_719 = poly_1 * poly_401
    poly_720 = poly_1 * poly_402
    poly_721 = poly_1 * poly_403
    poly_722 = poly_1 * poly_482
    poly_723 = poly_1 * poly_483
    poly_724 = poly_3 * poly_397
    poly_725 = poly_1 * poly_484
    poly_726 = poly_3 * poly_399
    poly_727 = poly_3 * poly_400
    poly_728 = poly_1 * poly_486
    poly_729 = poly_1 * poly_487
    poly_730 = poly_1 * poly_488
    poly_731 = poly_3 * poly_404
    poly_732 = poly_1 * poly_405
    poly_733 = poly_1 * poly_406
    poly_734 = poly_1 * poly_407
    poly_735 = poly_1 * poly_408
    poly_736 = poly_1 * poly_409
    poly_737 = poly_1 * poly_410
    poly_738 = poly_1 * poly_411
    poly_739 = poly_1 * poly_412
    poly_740 = poly_1 * poly_413
    poly_741 = poly_1 * poly_414
    poly_742 = poly_1 * poly_415
    poly_743 = poly_1 * poly_416
    poly_744 = poly_1 * poly_417
    poly_745 = poly_1 * poly_418
    poly_746 = poly_1 * poly_420
    poly_747 = poly_1 * poly_421
    poly_748 = poly_1 * poly_424
    poly_749 = poly_5 * poly_217 - poly_612
    poly_750 = poly_5 * poly_226 - poly_623
    poly_751 = poly_1 * poly_427
    poly_752 = poly_5 * poly_229 - poly_626
    poly_753 = poly_1 * poly_429
    poly_754 = poly_1 * poly_430
    poly_755 = poly_1 * poly_493
    poly_756 = poly_1 * poly_431
    poly_757 = poly_6 * poly_267
    poly_758 = poly_1 * poly_433
    poly_759 = poly_59 * poly_60
    poly_760 = poly_1 * poly_494
    poly_761 = poly_5 * poly_268
    poly_762 = poly_1 * poly_435
    poly_763 = poly_7 * poly_267
    poly_764 = poly_7 * poly_268
    poly_765 = poly_1 * poly_438
    poly_766 = poly_59 * poly_61
    poly_767 = poly_60 * poly_61
    poly_768 = poly_1 * poly_495
    poly_769 = poly_5 * poly_269
    poly_770 = poly_6 * poly_269
    poly_771 = poly_1 * poly_441
    poly_772 = poly_1 * poly_497
    poly_773 = poly_1 * poly_442
    poly_774 = poly_5 * poly_232 - poly_623
    poly_775 = poly_1 * poly_498
    poly_776 = poly_5 * poly_272 - poly_764
    poly_777 = poly_1 * poly_444
    poly_778 = poly_5 * poly_234 - poly_626
    poly_779 = poly_6 * poly_235 - poly_626
    poly_780 = poly_1 * poly_499
    poly_781 = poly_5 * poly_273 - poly_770
    poly_782 = poly_6 * poly_273 - poly_769
    poly_783 = poly_1 * poly_501
    poly_784 = poly_1 * poly_502
    poly_785 = poly_5 * poly_276 - poly_779
    poly_786 = poly_1 * poly_503
    poly_787 = poly_5 * poly_277 - poly_782
    poly_788 = poly_6 * poly_277 - poly_781
    poly_789 = poly_1 * poly_447
    poly_790 = poly_1 * poly_448
    poly_791 = poly_1 * poly_449
    poly_792 = poly_1 * poly_453
    poly_793 = poly_3 * poly_493
    poly_794 = poly_3 * poly_494
    poly_795 = poly_3 * poly_495
    poly_796 = poly_1 * poly_457
    poly_797 = poly_3 * poly_497
    poly_798 = poly_3 * poly_498
    poly_799 = poly_3 * poly_499
    poly_800 = poly_1 * poly_505
    poly_801 = poly_3 * poly_501
    poly_802 = poly_3 * poly_502
    poly_803 = poly_3 * poly_503
    poly_804 = poly_1 * poly_461
    poly_805 = poly_1 * poly_462
    poly_806 = poly_1 * poly_463
    poly_807 = poly_3 * poly_450
    poly_808 = poly_3 * poly_451
    poly_809 = poly_3 * poly_452
    poly_810 = poly_1 * poly_467
    poly_811 = poly_3 * poly_454
    poly_812 = poly_3 * poly_455
    poly_813 = poly_3 * poly_456
    poly_814 = poly_1 * poly_507
    poly_815 = poly_3 * poly_458
    poly_816 = poly_3 * poly_459
    poly_817 = poly_3 * poly_460
    poly_818 = poly_1 * poly_471
    poly_819 = poly_1 * poly_472
    poly_820 = poly_1 * poly_476
    poly_821 = poly_3 * poly_464
    poly_822 = poly_3 * poly_465
    poly_823 = poly_3 * poly_466
    poly_824 = poly_1 * poly_509
    poly_825 = poly_3 * poly_468
    poly_826 = poly_3 * poly_469
    poly_827 = poly_3 * poly_470
    poly_828 = poly_1 * poly_480
    poly_829 = poly_1 * poly_481
    poly_830 = poly_3 * poly_473
    poly_831 = poly_3 * poly_474
    poly_832 = poly_3 * poly_475
    poly_833 = poly_1 * poly_511
    poly_834 = poly_3 * poly_477
    poly_835 = poly_3 * poly_478
    poly_836 = poly_3 * poly_479
    poly_837 = poly_1 * poly_485
    poly_838 = poly_1 * poly_513
    poly_839 = poly_3 * poly_482
    poly_840 = poly_3 * poly_483
    poly_841 = poly_3 * poly_484
    poly_842 = poly_1 * poly_515
    poly_843 = poly_3 * poly_486
    poly_844 = poly_3 * poly_487
    poly_845 = poly_3 * poly_488
    poly_846 = poly_1 * poly_489
    poly_847 = poly_1 * poly_490
    poly_848 = poly_1 * poly_491
    poly_849 = poly_1 * poly_492
    poly_850 = poly_5 * poly_267 - poly_602
    poly_851 = poly_6 * poly_268 - poly_606
    poly_852 = poly_7 * poly_269 - poly_614
    poly_853 = poly_1 * poly_496
    poly_854 = poly_5 * poly_271 - poly_621
    poly_855 = poly_6 * poly_272 - poly_623
    poly_856 = poly_7 * poly_273 - poly_626
    poly_857 = poly_1 * poly_500
    poly_858 = poly_5 * poly_275 - poly_630
    poly_859 = poly_6 * poly_276 - poly_630
    poly_860 = poly_7 * poly_277 - poly_630
    poly_861 = poly_1 * poly_517
    poly_862 = poly_5 * poly_289 - poly_788
    poly_863 = poly_6 * poly_289 - poly_787
    poly_864 = poly_7 * poly_289 - poly_785
    poly_865 = poly_1 * poly_504
    poly_866 = poly_3 * poly_517
    poly_867 = poly_1 * poly_506
    poly_868 = poly_3 * poly_505
    poly_869 = poly_1 * poly_508
    poly_870 = poly_3 * poly_507
    poly_871 = poly_1 * poly_510
    poly_872 = poly_3 * poly_509
    poly_873 = poly_1 * poly_512
    poly_874 = poly_3 * poly_511
    poly_875 = poly_1 * poly_514
    poly_876 = poly_3 * poly_513
    poly_877 = poly_1 * poly_518
    poly_878 = poly_3 * poly_515
    poly_879 = poly_1 * poly_516
    poly_880 = poly_2 * poly_517 - poly_864 - poly_863 - poly_862
    poly_881 = poly_3 * poly_518

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
                      poly_881,])

    return poly
