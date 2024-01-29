import jax 
import jax.numpy as jnp 
from jax import jit

# File created from /Users/ravh011/Documents/GitHub/PIPMSA_jax/Data/Methane/PIP_deg_5/MOL_4_1_5.MONO 

# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
N_DISTANCES = 10
N_ATOMS = 5
N_XYZ = N_ATOMS * 3

# Total number of monomials = 496 

@jit
def f_monomials(r): 
    assert(r.shape == (N_DISTANCES,))

    mono = jnp.zeros(496) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,9) 
    mono_2 = jnp.take(r,8) 
    mono_3 = jnp.take(r,6) 
    mono_4 = jnp.take(r,3) 
    mono_5 = jnp.take(r,7) 
    mono_6 = jnp.take(r,5) 
    mono_7 = jnp.take(r,4) 
    mono_8 = jnp.take(r,2) 
    mono_9 = jnp.take(r,1) 
    mono_10 = jnp.take(r,0) 
    mono_11 = mono_1 * mono_2 
    mono_12 = mono_1 * mono_3 
    mono_13 = mono_2 * mono_3 
    mono_14 = mono_1 * mono_4 
    mono_15 = mono_2 * mono_4 
    mono_16 = mono_3 * mono_4 
    mono_17 = mono_3 * mono_5 
    mono_18 = mono_2 * mono_6 
    mono_19 = mono_1 * mono_7 
    mono_20 = mono_4 * mono_5 
    mono_21 = mono_4 * mono_6 
    mono_22 = mono_4 * mono_7 
    mono_23 = mono_2 * mono_8 
    mono_24 = mono_3 * mono_8 
    mono_25 = mono_1 * mono_9 
    mono_26 = mono_3 * mono_9 
    mono_27 = mono_1 * mono_10 
    mono_28 = mono_2 * mono_10 
    mono_29 = mono_7 * mono_8 
    mono_30 = mono_6 * mono_9 
    mono_31 = mono_5 * mono_10 
    mono_32 = mono_5 * mono_6 
    mono_33 = mono_5 * mono_7 
    mono_34 = mono_6 * mono_7 
    mono_35 = mono_5 * mono_8 
    mono_36 = mono_6 * mono_8 
    mono_37 = mono_5 * mono_9 
    mono_38 = mono_7 * mono_9 
    mono_39 = mono_8 * mono_9 
    mono_40 = mono_6 * mono_10 
    mono_41 = mono_7 * mono_10 
    mono_42 = mono_8 * mono_10 
    mono_43 = mono_9 * mono_10 
    mono_44 = mono_1 * mono_13 
    mono_45 = mono_1 * mono_15 
    mono_46 = mono_1 * mono_16 
    mono_47 = mono_2 * mono_16 
    mono_48 = mono_3 * mono_20 
    mono_49 = mono_2 * mono_21 
    mono_50 = mono_1 * mono_22 
    mono_51 = mono_2 * mono_24 
    mono_52 = mono_1 * mono_26 
    mono_53 = mono_1 * mono_28 
    mono_54 = mono_1 * mono_17 
    mono_55 = mono_2 * mono_17 
    mono_56 = mono_1 * mono_18 
    mono_57 = mono_3 * mono_18 
    mono_58 = mono_2 * mono_19 
    mono_59 = mono_3 * mono_19 
    mono_60 = mono_1 * mono_20 
    mono_61 = mono_2 * mono_20 
    mono_62 = mono_1 * mono_21 
    mono_63 = mono_3 * mono_21 
    mono_64 = mono_2 * mono_22 
    mono_65 = mono_3 * mono_22 
    mono_66 = mono_1 * mono_23 
    mono_67 = mono_1 * mono_24 
    mono_68 = mono_4 * mono_23 
    mono_69 = mono_4 * mono_24 
    mono_70 = mono_2 * mono_25 
    mono_71 = mono_2 * mono_26 
    mono_72 = mono_4 * mono_25 
    mono_73 = mono_4 * mono_26 
    mono_74 = mono_3 * mono_27 
    mono_75 = mono_3 * mono_28 
    mono_76 = mono_4 * mono_27 
    mono_77 = mono_4 * mono_28 
    mono_78 = mono_4 * mono_32 
    mono_79 = mono_4 * mono_33 
    mono_80 = mono_4 * mono_34 
    mono_81 = mono_3 * mono_35 
    mono_82 = mono_2 * mono_36 
    mono_83 = mono_3 * mono_37 
    mono_84 = mono_1 * mono_38 
    mono_85 = mono_3 * mono_39 
    mono_86 = mono_2 * mono_40 
    mono_87 = mono_1 * mono_41 
    mono_88 = mono_2 * mono_42 
    mono_89 = mono_1 * mono_43 
    mono_90 = mono_2 * mono_32 
    mono_91 = mono_3 * mono_32 
    mono_92 = mono_1 * mono_33 
    mono_93 = mono_3 * mono_33 
    mono_94 = mono_1 * mono_34 
    mono_95 = mono_2 * mono_34 
    mono_96 = mono_2 * mono_35 
    mono_97 = mono_3 * mono_36 
    mono_98 = mono_4 * mono_35 
    mono_99 = mono_4 * mono_36 
    mono_100 = mono_1 * mono_37 
    mono_101 = mono_3 * mono_38 
    mono_102 = mono_4 * mono_37 
    mono_103 = mono_4 * mono_38 
    mono_104 = mono_1 * mono_39 
    mono_105 = mono_2 * mono_39 
    mono_106 = mono_1 * mono_40 
    mono_107 = mono_2 * mono_41 
    mono_108 = mono_4 * mono_40 
    mono_109 = mono_4 * mono_41 
    mono_110 = mono_1 * mono_42 
    mono_111 = mono_3 * mono_42 
    mono_112 = mono_2 * mono_43 
    mono_113 = mono_3 * mono_43 
    mono_114 = mono_5 * mono_29 
    mono_115 = mono_6 * mono_29 
    mono_116 = mono_5 * mono_30 
    mono_117 = mono_6 * mono_38 
    mono_118 = mono_6 * mono_39 
    mono_119 = mono_7 * mono_39 
    mono_120 = mono_5 * mono_40 
    mono_121 = mono_5 * mono_41 
    mono_122 = mono_5 * mono_42 
    mono_123 = mono_7 * mono_42 
    mono_124 = mono_5 * mono_43 
    mono_125 = mono_6 * mono_43 
    mono_126 = mono_5 * mono_34 
    mono_127 = mono_5 * mono_39 
    mono_128 = mono_6 * mono_42 
    mono_129 = mono_7 * mono_43 
    mono_130 = mono_5 * mono_36 
    mono_131 = mono_5 * mono_38 
    mono_132 = mono_6 * mono_41 
    mono_133 = mono_8 * mono_43 
    mono_134 = mono_1 * mono_47 
    mono_135 = mono_1 * mono_48 
    mono_136 = mono_2 * mono_48 
    mono_137 = mono_1 * mono_49 
    mono_138 = mono_2 * mono_63 
    mono_139 = mono_1 * mono_64 
    mono_140 = mono_1 * mono_65 
    mono_141 = mono_1 * mono_51 
    mono_142 = mono_2 * mono_69 
    mono_143 = mono_1 * mono_71 
    mono_144 = mono_1 * mono_73 
    mono_145 = mono_1 * mono_75 
    mono_146 = mono_1 * mono_77 
    mono_147 = mono_7 * mono_66 
    mono_148 = mono_7 * mono_67 
    mono_149 = mono_7 * mono_68 
    mono_150 = mono_7 * mono_69 
    mono_151 = mono_6 * mono_70 
    mono_152 = mono_6 * mono_71 
    mono_153 = mono_6 * mono_72 
    mono_154 = mono_6 * mono_73 
    mono_155 = mono_5 * mono_74 
    mono_156 = mono_5 * mono_75 
    mono_157 = mono_5 * mono_76 
    mono_158 = mono_5 * mono_77 
    mono_159 = mono_2 * mono_78 
    mono_160 = mono_3 * mono_78 
    mono_161 = mono_1 * mono_79 
    mono_162 = mono_3 * mono_79 
    mono_163 = mono_1 * mono_80 
    mono_164 = mono_2 * mono_80 
    mono_165 = mono_2 * mono_81 
    mono_166 = mono_2 * mono_97 
    mono_167 = mono_3 * mono_98 
    mono_168 = mono_2 * mono_99 
    mono_169 = mono_1 * mono_83 
    mono_170 = mono_1 * mono_101 
    mono_171 = mono_3 * mono_102 
    mono_172 = mono_1 * mono_103 
    mono_173 = mono_1 * mono_85 
    mono_174 = mono_2 * mono_85 
    mono_175 = mono_1 * mono_86 
    mono_176 = mono_1 * mono_107 
    mono_177 = mono_2 * mono_108 
    mono_178 = mono_1 * mono_109 
    mono_179 = mono_1 * mono_88 
    mono_180 = mono_2 * mono_111 
    mono_181 = mono_1 * mono_112 
    mono_182 = mono_1 * mono_113 
    mono_183 = mono_2 * mono_91 
    mono_184 = mono_1 * mono_93 
    mono_185 = mono_1 * mono_95 
    mono_186 = mono_2 * mono_98 
    mono_187 = mono_3 * mono_99 
    mono_188 = mono_1 * mono_102 
    mono_189 = mono_3 * mono_103 
    mono_190 = mono_1 * mono_105 
    mono_191 = mono_1 * mono_108 
    mono_192 = mono_2 * mono_109 
    mono_193 = mono_1 * mono_111 
    mono_194 = mono_2 * mono_113 
    mono_195 = mono_3 * mono_114 
    mono_196 = mono_2 * mono_115 
    mono_197 = mono_4 * mono_114 
    mono_198 = mono_4 * mono_115 
    mono_199 = mono_3 * mono_116 
    mono_200 = mono_1 * mono_117 
    mono_201 = mono_4 * mono_116 
    mono_202 = mono_4 * mono_117 
    mono_203 = mono_2 * mono_118 
    mono_204 = mono_3 * mono_118 
    mono_205 = mono_1 * mono_119 
    mono_206 = mono_3 * mono_119 
    mono_207 = mono_2 * mono_120 
    mono_208 = mono_1 * mono_121 
    mono_209 = mono_4 * mono_120 
    mono_210 = mono_4 * mono_121 
    mono_211 = mono_2 * mono_122 
    mono_212 = mono_3 * mono_122 
    mono_213 = mono_1 * mono_123 
    mono_214 = mono_2 * mono_123 
    mono_215 = mono_1 * mono_124 
    mono_216 = mono_3 * mono_124 
    mono_217 = mono_1 * mono_125 
    mono_218 = mono_2 * mono_125 
    mono_219 = mono_6 * mono_119 
    mono_220 = mono_5 * mono_123 
    mono_221 = mono_5 * mono_125 
    mono_222 = mono_4 * mono_126 
    mono_223 = mono_3 * mono_127 
    mono_224 = mono_2 * mono_128 
    mono_225 = mono_1 * mono_129 
    mono_226 = mono_1 * mono_78 
    mono_227 = mono_2 * mono_79 
    mono_228 = mono_3 * mono_80 
    mono_229 = mono_1 * mono_81 
    mono_230 = mono_1 * mono_82 
    mono_231 = mono_2 * mono_83 
    mono_232 = mono_2 * mono_84 
    mono_233 = mono_4 * mono_85 
    mono_234 = mono_3 * mono_86 
    mono_235 = mono_3 * mono_87 
    mono_236 = mono_4 * mono_88 
    mono_237 = mono_4 * mono_89 
    mono_238 = mono_2 * mono_130 
    mono_239 = mono_3 * mono_130 
    mono_240 = mono_4 * mono_130 
    mono_241 = mono_1 * mono_131 
    mono_242 = mono_3 * mono_131 
    mono_243 = mono_4 * mono_131 
    mono_244 = mono_1 * mono_132 
    mono_245 = mono_2 * mono_132 
    mono_246 = mono_4 * mono_132 
    mono_247 = mono_1 * mono_133 
    mono_248 = mono_2 * mono_133 
    mono_249 = mono_3 * mono_133 
    mono_250 = mono_5 * mono_115 
    mono_251 = mono_5 * mono_117 
    mono_252 = mono_5 * mono_118 
    mono_253 = mono_5 * mono_119 
    mono_254 = mono_5 * mono_132 
    mono_255 = mono_5 * mono_128 
    mono_256 = mono_6 * mono_123 
    mono_257 = mono_5 * mono_129 
    mono_258 = mono_6 * mono_129 
    mono_259 = mono_5 * mono_133 
    mono_260 = mono_6 * mono_133 
    mono_261 = mono_7 * mono_133 
    mono_262 = mono_2 * mono_160 
    mono_263 = mono_1 * mono_162 
    mono_264 = mono_1 * mono_164 
    mono_265 = mono_2 * mono_167 
    mono_266 = mono_2 * mono_187 
    mono_267 = mono_1 * mono_171 
    mono_268 = mono_1 * mono_189 
    mono_269 = mono_1 * mono_174 
    mono_270 = mono_1 * mono_177 
    mono_271 = mono_1 * mono_192 
    mono_272 = mono_1 * mono_180 
    mono_273 = mono_1 * mono_194 
    mono_274 = mono_3 * mono_197 
    mono_275 = mono_2 * mono_198 
    mono_276 = mono_3 * mono_201 
    mono_277 = mono_1 * mono_202 
    mono_278 = mono_2 * mono_204 
    mono_279 = mono_1 * mono_206 
    mono_280 = mono_2 * mono_209 
    mono_281 = mono_1 * mono_210 
    mono_282 = mono_2 * mono_212 
    mono_283 = mono_1 * mono_214 
    mono_284 = mono_1 * mono_216 
    mono_285 = mono_1 * mono_218 
    mono_286 = mono_1 * mono_159 
    mono_287 = mono_1 * mono_160 
    mono_288 = mono_1 * mono_227 
    mono_289 = mono_2 * mono_162 
    mono_290 = mono_1 * mono_228 
    mono_291 = mono_2 * mono_228 
    mono_292 = mono_1 * mono_165 
    mono_293 = mono_1 * mono_166 
    mono_294 = mono_1 * mono_167 
    mono_295 = mono_1 * mono_168 
    mono_296 = mono_1 * mono_231 
    mono_297 = mono_2 * mono_170 
    mono_298 = mono_2 * mono_171 
    mono_299 = mono_2 * mono_172 
    mono_300 = mono_1 * mono_233 
    mono_301 = mono_2 * mono_233 
    mono_302 = mono_1 * mono_234 
    mono_303 = mono_2 * mono_235 
    mono_304 = mono_3 * mono_177 
    mono_305 = mono_3 * mono_178 
    mono_306 = mono_1 * mono_236 
    mono_307 = mono_3 * mono_236 
    mono_308 = mono_2 * mono_237 
    mono_309 = mono_3 * mono_237 
    mono_310 = mono_1 * mono_195 
    mono_311 = mono_1 * mono_196 
    mono_312 = mono_2 * mono_197 
    mono_313 = mono_3 * mono_198 
    mono_314 = mono_2 * mono_199 
    mono_315 = mono_2 * mono_200 
    mono_316 = mono_1 * mono_201 
    mono_317 = mono_3 * mono_202 
    mono_318 = mono_1 * mono_203 
    mono_319 = mono_2 * mono_205 
    mono_320 = mono_4 * mono_204 
    mono_321 = mono_4 * mono_206 
    mono_322 = mono_3 * mono_207 
    mono_323 = mono_3 * mono_208 
    mono_324 = mono_1 * mono_209 
    mono_325 = mono_2 * mono_210 
    mono_326 = mono_1 * mono_212 
    mono_327 = mono_3 * mono_213 
    mono_328 = mono_4 * mono_211 
    mono_329 = mono_4 * mono_214 
    mono_330 = mono_2 * mono_216 
    mono_331 = mono_3 * mono_218 
    mono_332 = mono_4 * mono_215 
    mono_333 = mono_4 * mono_217 
    mono_334 = mono_2 * mono_195 
    mono_335 = mono_3 * mono_196 
    mono_336 = mono_1 * mono_197 
    mono_337 = mono_1 * mono_198 
    mono_338 = mono_1 * mono_199 
    mono_339 = mono_3 * mono_200 
    mono_340 = mono_2 * mono_201 
    mono_341 = mono_2 * mono_202 
    mono_342 = mono_1 * mono_204 
    mono_343 = mono_2 * mono_206 
    mono_344 = mono_4 * mono_203 
    mono_345 = mono_4 * mono_205 
    mono_346 = mono_1 * mono_207 
    mono_347 = mono_2 * mono_208 
    mono_348 = mono_3 * mono_209 
    mono_349 = mono_3 * mono_210 
    mono_350 = mono_1 * mono_211 
    mono_351 = mono_3 * mono_214 
    mono_352 = mono_4 * mono_212 
    mono_353 = mono_4 * mono_213 
    mono_354 = mono_2 * mono_215 
    mono_355 = mono_3 * mono_217 
    mono_356 = mono_4 * mono_216 
    mono_357 = mono_4 * mono_218 
    mono_358 = mono_1 * mono_222 
    mono_359 = mono_2 * mono_222 
    mono_360 = mono_3 * mono_222 
    mono_361 = mono_1 * mono_223 
    mono_362 = mono_2 * mono_223 
    mono_363 = mono_4 * mono_223 
    mono_364 = mono_1 * mono_224 
    mono_365 = mono_3 * mono_224 
    mono_366 = mono_4 * mono_224 
    mono_367 = mono_2 * mono_225 
    mono_368 = mono_3 * mono_225 
    mono_369 = mono_4 * mono_225 
    mono_370 = mono_2 * mono_239 
    mono_371 = mono_2 * mono_240 
    mono_372 = mono_3 * mono_240 
    mono_373 = mono_1 * mono_242 
    mono_374 = mono_1 * mono_243 
    mono_375 = mono_3 * mono_243 
    mono_376 = mono_1 * mono_245 
    mono_377 = mono_1 * mono_246 
    mono_378 = mono_2 * mono_246 
    mono_379 = mono_1 * mono_248 
    mono_380 = mono_1 * mono_249 
    mono_381 = mono_2 * mono_249 
    mono_382 = mono_4 * mono_250 
    mono_383 = mono_4 * mono_251 
    mono_384 = mono_3 * mono_252 
    mono_385 = mono_3 * mono_253 
    mono_386 = mono_4 * mono_254 
    mono_387 = mono_2 * mono_255 
    mono_388 = mono_2 * mono_256 
    mono_389 = mono_1 * mono_257 
    mono_390 = mono_1 * mono_258 
    mono_391 = mono_3 * mono_259 
    mono_392 = mono_2 * mono_260 
    mono_393 = mono_1 * mono_261 
    mono_394 = mono_2 * mono_250 
    mono_395 = mono_3 * mono_250 
    mono_396 = mono_1 * mono_251 
    mono_397 = mono_3 * mono_251 
    mono_398 = mono_2 * mono_252 
    mono_399 = mono_1 * mono_253 
    mono_400 = mono_4 * mono_252 
    mono_401 = mono_4 * mono_253 
    mono_402 = mono_1 * mono_254 
    mono_403 = mono_2 * mono_254 
    mono_404 = mono_3 * mono_255 
    mono_405 = mono_1 * mono_256 
    mono_406 = mono_4 * mono_255 
    mono_407 = mono_4 * mono_256 
    mono_408 = mono_3 * mono_257 
    mono_409 = mono_2 * mono_258 
    mono_410 = mono_4 * mono_257 
    mono_411 = mono_4 * mono_258 
    mono_412 = mono_1 * mono_259 
    mono_413 = mono_2 * mono_259 
    mono_414 = mono_1 * mono_260 
    mono_415 = mono_3 * mono_260 
    mono_416 = mono_2 * mono_261 
    mono_417 = mono_3 * mono_261 
    mono_418 = mono_5 * mono_219 
    mono_419 = mono_5 * mono_256 
    mono_420 = mono_5 * mono_258 
    mono_421 = mono_5 * mono_260 
    mono_422 = mono_5 * mono_261 
    mono_423 = mono_6 * mono_261 
    mono_424 = mono_3 * mono_135 
    mono_425 = mono_3 * mono_136 
    mono_426 = mono_2 * mono_137 
    mono_427 = mono_2 * mono_138 
    mono_428 = mono_1 * mono_139 
    mono_429 = mono_1 * mono_140 
    mono_430 = mono_4 * mono_135 
    mono_431 = mono_4 * mono_136 
    mono_432 = mono_4 * mono_137 
    mono_433 = mono_4 * mono_138 
    mono_434 = mono_4 * mono_139 
    mono_435 = mono_4 * mono_140 
    mono_436 = mono_2 * mono_141 
    mono_437 = mono_3 * mono_141 
    mono_438 = mono_2 * mono_142 
    mono_439 = mono_3 * mono_142 
    mono_440 = mono_1 * mono_143 
    mono_441 = mono_3 * mono_143 
    mono_442 = mono_1 * mono_144 
    mono_443 = mono_3 * mono_144 
    mono_444 = mono_1 * mono_145 
    mono_445 = mono_2 * mono_145 
    mono_446 = mono_1 * mono_146 
    mono_447 = mono_2 * mono_146 
    mono_448 = mono_4 * mono_159 
    mono_449 = mono_4 * mono_160 
    mono_450 = mono_4 * mono_161 
    mono_451 = mono_4 * mono_162 
    mono_452 = mono_4 * mono_163 
    mono_453 = mono_4 * mono_164 
    mono_454 = mono_3 * mono_165 
    mono_455 = mono_2 * mono_166 
    mono_456 = mono_3 * mono_167 
    mono_457 = mono_2 * mono_168 
    mono_458 = mono_3 * mono_169 
    mono_459 = mono_1 * mono_170 
    mono_460 = mono_3 * mono_171 
    mono_461 = mono_1 * mono_172 
    mono_462 = mono_3 * mono_173 
    mono_463 = mono_3 * mono_174 
    mono_464 = mono_2 * mono_175 
    mono_465 = mono_1 * mono_176 
    mono_466 = mono_2 * mono_177 
    mono_467 = mono_1 * mono_178 
    mono_468 = mono_2 * mono_179 
    mono_469 = mono_2 * mono_180 
    mono_470 = mono_1 * mono_181 
    mono_471 = mono_1 * mono_182 
    mono_472 = mono_19 * mono_114 
    mono_473 = mono_19 * mono_115 
    mono_474 = mono_23 * mono_114 
    mono_475 = mono_24 * mono_115 
    mono_476 = mono_18 * mono_116 
    mono_477 = mono_18 * mono_117 
    mono_478 = mono_21 * mono_118 
    mono_479 = mono_22 * mono_119 
    mono_480 = mono_23 * mono_119 
    mono_481 = mono_25 * mono_116 
    mono_482 = mono_26 * mono_117 
    mono_483 = mono_25 * mono_118 
    mono_484 = mono_17 * mono_120 
    mono_485 = mono_17 * mono_121 
    mono_486 = mono_20 * mono_122 
    mono_487 = mono_22 * mono_123 
    mono_488 = mono_24 * mono_123 
    mono_489 = mono_20 * mono_124 
    mono_490 = mono_21 * mono_125 
    mono_491 = mono_26 * mono_125 
    mono_492 = mono_27 * mono_120 
    mono_493 = mono_28 * mono_121 
    mono_494 = mono_27 * mono_122 
    mono_495 = mono_28 * mono_124 

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
    mono_251,    mono_252,    mono_253,    mono_254,    mono_255, 
    mono_256,    mono_257,    mono_258,    mono_259,    mono_260, 
    mono_261,    mono_262,    mono_263,    mono_264,    mono_265, 
    mono_266,    mono_267,    mono_268,    mono_269,    mono_270, 
    mono_271,    mono_272,    mono_273,    mono_274,    mono_275, 
    mono_276,    mono_277,    mono_278,    mono_279,    mono_280, 
    mono_281,    mono_282,    mono_283,    mono_284,    mono_285, 
    mono_286,    mono_287,    mono_288,    mono_289,    mono_290, 
    mono_291,    mono_292,    mono_293,    mono_294,    mono_295, 
    mono_296,    mono_297,    mono_298,    mono_299,    mono_300, 
    mono_301,    mono_302,    mono_303,    mono_304,    mono_305, 
    mono_306,    mono_307,    mono_308,    mono_309,    mono_310, 
    mono_311,    mono_312,    mono_313,    mono_314,    mono_315, 
    mono_316,    mono_317,    mono_318,    mono_319,    mono_320, 
    mono_321,    mono_322,    mono_323,    mono_324,    mono_325, 
    mono_326,    mono_327,    mono_328,    mono_329,    mono_330, 
    mono_331,    mono_332,    mono_333,    mono_334,    mono_335, 
    mono_336,    mono_337,    mono_338,    mono_339,    mono_340, 
    mono_341,    mono_342,    mono_343,    mono_344,    mono_345, 
    mono_346,    mono_347,    mono_348,    mono_349,    mono_350, 
    mono_351,    mono_352,    mono_353,    mono_354,    mono_355, 
    mono_356,    mono_357,    mono_358,    mono_359,    mono_360, 
    mono_361,    mono_362,    mono_363,    mono_364,    mono_365, 
    mono_366,    mono_367,    mono_368,    mono_369,    mono_370, 
    mono_371,    mono_372,    mono_373,    mono_374,    mono_375, 
    mono_376,    mono_377,    mono_378,    mono_379,    mono_380, 
    mono_381,    mono_382,    mono_383,    mono_384,    mono_385, 
    mono_386,    mono_387,    mono_388,    mono_389,    mono_390, 
    mono_391,    mono_392,    mono_393,    mono_394,    mono_395, 
    mono_396,    mono_397,    mono_398,    mono_399,    mono_400, 
    mono_401,    mono_402,    mono_403,    mono_404,    mono_405, 
    mono_406,    mono_407,    mono_408,    mono_409,    mono_410, 
    mono_411,    mono_412,    mono_413,    mono_414,    mono_415, 
    mono_416,    mono_417,    mono_418,    mono_419,    mono_420, 
    mono_421,    mono_422,    mono_423,    mono_424,    mono_425, 
    mono_426,    mono_427,    mono_428,    mono_429,    mono_430, 
    mono_431,    mono_432,    mono_433,    mono_434,    mono_435, 
    mono_436,    mono_437,    mono_438,    mono_439,    mono_440, 
    mono_441,    mono_442,    mono_443,    mono_444,    mono_445, 
    mono_446,    mono_447,    mono_448,    mono_449,    mono_450, 
    mono_451,    mono_452,    mono_453,    mono_454,    mono_455, 
    mono_456,    mono_457,    mono_458,    mono_459,    mono_460, 
    mono_461,    mono_462,    mono_463,    mono_464,    mono_465, 
    mono_466,    mono_467,    mono_468,    mono_469,    mono_470, 
    mono_471,    mono_472,    mono_473,    mono_474,    mono_475, 
    mono_476,    mono_477,    mono_478,    mono_479,    mono_480, 
    mono_481,    mono_482,    mono_483,    mono_484,    mono_485, 
    mono_486,    mono_487,    mono_488,    mono_489,    mono_490, 
    mono_491,    mono_492,    mono_493,    mono_494,    mono_495, 
    ]) 

    return mono 



