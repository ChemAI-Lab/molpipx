import jax 
import jax.numpy as jnp 
from jax import jit

from pipx.msa_files.molecule_A2BCD.monomials_MOL_2_1_1_1_4 import f_monomials as f_monos 


# File created from /gpfs/fs1/home/r/ravh011/ravh011/PIPMSA_jax/pipjax/msa_files/molecule_A2BCD/MOL_2_1_1_1_4.POLY 


N_POLYS = 561

# Total number of monomials = 561 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(561) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = jnp.take(mono,4) + jnp.take(mono,5) 
    poly_5 = jnp.take(mono,6) + jnp.take(mono,7) 
    poly_6 = jnp.take(mono,8) + jnp.take(mono,9) 
    poly_7 = jnp.take(mono,10) 
    poly_8 = poly_1 * poly_2 
    poly_9 = poly_1 * poly_3 
    poly_10 = poly_2 * poly_3 
    poly_11 = poly_1 * poly_4 
    poly_12 = poly_2 * poly_4 
    poly_13 = poly_3 * poly_4 
    poly_14 = jnp.take(mono,11) 
    poly_15 = poly_1 * poly_5 
    poly_16 = poly_2 * poly_5 
    poly_17 = poly_3 * poly_5 
    poly_18 = jnp.take(mono,12) + jnp.take(mono,13) 
    poly_19 = jnp.take(mono,14) 
    poly_20 = poly_4 * poly_5 - poly_18 
    poly_21 = poly_1 * poly_6 
    poly_22 = poly_2 * poly_6 
    poly_23 = poly_3 * poly_6 
    poly_24 = jnp.take(mono,15) + jnp.take(mono,16) 
    poly_25 = jnp.take(mono,17) + jnp.take(mono,18) 
    poly_26 = jnp.take(mono,19) 
    poly_27 = poly_4 * poly_6 - poly_24 
    poly_28 = poly_5 * poly_6 - poly_25 
    poly_29 = poly_1 * poly_7 
    poly_30 = poly_2 * poly_7 
    poly_31 = poly_3 * poly_7 
    poly_32 = poly_7 * poly_4 
    poly_33 = poly_7 * poly_5 
    poly_34 = poly_7 * poly_6 
    poly_35 = poly_1 * poly_1 
    poly_36 = poly_2 * poly_2 
    poly_37 = poly_3 * poly_3 
    poly_38 = poly_4 * poly_4 - poly_14 - poly_14 
    poly_39 = poly_5 * poly_5 - poly_19 - poly_19 
    poly_40 = poly_6 * poly_6 - poly_26 - poly_26 
    poly_41 = poly_7 * poly_7 
    poly_42 = poly_1 * poly_10 
    poly_43 = poly_1 * poly_12 
    poly_44 = poly_1 * poly_13 
    poly_45 = poly_2 * poly_13 
    poly_46 = poly_1 * poly_14 
    poly_47 = poly_2 * poly_14 
    poly_48 = poly_3 * poly_14 
    poly_49 = poly_1 * poly_16 
    poly_50 = poly_1 * poly_17 
    poly_51 = poly_2 * poly_17 
    poly_52 = poly_1 * poly_18 
    poly_53 = poly_2 * poly_18 
    poly_54 = poly_3 * poly_18 
    poly_55 = poly_1 * poly_19 
    poly_56 = poly_2 * poly_19 
    poly_57 = poly_3 * poly_19 
    poly_58 = poly_1 * poly_20 
    poly_59 = poly_2 * poly_20 
    poly_60 = poly_3 * poly_20 
    poly_61 = poly_14 * poly_5 
    poly_62 = poly_19 * poly_4 
    poly_63 = poly_1 * poly_22 
    poly_64 = poly_1 * poly_23 
    poly_65 = poly_2 * poly_23 
    poly_66 = poly_1 * poly_24 
    poly_67 = poly_2 * poly_24 
    poly_68 = poly_3 * poly_24 
    poly_69 = poly_1 * poly_25 
    poly_70 = poly_2 * poly_25 
    poly_71 = poly_3 * poly_25 
    poly_72 = jnp.take(mono,20) + jnp.take(mono,21) 
    poly_73 = poly_1 * poly_26 
    poly_74 = poly_2 * poly_26 
    poly_75 = poly_3 * poly_26 
    poly_76 = poly_1 * poly_27 
    poly_77 = poly_2 * poly_27 
    poly_78 = poly_3 * poly_27 
    poly_79 = poly_14 * poly_6 
    poly_80 = poly_4 * poly_25 - poly_72 
    poly_81 = poly_26 * poly_4 
    poly_82 = poly_1 * poly_28 
    poly_83 = poly_2 * poly_28 
    poly_84 = poly_3 * poly_28 
    poly_85 = poly_5 * poly_24 - poly_72 
    poly_86 = poly_19 * poly_6 
    poly_87 = poly_26 * poly_5 
    poly_88 = poly_4 * poly_28 - poly_85 
    poly_89 = poly_1 * poly_30 
    poly_90 = poly_1 * poly_31 
    poly_91 = poly_2 * poly_31 
    poly_92 = poly_1 * poly_32 
    poly_93 = poly_2 * poly_32 
    poly_94 = poly_3 * poly_32 
    poly_95 = poly_7 * poly_14 
    poly_96 = poly_1 * poly_33 
    poly_97 = poly_2 * poly_33 
    poly_98 = poly_3 * poly_33 
    poly_99 = poly_7 * poly_18 
    poly_100 = poly_7 * poly_19 
    poly_101 = poly_7 * poly_20 
    poly_102 = poly_1 * poly_34 
    poly_103 = poly_2 * poly_34 
    poly_104 = poly_3 * poly_34 
    poly_105 = poly_7 * poly_24 
    poly_106 = poly_7 * poly_25 
    poly_107 = poly_7 * poly_26 
    poly_108 = poly_7 * poly_27 
    poly_109 = poly_7 * poly_28 
    poly_110 = poly_1 * poly_8 
    poly_111 = poly_1 * poly_36 
    poly_112 = poly_1 * poly_9 
    poly_113 = poly_2 * poly_10 
    poly_114 = poly_1 * poly_37 
    poly_115 = poly_2 * poly_37 
    poly_116 = poly_1 * poly_11 
    poly_117 = poly_2 * poly_12 
    poly_118 = poly_3 * poly_13 
    poly_119 = poly_1 * poly_38 
    poly_120 = poly_2 * poly_38 
    poly_121 = poly_3 * poly_38 
    poly_122 = poly_14 * poly_4 
    poly_123 = poly_1 * poly_15 
    poly_124 = poly_2 * poly_16 
    poly_125 = poly_3 * poly_17 
    poly_126 = poly_4 * poly_18 - poly_61 
    poly_127 = poly_4 * poly_20 - poly_61 
    poly_128 = poly_1 * poly_39 
    poly_129 = poly_2 * poly_39 
    poly_130 = poly_3 * poly_39 
    poly_131 = poly_5 * poly_18 - poly_62 
    poly_132 = poly_19 * poly_5 
    poly_133 = poly_4 * poly_39 - poly_131 
    poly_134 = poly_1 * poly_21 
    poly_135 = poly_2 * poly_22 
    poly_136 = poly_3 * poly_23 
    poly_137 = poly_4 * poly_24 - poly_79 
    poly_138 = poly_5 * poly_25 - poly_86 
    poly_139 = poly_4 * poly_27 - poly_79 
    poly_140 = poly_5 * poly_28 - poly_86 
    poly_141 = poly_1 * poly_40 
    poly_142 = poly_2 * poly_40 
    poly_143 = poly_3 * poly_40 
    poly_144 = poly_6 * poly_24 - poly_81 
    poly_145 = poly_6 * poly_25 - poly_87 
    poly_146 = poly_26 * poly_6 
    poly_147 = poly_4 * poly_40 - poly_144 
    poly_148 = poly_5 * poly_40 - poly_145 
    poly_149 = poly_1 * poly_29 
    poly_150 = poly_2 * poly_30 
    poly_151 = poly_3 * poly_31 
    poly_152 = poly_7 * poly_38 
    poly_153 = poly_7 * poly_39 
    poly_154 = poly_7 * poly_40 
    poly_155 = poly_1 * poly_41 
    poly_156 = poly_2 * poly_41 
    poly_157 = poly_3 * poly_41 
    poly_158 = poly_7 * poly_32 
    poly_159 = poly_7 * poly_33 
    poly_160 = poly_7 * poly_34 
    poly_161 = poly_1 * poly_35 
    poly_162 = poly_2 * poly_36 
    poly_163 = poly_3 * poly_37 
    poly_164 = poly_4 * poly_38 - poly_122 
    poly_165 = poly_5 * poly_39 - poly_132 
    poly_166 = poly_6 * poly_40 - poly_146 
    poly_167 = poly_7 * poly_41 
    poly_168 = poly_1 * poly_45 
    poly_169 = poly_1 * poly_47 
    poly_170 = poly_1 * poly_48 
    poly_171 = poly_2 * poly_48 
    poly_172 = poly_1 * poly_51 
    poly_173 = poly_1 * poly_53 
    poly_174 = poly_1 * poly_54 
    poly_175 = poly_2 * poly_54 
    poly_176 = poly_1 * poly_56 
    poly_177 = poly_1 * poly_57 
    poly_178 = poly_2 * poly_57 
    poly_179 = poly_1 * poly_59 
    poly_180 = poly_1 * poly_60 
    poly_181 = poly_2 * poly_60 
    poly_182 = poly_1 * poly_61 
    poly_183 = poly_2 * poly_61 
    poly_184 = poly_3 * poly_61 
    poly_185 = poly_1 * poly_62 
    poly_186 = poly_2 * poly_62 
    poly_187 = poly_3 * poly_62 
    poly_188 = poly_14 * poly_19 
    poly_189 = poly_1 * poly_65 
    poly_190 = poly_1 * poly_67 
    poly_191 = poly_1 * poly_68 
    poly_192 = poly_2 * poly_68 
    poly_193 = poly_1 * poly_70 
    poly_194 = poly_1 * poly_71 
    poly_195 = poly_2 * poly_71 
    poly_196 = poly_1 * poly_72 
    poly_197 = poly_2 * poly_72 
    poly_198 = poly_3 * poly_72 
    poly_199 = poly_1 * poly_74 
    poly_200 = poly_1 * poly_75 
    poly_201 = poly_2 * poly_75 
    poly_202 = poly_1 * poly_77 
    poly_203 = poly_1 * poly_78 
    poly_204 = poly_2 * poly_78 
    poly_205 = poly_1 * poly_79 
    poly_206 = poly_2 * poly_79 
    poly_207 = poly_3 * poly_79 
    poly_208 = poly_1 * poly_80 
    poly_209 = poly_2 * poly_80 
    poly_210 = poly_3 * poly_80 
    poly_211 = poly_14 * poly_25 
    poly_212 = poly_1 * poly_81 
    poly_213 = poly_2 * poly_81 
    poly_214 = poly_3 * poly_81 
    poly_215 = poly_14 * poly_26 
    poly_216 = poly_1 * poly_83 
    poly_217 = poly_1 * poly_84 
    poly_218 = poly_2 * poly_84 
    poly_219 = poly_1 * poly_85 
    poly_220 = poly_2 * poly_85 
    poly_221 = poly_3 * poly_85 
    poly_222 = poly_1 * poly_86 
    poly_223 = poly_2 * poly_86 
    poly_224 = poly_3 * poly_86 
    poly_225 = poly_19 * poly_24 
    poly_226 = poly_1 * poly_87 
    poly_227 = poly_2 * poly_87 
    poly_228 = poly_3 * poly_87 
    poly_229 = poly_26 * poly_18 
    poly_230 = poly_19 * poly_26 
    poly_231 = poly_1 * poly_88 
    poly_232 = poly_2 * poly_88 
    poly_233 = poly_3 * poly_88 
    poly_234 = poly_14 * poly_28 
    poly_235 = poly_19 * poly_27 
    poly_236 = poly_26 * poly_20 
    poly_237 = poly_1 * poly_91 
    poly_238 = poly_1 * poly_93 
    poly_239 = poly_1 * poly_94 
    poly_240 = poly_2 * poly_94 
    poly_241 = poly_1 * poly_95 
    poly_242 = poly_2 * poly_95 
    poly_243 = poly_3 * poly_95 
    poly_244 = poly_1 * poly_97 
    poly_245 = poly_1 * poly_98 
    poly_246 = poly_2 * poly_98 
    poly_247 = poly_1 * poly_99 
    poly_248 = poly_2 * poly_99 
    poly_249 = poly_3 * poly_99 
    poly_250 = poly_1 * poly_100 
    poly_251 = poly_2 * poly_100 
    poly_252 = poly_3 * poly_100 
    poly_253 = poly_1 * poly_101 
    poly_254 = poly_2 * poly_101 
    poly_255 = poly_3 * poly_101 
    poly_256 = poly_7 * poly_61 
    poly_257 = poly_7 * poly_62 
    poly_258 = poly_1 * poly_103 
    poly_259 = poly_1 * poly_104 
    poly_260 = poly_2 * poly_104 
    poly_261 = poly_1 * poly_105 
    poly_262 = poly_2 * poly_105 
    poly_263 = poly_3 * poly_105 
    poly_264 = poly_1 * poly_106 
    poly_265 = poly_2 * poly_106 
    poly_266 = poly_3 * poly_106 
    poly_267 = poly_7 * poly_72 
    poly_268 = poly_1 * poly_107 
    poly_269 = poly_2 * poly_107 
    poly_270 = poly_3 * poly_107 
    poly_271 = poly_1 * poly_108 
    poly_272 = poly_2 * poly_108 
    poly_273 = poly_3 * poly_108 
    poly_274 = poly_7 * poly_79 
    poly_275 = poly_7 * poly_80 
    poly_276 = poly_7 * poly_81 
    poly_277 = poly_1 * poly_109 
    poly_278 = poly_2 * poly_109 
    poly_279 = poly_3 * poly_109 
    poly_280 = poly_7 * poly_85 
    poly_281 = poly_7 * poly_86 
    poly_282 = poly_7 * poly_87 
    poly_283 = poly_7 * poly_88 
    poly_284 = poly_1 * poly_42 
    poly_285 = poly_1 * poly_113 
    poly_286 = poly_1 * poly_115 
    poly_287 = poly_1 * poly_43 
    poly_288 = poly_1 * poly_117 
    poly_289 = poly_1 * poly_44 
    poly_290 = poly_2 * poly_45 
    poly_291 = poly_1 * poly_118 
    poly_292 = poly_2 * poly_118 
    poly_293 = poly_1 * poly_46 
    poly_294 = poly_2 * poly_47 
    poly_295 = poly_3 * poly_48 
    poly_296 = poly_1 * poly_120 
    poly_297 = poly_1 * poly_121 
    poly_298 = poly_2 * poly_121 
    poly_299 = poly_1 * poly_122 
    poly_300 = poly_2 * poly_122 
    poly_301 = poly_3 * poly_122 
    poly_302 = poly_1 * poly_49 
    poly_303 = poly_1 * poly_124 
    poly_304 = poly_1 * poly_50 
    poly_305 = poly_2 * poly_51 
    poly_306 = poly_1 * poly_125 
    poly_307 = poly_2 * poly_125 
    poly_308 = poly_1 * poly_52 
    poly_309 = poly_2 * poly_53 
    poly_310 = poly_3 * poly_54 
    poly_311 = poly_1 * poly_126 
    poly_312 = poly_2 * poly_126 
    poly_313 = poly_3 * poly_126 
    poly_314 = poly_1 * poly_55 
    poly_315 = poly_2 * poly_56 
    poly_316 = poly_3 * poly_57 
    poly_317 = poly_1 * poly_58 
    poly_318 = poly_2 * poly_59 
    poly_319 = poly_3 * poly_60 
    poly_320 = poly_14 * poly_18 
    poly_321 = poly_1 * poly_127 
    poly_322 = poly_2 * poly_127 
    poly_323 = poly_3 * poly_127 
    poly_324 = poly_14 * poly_20 
    poly_325 = poly_19 * poly_38 
    poly_326 = poly_1 * poly_129 
    poly_327 = poly_1 * poly_130 
    poly_328 = poly_2 * poly_130 
    poly_329 = poly_1 * poly_131 
    poly_330 = poly_2 * poly_131 
    poly_331 = poly_3 * poly_131 
    poly_332 = poly_1 * poly_132 
    poly_333 = poly_2 * poly_132 
    poly_334 = poly_3 * poly_132 
    poly_335 = poly_19 * poly_18 
    poly_336 = poly_1 * poly_133 
    poly_337 = poly_2 * poly_133 
    poly_338 = poly_3 * poly_133 
    poly_339 = poly_14 * poly_39 
    poly_340 = poly_19 * poly_20 
    poly_341 = poly_1 * poly_63 
    poly_342 = poly_1 * poly_135 
    poly_343 = poly_1 * poly_64 
    poly_344 = poly_2 * poly_65 
    poly_345 = poly_1 * poly_136 
    poly_346 = poly_2 * poly_136 
    poly_347 = poly_1 * poly_66 
    poly_348 = poly_2 * poly_67 
    poly_349 = poly_3 * poly_68 
    poly_350 = poly_1 * poly_137 
    poly_351 = poly_2 * poly_137 
    poly_352 = poly_3 * poly_137 
    poly_353 = poly_1 * poly_69 
    poly_354 = poly_2 * poly_70 
    poly_355 = poly_3 * poly_71 
    poly_356 = poly_4 * poly_72 - poly_211 
    poly_357 = poly_1 * poly_138 
    poly_358 = poly_2 * poly_138 
    poly_359 = poly_3 * poly_138 
    poly_360 = poly_5 * poly_72 - poly_225 
    poly_361 = poly_1 * poly_73 
    poly_362 = poly_2 * poly_74 
    poly_363 = poly_3 * poly_75 
    poly_364 = poly_1 * poly_76 
    poly_365 = poly_2 * poly_77 
    poly_366 = poly_3 * poly_78 
    poly_367 = poly_14 * poly_24 
    poly_368 = poly_4 * poly_138 - poly_360 
    poly_369 = poly_1 * poly_139 
    poly_370 = poly_2 * poly_139 
    poly_371 = poly_3 * poly_139 
    poly_372 = poly_14 * poly_27 
    poly_373 = poly_4 * poly_80 - poly_211 
    poly_374 = poly_26 * poly_38 
    poly_375 = poly_1 * poly_82 
    poly_376 = poly_2 * poly_83 
    poly_377 = poly_3 * poly_84 
    poly_378 = poly_4 * poly_85 - poly_234 
    poly_379 = poly_19 * poly_25 
    poly_380 = poly_4 * poly_88 - poly_234 
    poly_381 = poly_1 * poly_140 
    poly_382 = poly_2 * poly_140 
    poly_383 = poly_3 * poly_140 
    poly_384 = poly_5 * poly_85 - poly_225 
    poly_385 = poly_19 * poly_28 
    poly_386 = poly_26 * poly_39 
    poly_387 = poly_4 * poly_140 - poly_384 
    poly_388 = poly_1 * poly_142 
    poly_389 = poly_1 * poly_143 
    poly_390 = poly_2 * poly_143 
    poly_391 = poly_1 * poly_144 
    poly_392 = poly_2 * poly_144 
    poly_393 = poly_3 * poly_144 
    poly_394 = poly_1 * poly_145 
    poly_395 = poly_2 * poly_145 
    poly_396 = poly_3 * poly_145 
    poly_397 = poly_6 * poly_72 - poly_236 
    poly_398 = poly_1 * poly_146 
    poly_399 = poly_2 * poly_146 
    poly_400 = poly_3 * poly_146 
    poly_401 = poly_26 * poly_24 
    poly_402 = poly_26 * poly_25 
    poly_403 = poly_1 * poly_147 
    poly_404 = poly_2 * poly_147 
    poly_405 = poly_3 * poly_147 
    poly_406 = poly_14 * poly_40 
    poly_407 = poly_4 * poly_145 - poly_397 
    poly_408 = poly_26 * poly_27 
    poly_409 = poly_1 * poly_148 
    poly_410 = poly_2 * poly_148 
    poly_411 = poly_3 * poly_148 
    poly_412 = poly_5 * poly_144 - poly_397 
    poly_413 = poly_19 * poly_40 
    poly_414 = poly_26 * poly_28 
    poly_415 = poly_4 * poly_148 - poly_412 
    poly_416 = poly_1 * poly_89 
    poly_417 = poly_1 * poly_150 
    poly_418 = poly_1 * poly_90 
    poly_419 = poly_2 * poly_91 
    poly_420 = poly_1 * poly_151 
    poly_421 = poly_2 * poly_151 
    poly_422 = poly_1 * poly_92 
    poly_423 = poly_2 * poly_93 
    poly_424 = poly_3 * poly_94 
    poly_425 = poly_1 * poly_152 
    poly_426 = poly_2 * poly_152 
    poly_427 = poly_3 * poly_152 
    poly_428 = poly_7 * poly_122 
    poly_429 = poly_1 * poly_96 
    poly_430 = poly_2 * poly_97 
    poly_431 = poly_3 * poly_98 
    poly_432 = poly_7 * poly_126 
    poly_433 = poly_7 * poly_127 
    poly_434 = poly_1 * poly_153 
    poly_435 = poly_2 * poly_153 
    poly_436 = poly_3 * poly_153 
    poly_437 = poly_7 * poly_131 
    poly_438 = poly_7 * poly_132 
    poly_439 = poly_7 * poly_133 
    poly_440 = poly_1 * poly_102 
    poly_441 = poly_2 * poly_103 
    poly_442 = poly_3 * poly_104 
    poly_443 = poly_7 * poly_137 
    poly_444 = poly_7 * poly_138 
    poly_445 = poly_7 * poly_139 
    poly_446 = poly_7 * poly_140 
    poly_447 = poly_1 * poly_154 
    poly_448 = poly_2 * poly_154 
    poly_449 = poly_3 * poly_154 
    poly_450 = poly_7 * poly_144 
    poly_451 = poly_7 * poly_145 
    poly_452 = poly_7 * poly_146 
    poly_453 = poly_7 * poly_147 
    poly_454 = poly_7 * poly_148 
    poly_455 = poly_1 * poly_156 
    poly_456 = poly_1 * poly_157 
    poly_457 = poly_2 * poly_157 
    poly_458 = poly_1 * poly_158 
    poly_459 = poly_2 * poly_158 
    poly_460 = poly_3 * poly_158 
    poly_461 = poly_7 * poly_95 
    poly_462 = poly_1 * poly_159 
    poly_463 = poly_2 * poly_159 
    poly_464 = poly_3 * poly_159 
    poly_465 = poly_7 * poly_99 
    poly_466 = poly_7 * poly_100 
    poly_467 = poly_7 * poly_101 
    poly_468 = poly_1 * poly_160 
    poly_469 = poly_2 * poly_160 
    poly_470 = poly_3 * poly_160 
    poly_471 = poly_7 * poly_105 
    poly_472 = poly_7 * poly_106 
    poly_473 = poly_7 * poly_107 
    poly_474 = poly_7 * poly_108 
    poly_475 = poly_7 * poly_109 
    poly_476 = poly_1 * poly_110 
    poly_477 = poly_1 * poly_111 
    poly_478 = poly_1 * poly_162 
    poly_479 = poly_1 * poly_112 
    poly_480 = poly_2 * poly_113 
    poly_481 = poly_1 * poly_114 
    poly_482 = poly_2 * poly_115 
    poly_483 = poly_1 * poly_163 
    poly_484 = poly_2 * poly_163 
    poly_485 = poly_1 * poly_116 
    poly_486 = poly_2 * poly_117 
    poly_487 = poly_3 * poly_118 
    poly_488 = poly_1 * poly_119 
    poly_489 = poly_2 * poly_120 
    poly_490 = poly_3 * poly_121 
    poly_491 = poly_14 * poly_14 
    poly_492 = poly_1 * poly_164 
    poly_493 = poly_2 * poly_164 
    poly_494 = poly_3 * poly_164 
    poly_495 = poly_14 * poly_38 
    poly_496 = poly_1 * poly_123 
    poly_497 = poly_2 * poly_124 
    poly_498 = poly_3 * poly_125 
    poly_499 = poly_4 * poly_126 - poly_320 
    poly_500 = poly_4 * poly_127 - poly_324 
    poly_501 = poly_1 * poly_128 
    poly_502 = poly_2 * poly_129 
    poly_503 = poly_3 * poly_130 
    poly_504 = poly_4 * poly_131 - poly_339 
    poly_505 = poly_19 * poly_19 
    poly_506 = poly_4 * poly_133 - poly_339 
    poly_507 = poly_1 * poly_165 
    poly_508 = poly_2 * poly_165 
    poly_509 = poly_3 * poly_165 
    poly_510 = poly_5 * poly_131 - poly_335 
    poly_511 = poly_19 * poly_39 
    poly_512 = poly_4 * poly_165 - poly_510 
    poly_513 = poly_1 * poly_134 
    poly_514 = poly_2 * poly_135 
    poly_515 = poly_3 * poly_136 
    poly_516 = poly_4 * poly_137 - poly_367 
    poly_517 = poly_5 * poly_138 - poly_379 
    poly_518 = poly_4 * poly_139 - poly_372 
    poly_519 = poly_5 * poly_140 - poly_385 
    poly_520 = poly_1 * poly_141 
    poly_521 = poly_2 * poly_142 
    poly_522 = poly_3 * poly_143 
    poly_523 = poly_4 * poly_144 - poly_406 
    poly_524 = poly_5 * poly_145 - poly_413 
    poly_525 = poly_26 * poly_26 
    poly_526 = poly_4 * poly_147 - poly_406 
    poly_527 = poly_5 * poly_148 - poly_413 
    poly_528 = poly_1 * poly_166 
    poly_529 = poly_2 * poly_166 
    poly_530 = poly_3 * poly_166 
    poly_531 = poly_6 * poly_144 - poly_401 
    poly_532 = poly_6 * poly_145 - poly_402 
    poly_533 = poly_26 * poly_40 
    poly_534 = poly_4 * poly_166 - poly_531 
    poly_535 = poly_5 * poly_166 - poly_532 
    poly_536 = poly_1 * poly_149 
    poly_537 = poly_2 * poly_150 
    poly_538 = poly_3 * poly_151 
    poly_539 = poly_7 * poly_164 
    poly_540 = poly_7 * poly_165 
    poly_541 = poly_7 * poly_166 
    poly_542 = poly_1 * poly_155 
    poly_543 = poly_2 * poly_156 
    poly_544 = poly_3 * poly_157 
    poly_545 = poly_7 * poly_152 
    poly_546 = poly_7 * poly_153 
    poly_547 = poly_7 * poly_154 
    poly_548 = poly_1 * poly_167 
    poly_549 = poly_2 * poly_167 
    poly_550 = poly_3 * poly_167 
    poly_551 = poly_7 * poly_158 
    poly_552 = poly_7 * poly_159 
    poly_553 = poly_7 * poly_160 
    poly_554 = poly_1 * poly_161 
    poly_555 = poly_2 * poly_162 
    poly_556 = poly_3 * poly_163 
    poly_557 = poly_4 * poly_164 - poly_495 
    poly_558 = poly_5 * poly_165 - poly_511 
    poly_559 = poly_6 * poly_166 - poly_533 
    poly_560 = poly_7 * poly_167 

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
    ]) 

    return poly 



