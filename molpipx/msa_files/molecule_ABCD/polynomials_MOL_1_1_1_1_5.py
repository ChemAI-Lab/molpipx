import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABCD.monomials_MOL_1_1_1_1_5 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_1_5.POLY 

N_POLYS = 462

# Total number of monomials = 462 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(462) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = jnp.take(mono,4) 
    poly_5 = jnp.take(mono,5) 
    poly_6 = jnp.take(mono,6) 
    poly_7 = poly_1 * poly_2 
    poly_8 = poly_1 * poly_3 
    poly_9 = poly_2 * poly_3 
    poly_10 = poly_1 * poly_4 
    poly_11 = poly_2 * poly_4 
    poly_12 = poly_3 * poly_4 
    poly_13 = poly_1 * poly_5 
    poly_14 = poly_2 * poly_5 
    poly_15 = poly_3 * poly_5 
    poly_16 = poly_4 * poly_5 
    poly_17 = poly_1 * poly_6 
    poly_18 = poly_2 * poly_6 
    poly_19 = poly_3 * poly_6 
    poly_20 = poly_4 * poly_6 
    poly_21 = poly_5 * poly_6 
    poly_22 = poly_1 * poly_1 
    poly_23 = poly_2 * poly_2 
    poly_24 = poly_3 * poly_3 
    poly_25 = poly_4 * poly_4 
    poly_26 = poly_5 * poly_5 
    poly_27 = poly_6 * poly_6 
    poly_28 = poly_1 * poly_9 
    poly_29 = poly_1 * poly_11 
    poly_30 = poly_1 * poly_12 
    poly_31 = poly_2 * poly_12 
    poly_32 = poly_1 * poly_14 
    poly_33 = poly_1 * poly_15 
    poly_34 = poly_2 * poly_15 
    poly_35 = poly_1 * poly_16 
    poly_36 = poly_2 * poly_16 
    poly_37 = poly_3 * poly_16 
    poly_38 = poly_1 * poly_18 
    poly_39 = poly_1 * poly_19 
    poly_40 = poly_2 * poly_19 
    poly_41 = poly_1 * poly_20 
    poly_42 = poly_2 * poly_20 
    poly_43 = poly_3 * poly_20 
    poly_44 = poly_1 * poly_21 
    poly_45 = poly_2 * poly_21 
    poly_46 = poly_3 * poly_21 
    poly_47 = poly_4 * poly_21 
    poly_48 = poly_1 * poly_7 
    poly_49 = poly_1 * poly_23 
    poly_50 = poly_1 * poly_8 
    poly_51 = poly_2 * poly_9 
    poly_52 = poly_1 * poly_24 
    poly_53 = poly_2 * poly_24 
    poly_54 = poly_1 * poly_10 
    poly_55 = poly_2 * poly_11 
    poly_56 = poly_3 * poly_12 
    poly_57 = poly_1 * poly_25 
    poly_58 = poly_2 * poly_25 
    poly_59 = poly_3 * poly_25 
    poly_60 = poly_1 * poly_13 
    poly_61 = poly_2 * poly_14 
    poly_62 = poly_3 * poly_15 
    poly_63 = poly_4 * poly_16 
    poly_64 = poly_1 * poly_26 
    poly_65 = poly_2 * poly_26 
    poly_66 = poly_3 * poly_26 
    poly_67 = poly_4 * poly_26 
    poly_68 = poly_1 * poly_17 
    poly_69 = poly_2 * poly_18 
    poly_70 = poly_3 * poly_19 
    poly_71 = poly_4 * poly_20 
    poly_72 = poly_5 * poly_21 
    poly_73 = poly_1 * poly_27 
    poly_74 = poly_2 * poly_27 
    poly_75 = poly_3 * poly_27 
    poly_76 = poly_4 * poly_27 
    poly_77 = poly_5 * poly_27 
    poly_78 = poly_1 * poly_22 
    poly_79 = poly_2 * poly_23 
    poly_80 = poly_3 * poly_24 
    poly_81 = poly_4 * poly_25 
    poly_82 = poly_5 * poly_26 
    poly_83 = poly_6 * poly_27 
    poly_84 = poly_1 * poly_31 
    poly_85 = poly_1 * poly_34 
    poly_86 = poly_1 * poly_36 
    poly_87 = poly_1 * poly_37 
    poly_88 = poly_2 * poly_37 
    poly_89 = poly_1 * poly_40 
    poly_90 = poly_1 * poly_42 
    poly_91 = poly_1 * poly_43 
    poly_92 = poly_2 * poly_43 
    poly_93 = poly_1 * poly_45 
    poly_94 = poly_1 * poly_46 
    poly_95 = poly_2 * poly_46 
    poly_96 = poly_1 * poly_47 
    poly_97 = poly_2 * poly_47 
    poly_98 = poly_3 * poly_47 
    poly_99 = poly_1 * poly_28 
    poly_100 = poly_1 * poly_51 
    poly_101 = poly_1 * poly_53 
    poly_102 = poly_1 * poly_29 
    poly_103 = poly_1 * poly_55 
    poly_104 = poly_1 * poly_30 
    poly_105 = poly_2 * poly_31 
    poly_106 = poly_1 * poly_56 
    poly_107 = poly_2 * poly_56 
    poly_108 = poly_1 * poly_58 
    poly_109 = poly_1 * poly_59 
    poly_110 = poly_2 * poly_59 
    poly_111 = poly_1 * poly_32 
    poly_112 = poly_1 * poly_61 
    poly_113 = poly_1 * poly_33 
    poly_114 = poly_2 * poly_34 
    poly_115 = poly_1 * poly_62 
    poly_116 = poly_2 * poly_62 
    poly_117 = poly_1 * poly_35 
    poly_118 = poly_2 * poly_36 
    poly_119 = poly_3 * poly_37 
    poly_120 = poly_1 * poly_63 
    poly_121 = poly_2 * poly_63 
    poly_122 = poly_3 * poly_63 
    poly_123 = poly_1 * poly_65 
    poly_124 = poly_1 * poly_66 
    poly_125 = poly_2 * poly_66 
    poly_126 = poly_1 * poly_67 
    poly_127 = poly_2 * poly_67 
    poly_128 = poly_3 * poly_67 
    poly_129 = poly_1 * poly_38 
    poly_130 = poly_1 * poly_69 
    poly_131 = poly_1 * poly_39 
    poly_132 = poly_2 * poly_40 
    poly_133 = poly_1 * poly_70 
    poly_134 = poly_2 * poly_70 
    poly_135 = poly_1 * poly_41 
    poly_136 = poly_2 * poly_42 
    poly_137 = poly_3 * poly_43 
    poly_138 = poly_1 * poly_71 
    poly_139 = poly_2 * poly_71 
    poly_140 = poly_3 * poly_71 
    poly_141 = poly_1 * poly_44 
    poly_142 = poly_2 * poly_45 
    poly_143 = poly_3 * poly_46 
    poly_144 = poly_4 * poly_47 
    poly_145 = poly_1 * poly_72 
    poly_146 = poly_2 * poly_72 
    poly_147 = poly_3 * poly_72 
    poly_148 = poly_4 * poly_72 
    poly_149 = poly_1 * poly_74 
    poly_150 = poly_1 * poly_75 
    poly_151 = poly_2 * poly_75 
    poly_152 = poly_1 * poly_76 
    poly_153 = poly_2 * poly_76 
    poly_154 = poly_3 * poly_76 
    poly_155 = poly_1 * poly_77 
    poly_156 = poly_2 * poly_77 
    poly_157 = poly_3 * poly_77 
    poly_158 = poly_4 * poly_77 
    poly_159 = poly_1 * poly_48 
    poly_160 = poly_1 * poly_49 
    poly_161 = poly_1 * poly_79 
    poly_162 = poly_1 * poly_50 
    poly_163 = poly_2 * poly_51 
    poly_164 = poly_1 * poly_52 
    poly_165 = poly_2 * poly_53 
    poly_166 = poly_1 * poly_80 
    poly_167 = poly_2 * poly_80 
    poly_168 = poly_1 * poly_54 
    poly_169 = poly_2 * poly_55 
    poly_170 = poly_3 * poly_56 
    poly_171 = poly_1 * poly_57 
    poly_172 = poly_2 * poly_58 
    poly_173 = poly_3 * poly_59 
    poly_174 = poly_1 * poly_81 
    poly_175 = poly_2 * poly_81 
    poly_176 = poly_3 * poly_81 
    poly_177 = poly_1 * poly_60 
    poly_178 = poly_2 * poly_61 
    poly_179 = poly_3 * poly_62 
    poly_180 = poly_4 * poly_63 
    poly_181 = poly_1 * poly_64 
    poly_182 = poly_2 * poly_65 
    poly_183 = poly_3 * poly_66 
    poly_184 = poly_4 * poly_67 
    poly_185 = poly_1 * poly_82 
    poly_186 = poly_2 * poly_82 
    poly_187 = poly_3 * poly_82 
    poly_188 = poly_4 * poly_82 
    poly_189 = poly_1 * poly_68 
    poly_190 = poly_2 * poly_69 
    poly_191 = poly_3 * poly_70 
    poly_192 = poly_4 * poly_71 
    poly_193 = poly_5 * poly_72 
    poly_194 = poly_1 * poly_73 
    poly_195 = poly_2 * poly_74 
    poly_196 = poly_3 * poly_75 
    poly_197 = poly_4 * poly_76 
    poly_198 = poly_5 * poly_77 
    poly_199 = poly_1 * poly_83 
    poly_200 = poly_2 * poly_83 
    poly_201 = poly_3 * poly_83 
    poly_202 = poly_4 * poly_83 
    poly_203 = poly_5 * poly_83 
    poly_204 = poly_1 * poly_78 
    poly_205 = poly_2 * poly_79 
    poly_206 = poly_3 * poly_80 
    poly_207 = poly_4 * poly_81 
    poly_208 = poly_5 * poly_82 
    poly_209 = poly_6 * poly_83 
    poly_210 = poly_1 * poly_88 
    poly_211 = poly_1 * poly_92 
    poly_212 = poly_1 * poly_95 
    poly_213 = poly_1 * poly_97 
    poly_214 = poly_1 * poly_98 
    poly_215 = poly_2 * poly_98 
    poly_216 = poly_1 * poly_84 
    poly_217 = poly_1 * poly_105 
    poly_218 = poly_1 * poly_107 
    poly_219 = poly_1 * poly_110 
    poly_220 = poly_1 * poly_85 
    poly_221 = poly_1 * poly_114 
    poly_222 = poly_1 * poly_116 
    poly_223 = poly_1 * poly_86 
    poly_224 = poly_1 * poly_118 
    poly_225 = poly_1 * poly_87 
    poly_226 = poly_2 * poly_88 
    poly_227 = poly_1 * poly_119 
    poly_228 = poly_2 * poly_119 
    poly_229 = poly_1 * poly_121 
    poly_230 = poly_1 * poly_122 
    poly_231 = poly_2 * poly_122 
    poly_232 = poly_1 * poly_125 
    poly_233 = poly_1 * poly_127 
    poly_234 = poly_1 * poly_128 
    poly_235 = poly_2 * poly_128 
    poly_236 = poly_1 * poly_89 
    poly_237 = poly_1 * poly_132 
    poly_238 = poly_1 * poly_134 
    poly_239 = poly_1 * poly_90 
    poly_240 = poly_1 * poly_136 
    poly_241 = poly_1 * poly_91 
    poly_242 = poly_2 * poly_92 
    poly_243 = poly_1 * poly_137 
    poly_244 = poly_2 * poly_137 
    poly_245 = poly_1 * poly_139 
    poly_246 = poly_1 * poly_140 
    poly_247 = poly_2 * poly_140 
    poly_248 = poly_1 * poly_93 
    poly_249 = poly_1 * poly_142 
    poly_250 = poly_1 * poly_94 
    poly_251 = poly_2 * poly_95 
    poly_252 = poly_1 * poly_143 
    poly_253 = poly_2 * poly_143 
    poly_254 = poly_1 * poly_96 
    poly_255 = poly_2 * poly_97 
    poly_256 = poly_3 * poly_98 
    poly_257 = poly_1 * poly_144 
    poly_258 = poly_2 * poly_144 
    poly_259 = poly_3 * poly_144 
    poly_260 = poly_1 * poly_146 
    poly_261 = poly_1 * poly_147 
    poly_262 = poly_2 * poly_147 
    poly_263 = poly_1 * poly_148 
    poly_264 = poly_2 * poly_148 
    poly_265 = poly_3 * poly_148 
    poly_266 = poly_1 * poly_151 
    poly_267 = poly_1 * poly_153 
    poly_268 = poly_1 * poly_154 
    poly_269 = poly_2 * poly_154 
    poly_270 = poly_1 * poly_156 
    poly_271 = poly_1 * poly_157 
    poly_272 = poly_2 * poly_157 
    poly_273 = poly_1 * poly_158 
    poly_274 = poly_2 * poly_158 
    poly_275 = poly_3 * poly_158 
    poly_276 = poly_1 * poly_99 
    poly_277 = poly_1 * poly_100 
    poly_278 = poly_1 * poly_163 
    poly_279 = poly_1 * poly_101 
    poly_280 = poly_1 * poly_165 
    poly_281 = poly_1 * poly_167 
    poly_282 = poly_1 * poly_102 
    poly_283 = poly_1 * poly_103 
    poly_284 = poly_1 * poly_169 
    poly_285 = poly_1 * poly_104 
    poly_286 = poly_2 * poly_105 
    poly_287 = poly_1 * poly_106 
    poly_288 = poly_2 * poly_107 
    poly_289 = poly_1 * poly_170 
    poly_290 = poly_2 * poly_170 
    poly_291 = poly_1 * poly_108 
    poly_292 = poly_1 * poly_172 
    poly_293 = poly_1 * poly_109 
    poly_294 = poly_2 * poly_110 
    poly_295 = poly_1 * poly_173 
    poly_296 = poly_2 * poly_173 
    poly_297 = poly_1 * poly_175 
    poly_298 = poly_1 * poly_176 
    poly_299 = poly_2 * poly_176 
    poly_300 = poly_1 * poly_111 
    poly_301 = poly_1 * poly_112 
    poly_302 = poly_1 * poly_178 
    poly_303 = poly_1 * poly_113 
    poly_304 = poly_2 * poly_114 
    poly_305 = poly_1 * poly_115 
    poly_306 = poly_2 * poly_116 
    poly_307 = poly_1 * poly_179 
    poly_308 = poly_2 * poly_179 
    poly_309 = poly_1 * poly_117 
    poly_310 = poly_2 * poly_118 
    poly_311 = poly_3 * poly_119 
    poly_312 = poly_1 * poly_120 
    poly_313 = poly_2 * poly_121 
    poly_314 = poly_3 * poly_122 
    poly_315 = poly_1 * poly_180 
    poly_316 = poly_2 * poly_180 
    poly_317 = poly_3 * poly_180 
    poly_318 = poly_1 * poly_123 
    poly_319 = poly_1 * poly_182 
    poly_320 = poly_1 * poly_124 
    poly_321 = poly_2 * poly_125 
    poly_322 = poly_1 * poly_183 
    poly_323 = poly_2 * poly_183 
    poly_324 = poly_1 * poly_126 
    poly_325 = poly_2 * poly_127 
    poly_326 = poly_3 * poly_128 
    poly_327 = poly_1 * poly_184 
    poly_328 = poly_2 * poly_184 
    poly_329 = poly_3 * poly_184 
    poly_330 = poly_1 * poly_186 
    poly_331 = poly_1 * poly_187 
    poly_332 = poly_2 * poly_187 
    poly_333 = poly_1 * poly_188 
    poly_334 = poly_2 * poly_188 
    poly_335 = poly_3 * poly_188 
    poly_336 = poly_1 * poly_129 
    poly_337 = poly_1 * poly_130 
    poly_338 = poly_1 * poly_190 
    poly_339 = poly_1 * poly_131 
    poly_340 = poly_2 * poly_132 
    poly_341 = poly_1 * poly_133 
    poly_342 = poly_2 * poly_134 
    poly_343 = poly_1 * poly_191 
    poly_344 = poly_2 * poly_191 
    poly_345 = poly_1 * poly_135 
    poly_346 = poly_2 * poly_136 
    poly_347 = poly_3 * poly_137 
    poly_348 = poly_1 * poly_138 
    poly_349 = poly_2 * poly_139 
    poly_350 = poly_3 * poly_140 
    poly_351 = poly_1 * poly_192 
    poly_352 = poly_2 * poly_192 
    poly_353 = poly_3 * poly_192 
    poly_354 = poly_1 * poly_141 
    poly_355 = poly_2 * poly_142 
    poly_356 = poly_3 * poly_143 
    poly_357 = poly_4 * poly_144 
    poly_358 = poly_1 * poly_145 
    poly_359 = poly_2 * poly_146 
    poly_360 = poly_3 * poly_147 
    poly_361 = poly_4 * poly_148 
    poly_362 = poly_1 * poly_193 
    poly_363 = poly_2 * poly_193 
    poly_364 = poly_3 * poly_193 
    poly_365 = poly_4 * poly_193 
    poly_366 = poly_1 * poly_149 
    poly_367 = poly_1 * poly_195 
    poly_368 = poly_1 * poly_150 
    poly_369 = poly_2 * poly_151 
    poly_370 = poly_1 * poly_196 
    poly_371 = poly_2 * poly_196 
    poly_372 = poly_1 * poly_152 
    poly_373 = poly_2 * poly_153 
    poly_374 = poly_3 * poly_154 
    poly_375 = poly_1 * poly_197 
    poly_376 = poly_2 * poly_197 
    poly_377 = poly_3 * poly_197 
    poly_378 = poly_1 * poly_155 
    poly_379 = poly_2 * poly_156 
    poly_380 = poly_3 * poly_157 
    poly_381 = poly_4 * poly_158 
    poly_382 = poly_1 * poly_198 
    poly_383 = poly_2 * poly_198 
    poly_384 = poly_3 * poly_198 
    poly_385 = poly_4 * poly_198 
    poly_386 = poly_1 * poly_200 
    poly_387 = poly_1 * poly_201 
    poly_388 = poly_2 * poly_201 
    poly_389 = poly_1 * poly_202 
    poly_390 = poly_2 * poly_202 
    poly_391 = poly_3 * poly_202 
    poly_392 = poly_1 * poly_203 
    poly_393 = poly_2 * poly_203 
    poly_394 = poly_3 * poly_203 
    poly_395 = poly_4 * poly_203 
    poly_396 = poly_1 * poly_159 
    poly_397 = poly_1 * poly_160 
    poly_398 = poly_1 * poly_161 
    poly_399 = poly_1 * poly_205 
    poly_400 = poly_1 * poly_162 
    poly_401 = poly_2 * poly_163 
    poly_402 = poly_1 * poly_164 
    poly_403 = poly_2 * poly_165 
    poly_404 = poly_1 * poly_166 
    poly_405 = poly_2 * poly_167 
    poly_406 = poly_1 * poly_206 
    poly_407 = poly_2 * poly_206 
    poly_408 = poly_1 * poly_168 
    poly_409 = poly_2 * poly_169 
    poly_410 = poly_3 * poly_170 
    poly_411 = poly_1 * poly_171 
    poly_412 = poly_2 * poly_172 
    poly_413 = poly_3 * poly_173 
    poly_414 = poly_1 * poly_174 
    poly_415 = poly_2 * poly_175 
    poly_416 = poly_3 * poly_176 
    poly_417 = poly_1 * poly_207 
    poly_418 = poly_2 * poly_207 
    poly_419 = poly_3 * poly_207 
    poly_420 = poly_1 * poly_177 
    poly_421 = poly_2 * poly_178 
    poly_422 = poly_3 * poly_179 
    poly_423 = poly_4 * poly_180 
    poly_424 = poly_1 * poly_181 
    poly_425 = poly_2 * poly_182 
    poly_426 = poly_3 * poly_183 
    poly_427 = poly_4 * poly_184 
    poly_428 = poly_1 * poly_185 
    poly_429 = poly_2 * poly_186 
    poly_430 = poly_3 * poly_187 
    poly_431 = poly_4 * poly_188 
    poly_432 = poly_1 * poly_208 
    poly_433 = poly_2 * poly_208 
    poly_434 = poly_3 * poly_208 
    poly_435 = poly_4 * poly_208 
    poly_436 = poly_1 * poly_189 
    poly_437 = poly_2 * poly_190 
    poly_438 = poly_3 * poly_191 
    poly_439 = poly_4 * poly_192 
    poly_440 = poly_5 * poly_193 
    poly_441 = poly_1 * poly_194 
    poly_442 = poly_2 * poly_195 
    poly_443 = poly_3 * poly_196 
    poly_444 = poly_4 * poly_197 
    poly_445 = poly_5 * poly_198 
    poly_446 = poly_1 * poly_199 
    poly_447 = poly_2 * poly_200 
    poly_448 = poly_3 * poly_201 
    poly_449 = poly_4 * poly_202 
    poly_450 = poly_5 * poly_203 
    poly_451 = poly_1 * poly_209 
    poly_452 = poly_2 * poly_209 
    poly_453 = poly_3 * poly_209 
    poly_454 = poly_4 * poly_209 
    poly_455 = poly_5 * poly_209 
    poly_456 = poly_1 * poly_204 
    poly_457 = poly_2 * poly_205 
    poly_458 = poly_3 * poly_206 
    poly_459 = poly_4 * poly_207 
    poly_460 = poly_5 * poly_208 
    poly_461 = poly_6 * poly_209 

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
    poly_461,    ]) 

    return poly 


