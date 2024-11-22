import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3B.monomials_MOL_3_1_8 import f_monomials as f_monos 

# File created from ./MOL_3_1_8.POLY 

N_POLYS = 590

# Total number of monomials = 590 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(590) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) + jnp.take(mono,2) + jnp.take(mono,3) 
    poly_2 = jnp.take(mono,4) + jnp.take(mono,5) + jnp.take(mono,6) 
    poly_3 = jnp.take(mono,7) + jnp.take(mono,8) + jnp.take(mono,9) 
    poly_4 = jnp.take(mono,10) + jnp.take(mono,11) + jnp.take(mono,12) 
    poly_5 = poly_1 * poly_2 - poly_4 
    poly_6 = jnp.take(mono,13) + jnp.take(mono,14) + jnp.take(mono,15) 
    poly_7 = poly_1 * poly_1 - poly_3 - poly_3 
    poly_8 = poly_2 * poly_2 - poly_6 - poly_6 
    poly_9 = jnp.take(mono,16) 
    poly_10 = jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) + jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) 
    poly_11 = poly_2 * poly_3 - poly_10 
    poly_12 = jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) + jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) 
    poly_13 = jnp.take(mono,29) 
    poly_14 = poly_1 * poly_6 - poly_12 
    poly_15 = poly_1 * poly_3 - poly_9 - poly_9 - poly_9 
    poly_16 = poly_1 * poly_4 - poly_10 
    poly_17 = poly_2 * poly_7 - poly_16 
    poly_18 = poly_2 * poly_4 - poly_12 
    poly_19 = poly_1 * poly_8 - poly_18 
    poly_20 = poly_2 * poly_6 - poly_13 - poly_13 - poly_13 
    poly_21 = poly_1 * poly_7 - poly_15 
    poly_22 = poly_2 * poly_8 - poly_20 
    poly_23 = poly_9 * poly_2 
    poly_24 = jnp.take(mono,30) + jnp.take(mono,31) + jnp.take(mono,32) 
    poly_25 = poly_3 * poly_6 - poly_24 
    poly_26 = poly_13 * poly_1 
    poly_27 = poly_9 * poly_1 
    poly_28 = poly_3 * poly_4 - poly_23 
    poly_29 = poly_1 * poly_10 - poly_23 - poly_28 - poly_23 
    poly_30 = poly_1 * poly_11 - poly_23 
    poly_31 = poly_1 * poly_12 - poly_25 - poly_24 - poly_24 
    poly_32 = poly_1 * poly_14 - poly_25 
    poly_33 = poly_4 * poly_5 - poly_25 - poly_31 
    poly_34 = poly_2 * poly_11 - poly_25 
    poly_35 = poly_4 * poly_6 - poly_26 
    poly_36 = poly_2 * poly_12 - poly_26 - poly_35 - poly_26 
    poly_37 = poly_13 * poly_2 
    poly_38 = poly_2 * poly_14 - poly_26 
    poly_39 = poly_3 * poly_3 - poly_27 - poly_27 
    poly_40 = poly_3 * poly_7 - poly_27 
    poly_41 = poly_1 * poly_16 - poly_28 
    poly_42 = poly_2 * poly_21 - poly_41 
    poly_43 = poly_1 * poly_18 - poly_33 
    poly_44 = poly_7 * poly_8 - poly_43 
    poly_45 = poly_6 * poly_6 - poly_37 - poly_37 
    poly_46 = poly_2 * poly_18 - poly_35 
    poly_47 = poly_1 * poly_22 - poly_46 
    poly_48 = poly_6 * poly_8 - poly_37 
    poly_49 = poly_1 * poly_21 - poly_40 
    poly_50 = poly_2 * poly_22 - poly_48 
    poly_51 = poly_9 * poly_6 
    poly_52 = poly_13 * poly_3 
    poly_53 = poly_9 * poly_4 
    poly_54 = poly_9 * poly_5 
    poly_55 = poly_1 * poly_24 - poly_51 
    poly_56 = poly_3 * poly_12 - poly_51 - poly_55 - poly_51 
    poly_57 = poly_3 * poly_14 - poly_51 
    poly_58 = poly_13 * poly_7 
    poly_59 = poly_9 * poly_8 
    poly_60 = poly_2 * poly_24 - poly_52 
    poly_61 = poly_13 * poly_4 
    poly_62 = poly_4 * poly_14 - poly_58 
    poly_63 = poly_6 * poly_11 - poly_52 
    poly_64 = poly_13 * poly_5 
    poly_65 = poly_9 * poly_3 
    poly_66 = poly_9 * poly_7 
    poly_67 = poly_3 * poly_16 - poly_53 
    poly_68 = poly_4 * poly_15 - poly_54 - poly_67 
    poly_69 = poly_2 * poly_39 - poly_68 
    poly_70 = poly_1 * poly_29 - poly_54 - poly_68 
    poly_71 = poly_7 * poly_11 - poly_53 
    poly_72 = poly_1 * poly_31 - poly_56 - poly_55 
    poly_73 = poly_1 * poly_32 - poly_57 
    poly_74 = poly_3 * poly_18 - poly_59 
    poly_75 = poly_2 * poly_29 - poly_57 - poly_55 
    poly_76 = poly_1 * poly_34 - poly_59 
    poly_77 = poly_6 * poly_16 - poly_58 
    poly_78 = poly_1 * poly_36 - poly_63 - poly_60 
    poly_79 = poly_2 * poly_32 - poly_58 
    poly_80 = poly_6 * poly_12 - poly_64 - poly_61 - poly_61 
    poly_81 = poly_13 * poly_6 
    poly_82 = poly_1 * poly_45 - poly_80 
    poly_83 = poly_2 * poly_33 - poly_62 - poly_60 
    poly_84 = poly_2 * poly_34 - poly_63 
    poly_85 = poly_6 * poly_18 - poly_61 
    poly_86 = poly_2 * poly_36 - poly_64 - poly_80 
    poly_87 = poly_13 * poly_8 
    poly_88 = poly_8 * poly_14 - poly_61 
    poly_89 = poly_1 * poly_39 - poly_65 
    poly_90 = poly_3 * poly_21 - poly_66 
    poly_91 = poly_1 * poly_41 - poly_67 
    poly_92 = poly_2 * poly_49 - poly_91 
    poly_93 = poly_1 * poly_43 - poly_74 
    poly_94 = poly_8 * poly_21 - poly_93 
    poly_95 = poly_1 * poly_46 - poly_83 
    poly_96 = poly_7 * poly_22 - poly_95 
    poly_97 = poly_2 * poly_45 - poly_81 
    poly_98 = poly_2 * poly_46 - poly_85 
    poly_99 = poly_1 * poly_50 - poly_98 
    poly_100 = poly_6 * poly_22 - poly_87 
    poly_101 = poly_1 * poly_49 - poly_90 
    poly_102 = poly_2 * poly_50 - poly_100 
    poly_103 = poly_9 * poly_13 
    poly_104 = poly_9 * poly_12 
    poly_105 = poly_9 * poly_14 
    poly_106 = poly_13 * poly_15 
    poly_107 = poly_9 * poly_20 
    poly_108 = poly_13 * poly_10 
    poly_109 = poly_13 * poly_11 
    poly_110 = poly_9 * poly_16 
    poly_111 = poly_9 * poly_10 
    poly_112 = poly_9 * poly_11 
    poly_113 = poly_9 * poly_17 
    poly_114 = poly_3 * poly_24 - poly_104 
    poly_115 = poly_7 * poly_24 - poly_105 
    poly_116 = poly_3 * poly_31 - poly_104 - poly_115 
    poly_117 = poly_6 * poly_39 - poly_114 
    poly_118 = poly_3 * poly_32 - poly_105 
    poly_119 = poly_13 * poly_21 
    poly_120 = poly_9 * poly_18 
    poly_121 = poly_9 * poly_19 
    poly_122 = poly_4 * poly_24 - poly_103 - poly_103 - poly_103 
    poly_123 = poly_1 * poly_60 - poly_107 - poly_122 
    poly_124 = poly_13 * poly_16 
    poly_125 = poly_14 * poly_16 - poly_119 
    poly_126 = poly_2 * poly_56 - poly_106 - poly_125 
    poly_127 = poly_4 * poly_32 - poly_119 
    poly_128 = poly_1 * poly_63 - poly_107 - poly_126 
    poly_129 = poly_13 * poly_17 
    poly_130 = poly_6 * poly_24 - poly_108 
    poly_131 = poly_13 * poly_12 
    poly_132 = poly_3 * poly_45 - poly_130 
    poly_133 = poly_13 * poly_14 
    poly_134 = poly_9 * poly_22 
    poly_135 = poly_8 * poly_24 - poly_109 
    poly_136 = poly_13 * poly_18 
    poly_137 = poly_14 * poly_18 - poly_124 
    poly_138 = poly_6 * poly_34 - poly_109 
    poly_139 = poly_13 * poly_19 
    poly_140 = poly_9 * poly_9 
    poly_141 = poly_9 * poly_15 
    poly_142 = poly_9 * poly_21 
    poly_143 = poly_3 * poly_41 - poly_110 
    poly_144 = poly_4 * poly_39 - poly_112 
    poly_145 = poly_1 * poly_68 - poly_111 - poly_144 
    poly_146 = poly_1 * poly_69 - poly_112 
    poly_147 = poly_1 * poly_70 - poly_113 - poly_145 
    poly_148 = poly_11 * poly_21 - poly_110 
    poly_149 = poly_1 * poly_72 - poly_116 - poly_115 
    poly_150 = poly_1 * poly_73 - poly_118 
    poly_151 = poly_3 * poly_43 - poly_120 
    poly_152 = poly_15 * poly_18 - poly_121 - poly_151 
    poly_153 = poly_2 * poly_69 - poly_117 
    poly_154 = poly_1 * poly_75 - poly_121 - poly_152 
    poly_155 = poly_7 * poly_34 - poly_120 
    poly_156 = poly_6 * poly_41 - poly_119 
    poly_157 = poly_1 * poly_78 - poly_126 - poly_123 
    poly_158 = poly_2 * poly_73 - poly_119 
    poly_159 = poly_1 * poly_80 - poly_132 - poly_130 - poly_130 
    poly_160 = poly_13 * poly_13 
    poly_161 = poly_1 * poly_82 - poly_132 
    poly_162 = poly_3 * poly_46 - poly_134 
    poly_163 = poly_2 * poly_75 - poly_127 - poly_123 
    poly_164 = poly_1 * poly_84 - poly_134 
    poly_165 = poly_6 * poly_43 - poly_124 
    poly_166 = poly_1 * poly_86 - poly_138 - poly_135 
    poly_167 = poly_8 * poly_32 - poly_124 
    poly_168 = poly_4 * poly_45 - poly_133 
    poly_169 = poly_2 * poly_80 - poly_131 - poly_168 
    poly_170 = poly_13 * poly_20 
    poly_171 = poly_2 * poly_82 - poly_133 
    poly_172 = poly_2 * poly_83 - poly_137 - poly_135 
    poly_173 = poly_2 * poly_84 - poly_138 
    poly_174 = poly_6 * poly_46 - poly_136 
    poly_175 = poly_2 * poly_86 - poly_139 - poly_169 
    poly_176 = poly_13 * poly_22 
    poly_177 = poly_14 * poly_22 - poly_136 
    poly_178 = poly_3 * poly_39 - poly_141 
    poly_179 = poly_1 * poly_89 - poly_141 - poly_178 - poly_178 
    poly_180 = poly_3 * poly_49 - poly_142 
    poly_181 = poly_1 * poly_91 - poly_143 
    poly_182 = poly_2 * poly_101 - poly_181 
    poly_183 = poly_1 * poly_93 - poly_151 
    poly_184 = poly_8 * poly_49 - poly_183 
    poly_185 = poly_1 * poly_95 - poly_162 
    poly_186 = poly_21 * poly_22 - poly_185 
    poly_187 = poly_6 * poly_45 - poly_170 
    poly_188 = poly_1 * poly_98 - poly_172 
    poly_189 = poly_7 * poly_50 - poly_188 
    poly_190 = poly_2 * poly_97 - poly_170 - poly_187 - poly_187 
    poly_191 = poly_2 * poly_98 - poly_174 
    poly_192 = poly_1 * poly_102 - poly_191 
    poly_193 = poly_6 * poly_50 - poly_176 
    poly_194 = poly_1 * poly_101 - poly_180 
    poly_195 = poly_2 * poly_102 - poly_193 
    poly_196 = poly_9 * poly_26 
    poly_197 = poly_9 * poly_37 
    poly_198 = poly_9 * poly_24 
    poly_199 = poly_9 * poly_31 
    poly_200 = poly_9 * poly_25 
    poly_201 = poly_13 * poly_39 
    poly_202 = poly_9 * poly_32 
    poly_203 = poly_13 * poly_40 
    poly_204 = poly_9 * poly_35 
    poly_205 = poly_9 * poly_36 
    poly_206 = poly_13 * poly_28 
    poly_207 = poly_9 * poly_38 
    poly_208 = poly_13 * poly_29 
    poly_209 = poly_13 * poly_30 
    poly_210 = poly_13 * poly_24 
    poly_211 = poly_9 * poly_45 
    poly_212 = poly_13 * poly_25 
    poly_213 = poly_9 * poly_48 
    poly_214 = poly_13 * poly_33 
    poly_215 = poly_13 * poly_34 
    poly_216 = poly_9 * poly_41 
    poly_217 = poly_9 * poly_28 
    poly_218 = poly_9 * poly_23 
    poly_219 = poly_9 * poly_29 
    poly_220 = poly_9 * poly_30 
    poly_221 = poly_9 * poly_42 
    poly_222 = poly_1 * poly_114 - poly_198 
    poly_223 = poly_21 * poly_24 - poly_202 
    poly_224 = poly_3 * poly_72 - poly_199 - poly_223 
    poly_225 = poly_1 * poly_116 - poly_199 - poly_224 
    poly_226 = poly_14 * poly_39 - poly_198 
    poly_227 = poly_3 * poly_73 - poly_202 
    poly_228 = poly_13 * poly_49 
    poly_229 = poly_9 * poly_43 
    poly_230 = poly_9 * poly_33 
    poly_231 = poly_9 * poly_34 
    poly_232 = poly_9 * poly_44 
    poly_233 = poly_16 * poly_24 - poly_196 
    poly_234 = poly_2 * poly_114 - poly_201 
    poly_235 = poly_1 * poly_123 - poly_205 - poly_234 
    poly_236 = poly_13 * poly_41 
    poly_237 = poly_14 * poly_41 - poly_228 
    poly_238 = poly_2 * poly_116 - poly_203 - poly_237 
    poly_239 = poly_16 * poly_32 - poly_228 
    poly_240 = poly_6 * poly_69 - poly_201 
    poly_241 = poly_4 * poly_73 - poly_228 
    poly_242 = poly_11 * poly_32 - poly_196 
    poly_243 = poly_13 * poly_42 
    poly_244 = poly_1 * poly_130 - poly_211 
    poly_245 = poly_13 * poly_31 
    poly_246 = poly_4 * poly_63 - poly_213 - poly_209 
    poly_247 = poly_13 * poly_26 
    poly_248 = poly_3 * poly_82 - poly_211 
    poly_249 = poly_13 * poly_32 
    poly_250 = poly_9 * poly_46 
    poly_251 = poly_9 * poly_47 
    poly_252 = poly_18 * poly_24 - poly_197 
    poly_253 = poly_1 * poly_135 - poly_213 - poly_252 
    poly_254 = poly_13 * poly_43 
    poly_255 = poly_14 * poly_43 - poly_236 
    poly_256 = poly_2 * poly_126 - poly_209 - poly_246 
    poly_257 = poly_18 * poly_32 - poly_236 
    poly_258 = poly_14 * poly_34 - poly_197 
    poly_259 = poly_13 * poly_44 
    poly_260 = poly_2 * poly_130 - poly_210 
    poly_261 = poly_13 * poly_35 
    poly_262 = poly_13 * poly_36 
    poly_263 = poly_4 * poly_82 - poly_249 
    poly_264 = poly_11 * poly_45 - poly_210 
    poly_265 = poly_13 * poly_38 
    poly_266 = poly_9 * poly_50 
    poly_267 = poly_22 * poly_24 - poly_215 
    poly_268 = poly_13 * poly_46 
    poly_269 = poly_14 * poly_46 - poly_254 
    poly_270 = poly_6 * poly_84 - poly_215 
    poly_271 = poly_13 * poly_47 
    poly_272 = poly_9 * poly_27 
    poly_273 = poly_9 * poly_39 
    poly_274 = poly_9 * poly_40 
    poly_275 = poly_9 * poly_49 
    poly_276 = poly_3 * poly_91 - poly_216 
    poly_277 = poly_16 * poly_39 - poly_218 
    poly_278 = poly_1 * poly_144 - poly_217 - poly_277 
    poly_279 = poly_2 * poly_178 - poly_278 
    poly_280 = poly_1 * poly_145 - poly_219 - poly_278 
    poly_281 = poly_7 * poly_69 - poly_218 
    poly_282 = poly_1 * poly_147 - poly_221 - poly_280 
    poly_283 = poly_11 * poly_49 - poly_216 
    poly_284 = poly_1 * poly_149 - poly_224 - poly_223 
    poly_285 = poly_1 * poly_150 - poly_227 
    poly_286 = poly_3 * poly_93 - poly_229 
    poly_287 = poly_18 * poly_39 - poly_231 
    poly_288 = poly_1 * poly_152 - poly_230 - poly_287 
    poly_289 = poly_1 * poly_153 - poly_231 
    poly_290 = poly_1 * poly_154 - poly_232 - poly_288 
    poly_291 = poly_21 * poly_34 - poly_229 
    poly_292 = poly_6 * poly_91 - poly_228 
    poly_293 = poly_1 * poly_157 - poly_238 - poly_235 
    poly_294 = poly_2 * poly_150 - poly_228 
    poly_295 = poly_1 * poly_159 - poly_246 - poly_244 
    poly_296 = poly_1 * poly_161 - poly_248 
    poly_297 = poly_3 * poly_95 - poly_250 
    poly_298 = poly_2 * poly_152 - poly_239 - poly_234 
    poly_299 = poly_2 * poly_153 - poly_240 
    poly_300 = poly_1 * poly_163 - poly_251 - poly_298 
    poly_301 = poly_7 * poly_84 - poly_250 
    poly_302 = poly_6 * poly_93 - poly_236 
    poly_303 = poly_1 * poly_166 - poly_256 - poly_253 
    poly_304 = poly_8 * poly_73 - poly_236 
    poly_305 = poly_16 * poly_45 - poly_249 
    poly_306 = poly_1 * poly_169 - poly_264 - poly_260 
    poly_307 = poly_13 * poly_37 
    poly_308 = poly_2 * poly_161 - poly_249 
    poly_309 = poly_6 * poly_80 - poly_262 - poly_261 
    poly_310 = poly_13 * poly_45 
    poly_311 = poly_1 * poly_187 - poly_309 
    poly_312 = poly_3 * poly_98 - poly_266 
    poly_313 = poly_2 * poly_163 - poly_257 - poly_253 
    poly_314 = poly_1 * poly_173 - poly_266 
    poly_315 = poly_6 * poly_95 - poly_254 
    poly_316 = poly_1 * poly_175 - poly_270 - poly_267 
    poly_317 = poly_22 * poly_32 - poly_254 
    poly_318 = poly_18 * poly_45 - poly_247 
    poly_319 = poly_2 * poly_169 - poly_262 - poly_309 
    poly_320 = poly_13 * poly_48 
    poly_321 = poly_8 * poly_82 - poly_247 
    poly_322 = poly_2 * poly_172 - poly_269 - poly_267 
    poly_323 = poly_2 * poly_173 - poly_270 
    poly_324 = poly_6 * poly_98 - poly_268 
    poly_325 = poly_2 * poly_175 - poly_271 - poly_319 
    poly_326 = poly_13 * poly_50 
    poly_327 = poly_14 * poly_50 - poly_268 
    poly_328 = poly_1 * poly_178 - poly_273 
    poly_329 = poly_21 * poly_39 - poly_272 
    poly_330 = poly_3 * poly_101 - poly_275 
    poly_331 = poly_1 * poly_181 - poly_276 
    poly_332 = poly_2 * poly_194 - poly_331 
    poly_333 = poly_1 * poly_183 - poly_286 
    poly_334 = poly_8 * poly_101 - poly_333 
    poly_335 = poly_1 * poly_185 - poly_297 
    poly_336 = poly_22 * poly_49 - poly_335 
    poly_337 = poly_1 * poly_188 - poly_312 
    poly_338 = poly_21 * poly_50 - poly_337 
    poly_339 = poly_2 * poly_187 - poly_310 
    poly_340 = poly_1 * poly_191 - poly_322 
    poly_341 = poly_7 * poly_102 - poly_340 
    poly_342 = poly_22 * poly_45 - poly_307 
    poly_343 = poly_2 * poly_191 - poly_324 
    poly_344 = poly_1 * poly_195 - poly_343 
    poly_345 = poly_6 * poly_102 - poly_326 
    poly_346 = poly_1 * poly_194 - poly_330 
    poly_347 = poly_2 * poly_195 - poly_345 
    poly_348 = poly_9 * poly_52 
    poly_349 = poly_9 * poly_58 
    poly_350 = poly_9 * poly_61 
    poly_351 = poly_9 * poly_64 
    poly_352 = poly_9 * poly_81 
    poly_353 = poly_9 * poly_87 
    poly_354 = poly_9 * poly_55 
    poly_355 = poly_9 * poly_72 
    poly_356 = poly_9 * poly_51 
    poly_357 = poly_9 * poly_56 
    poly_358 = poly_9 * poly_57 
    poly_359 = poly_13 * poly_89 
    poly_360 = poly_9 * poly_73 
    poly_361 = poly_13 * poly_90 
    poly_362 = poly_9 * poly_77 
    poly_363 = poly_9 * poly_60 
    poly_364 = poly_9 * poly_78 
    poly_365 = poly_13 * poly_67 
    poly_366 = poly_9 * poly_62 
    poly_367 = poly_9 * poly_63 
    poly_368 = poly_13 * poly_68 
    poly_369 = poly_13 * poly_69 
    poly_370 = poly_9 * poly_79 
    poly_371 = poly_13 * poly_70 
    poly_372 = poly_13 * poly_71 
    poly_373 = poly_13 * poly_55 
    poly_374 = poly_9 * poly_80 
    poly_375 = poly_13 * poly_56 
    poly_376 = poly_13 * poly_52 
    poly_377 = poly_9 * poly_82 
    poly_378 = poly_13 * poly_57 
    poly_379 = poly_9 * poly_85 
    poly_380 = poly_9 * poly_86 
    poly_381 = poly_13 * poly_74 
    poly_382 = poly_9 * poly_88 
    poly_383 = poly_13 * poly_75 
    poly_384 = poly_13 * poly_76 
    poly_385 = poly_13 * poly_60 
    poly_386 = poly_9 * poly_97 
    poly_387 = poly_13 * poly_62 
    poly_388 = poly_13 * poly_63 
    poly_389 = poly_9 * poly_100 
    poly_390 = poly_13 * poly_83 
    poly_391 = poly_13 * poly_84 
    poly_392 = poly_9 * poly_91 
    poly_393 = poly_9 * poly_67 
    poly_394 = poly_9 * poly_53 
    poly_395 = poly_9 * poly_68 
    poly_396 = poly_9 * poly_54 
    poly_397 = poly_9 * poly_69 
    poly_398 = poly_9 * poly_70 
    poly_399 = poly_9 * poly_71 
    poly_400 = poly_9 * poly_92 
    poly_401 = poly_3 * poly_114 - poly_354 
    poly_402 = poly_7 * poly_114 - poly_356 
    poly_403 = poly_24 * poly_49 - poly_360 
    poly_404 = poly_3 * poly_149 - poly_355 - poly_403 
    poly_405 = poly_1 * poly_224 - poly_355 - poly_404 
    poly_406 = poly_6 * poly_178 - poly_401 
    poly_407 = poly_32 * poly_39 - poly_356 
    poly_408 = poly_3 * poly_150 - poly_360 
    poly_409 = poly_13 * poly_101 
    poly_410 = poly_9 * poly_93 
    poly_411 = poly_9 * poly_74 
    poly_412 = poly_9 * poly_59 
    poly_413 = poly_9 * poly_75 
    poly_414 = poly_9 * poly_76 
    poly_415 = poly_9 * poly_94 
    poly_416 = poly_24 * poly_41 - poly_349 
    poly_417 = poly_4 * poly_114 - poly_348 
    poly_418 = poly_1 * poly_234 - poly_363 - poly_417 
    poly_419 = poly_1 * poly_235 - poly_364 - poly_418 
    poly_420 = poly_13 * poly_91 
    poly_421 = poly_14 * poly_91 - poly_409 
    poly_422 = poly_2 * poly_224 - poly_361 - poly_421 
    poly_423 = poly_32 * poly_41 - poly_409 
    poly_424 = poly_1 * poly_238 - poly_364 - poly_422 
    poly_425 = poly_16 * poly_73 - poly_409 
    poly_426 = poly_14 * poly_69 - poly_348 
    poly_427 = poly_4 * poly_150 - poly_409 
    poly_428 = poly_11 * poly_73 - poly_349 
    poly_429 = poly_13 * poly_92 
    poly_430 = poly_3 * poly_130 - poly_374 
    poly_431 = poly_7 * poly_130 - poly_377 
    poly_432 = poly_13 * poly_72 
    poly_433 = poly_3 * poly_159 - poly_374 - poly_431 
    poly_434 = poly_39 * poly_45 - poly_430 
    poly_435 = poly_13 * poly_58 
    poly_436 = poly_3 * poly_161 - poly_377 
    poly_437 = poly_13 * poly_73 
    poly_438 = poly_9 * poly_95 
    poly_439 = poly_9 * poly_83 
    poly_440 = poly_9 * poly_84 
    poly_441 = poly_9 * poly_96 
    poly_442 = poly_24 * poly_43 - poly_350 
    poly_443 = poly_8 * poly_114 - poly_369 
    poly_444 = poly_1 * poly_253 - poly_380 - poly_443 
    poly_445 = poly_13 * poly_93 
    poly_446 = poly_14 * poly_93 - poly_420 
    poly_447 = poly_2 * poly_238 - poly_372 - poly_433 
    poly_448 = poly_32 * poly_43 - poly_420 
    poly_449 = poly_6 * poly_153 - poly_369 
    poly_450 = poly_18 * poly_73 - poly_420 
    poly_451 = poly_32 * poly_34 - poly_350 
    poly_452 = poly_13 * poly_94 
    poly_453 = poly_4 * poly_130 - poly_352 
    poly_454 = poly_1 * poly_260 - poly_386 - poly_453 
    poly_455 = poly_13 * poly_77 
    poly_456 = poly_13 * poly_78 
    poly_457 = poly_13 * poly_61 
    poly_458 = poly_16 * poly_82 - poly_437 
    poly_459 = poly_2 * poly_246 - poly_375 - poly_458 
    poly_460 = poly_13 * poly_64 
    poly_461 = poly_4 * poly_161 - poly_437 
    poly_462 = poly_11 * poly_82 - poly_352 
    poly_463 = poly_13 * poly_79 
    poly_464 = poly_6 * poly_130 - poly_385 
    poly_465 = poly_13 * poly_80 
    poly_466 = poly_3 * poly_187 - poly_464 
    poly_467 = poly_13 * poly_82 
    poly_468 = poly_9 * poly_98 
    poly_469 = poly_9 * poly_99 
    poly_470 = poly_24 * poly_46 - poly_353 
    poly_471 = poly_1 * poly_267 - poly_389 - poly_470 
    poly_472 = poly_13 * poly_95 
    poly_473 = poly_14 * poly_95 - poly_445 
    poly_474 = poly_2 * poly_256 - poly_384 - poly_459 
    poly_475 = poly_32 * poly_46 - poly_445 
    poly_476 = poly_14 * poly_84 - poly_353 
    poly_477 = poly_13 * poly_96 
    poly_478 = poly_8 * poly_130 - poly_376 
    poly_479 = poly_13 * poly_85 
    poly_480 = poly_13 * poly_86 
    poly_481 = poly_18 * poly_82 - poly_435 
    poly_482 = poly_34 * poly_45 - poly_376 
    poly_483 = poly_13 * poly_88 
    poly_484 = poly_9 * poly_102 
    poly_485 = poly_24 * poly_50 - poly_391 
    poly_486 = poly_13 * poly_98 
    poly_487 = poly_14 * poly_98 - poly_472 
    poly_488 = poly_6 * poly_173 - poly_391 
    poly_489 = poly_13 * poly_99 
    poly_490 = poly_9 * poly_65 
    poly_491 = poly_9 * poly_66 
    poly_492 = poly_9 * poly_89 
    poly_493 = poly_9 * poly_90 
    poly_494 = poly_9 * poly_101 
    poly_495 = poly_3 * poly_181 - poly_392 
    poly_496 = poly_39 * poly_41 - poly_394 
    poly_497 = poly_4 * poly_178 - poly_397 
    poly_498 = poly_1 * poly_278 - poly_395 - poly_497 
    poly_499 = poly_1 * poly_279 - poly_397 
    poly_500 = poly_1 * poly_280 - poly_398 - poly_498 
    poly_501 = poly_21 * poly_69 - poly_394 
    poly_502 = poly_1 * poly_282 - poly_400 - poly_500 
    poly_503 = poly_11 * poly_101 - poly_392 
    poly_504 = poly_1 * poly_284 - poly_404 - poly_403 
    poly_505 = poly_1 * poly_285 - poly_408 
    poly_506 = poly_3 * poly_183 - poly_410 
    poly_507 = poly_39 * poly_43 - poly_412 
    poly_508 = poly_1 * poly_287 - poly_411 - poly_507 
    poly_509 = poly_2 * poly_279 - poly_406 
    poly_510 = poly_1 * poly_288 - poly_413 - poly_508 
    poly_511 = poly_7 * poly_153 - poly_412 
    poly_512 = poly_1 * poly_290 - poly_415 - poly_510 
    poly_513 = poly_34 * poly_49 - poly_410 
    poly_514 = poly_6 * poly_181 - poly_409 
    poly_515 = poly_1 * poly_293 - poly_422 - poly_419 
    poly_516 = poly_2 * poly_285 - poly_409 
    poly_517 = poly_1 * poly_295 - poly_433 - poly_431 
    poly_518 = poly_1 * poly_296 - poly_436 
    poly_519 = poly_3 * poly_185 - poly_438 
    poly_520 = poly_39 * poly_46 - poly_440 
    poly_521 = poly_1 * poly_298 - poly_439 - poly_520 
    poly_522 = poly_1 * poly_299 - poly_440 
    poly_523 = poly_1 * poly_300 - poly_441 - poly_521 
    poly_524 = poly_21 * poly_84 - poly_438 
    poly_525 = poly_6 * poly_183 - poly_420 
    poly_526 = poly_1 * poly_303 - poly_447 - poly_444 
    poly_527 = poly_8 * poly_150 - poly_420 
    poly_528 = poly_41 * poly_45 - poly_437 
    poly_529 = poly_1 * poly_306 - poly_459 - poly_454 
    poly_530 = poly_2 * poly_296 - poly_437 
    poly_531 = poly_4 * poly_169 - poly_388 - poly_478 
    poly_532 = poly_13 * poly_81 
    poly_533 = poly_1 * poly_311 - poly_466 
    poly_534 = poly_3 * poly_188 - poly_468 
    poly_535 = poly_2 * poly_298 - poly_448 - poly_443 
    poly_536 = poly_2 * poly_299 - poly_449 
    poly_537 = poly_1 * poly_313 - poly_469 - poly_535 
    poly_538 = poly_7 * poly_173 - poly_468 
    poly_539 = poly_6 * poly_185 - poly_445 
    poly_540 = poly_1 * poly_316 - poly_474 - poly_471 
    poly_541 = poly_22 * poly_73 - poly_445 
    poly_542 = poly_43 * poly_45 - poly_435 
    poly_543 = poly_1 * poly_319 - poly_482 - poly_478 
    poly_544 = poly_13 * poly_87 
    poly_545 = poly_8 * poly_161 - poly_435 
    poly_546 = poly_4 * poly_187 - poly_467 
    poly_547 = poly_2 * poly_309 - poly_465 - poly_546 
    poly_548 = poly_13 * poly_97 
    poly_549 = poly_2 * poly_311 - poly_467 
    poly_550 = poly_3 * poly_191 - poly_484 
    poly_551 = poly_2 * poly_313 - poly_475 - poly_471 
    poly_552 = poly_1 * poly_323 - poly_484 
    poly_553 = poly_6 * poly_188 - poly_472 
    poly_554 = poly_1 * poly_325 - poly_488 - poly_485 
    poly_555 = poly_32 * poly_50 - poly_472 
    poly_556 = poly_45 * poly_46 - poly_457 
    poly_557 = poly_2 * poly_319 - poly_480 - poly_547 
    poly_558 = poly_13 * poly_100 
    poly_559 = poly_22 * poly_82 - poly_457 
    poly_560 = poly_2 * poly_322 - poly_487 - poly_485 
    poly_561 = poly_2 * poly_323 - poly_488 
    poly_562 = poly_6 * poly_191 - poly_486 
    poly_563 = poly_2 * poly_325 - poly_489 - poly_557 
    poly_564 = poly_13 * poly_102 
    poly_565 = poly_14 * poly_102 - poly_486 
    poly_566 = poly_3 * poly_178 - poly_492 
    poly_567 = poly_7 * poly_178 - poly_490 
    poly_568 = poly_39 * poly_49 - poly_491 
    poly_569 = poly_3 * poly_194 - poly_494 
    poly_570 = poly_1 * poly_331 - poly_495 
    poly_571 = poly_2 * poly_346 - poly_570 
    poly_572 = poly_1 * poly_333 - poly_506 
    poly_573 = poly_8 * poly_194 - poly_572 
    poly_574 = poly_1 * poly_335 - poly_519 
    poly_575 = poly_22 * poly_101 - poly_574 
    poly_576 = poly_1 * poly_337 - poly_534 
    poly_577 = poly_49 * poly_50 - poly_576 
    poly_578 = poly_6 * poly_187 - poly_548 
    poly_579 = poly_1 * poly_340 - poly_550 
    poly_580 = poly_21 * poly_102 - poly_579 
    poly_581 = poly_8 * poly_187 - poly_532 
    poly_582 = poly_1 * poly_343 - poly_560 
    poly_583 = poly_7 * poly_195 - poly_582 
    poly_584 = poly_45 * poly_50 - poly_544 
    poly_585 = poly_2 * poly_343 - poly_562 
    poly_586 = poly_1 * poly_347 - poly_585 
    poly_587 = poly_6 * poly_195 - poly_564 
    poly_588 = poly_1 * poly_346 - poly_569 
    poly_589 = poly_2 * poly_347 - poly_587 

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
    poly_561,    poly_562,    poly_563,    poly_564,    poly_565, 
    poly_566,    poly_567,    poly_568,    poly_569,    poly_570, 
    poly_571,    poly_572,    poly_573,    poly_574,    poly_575, 
    poly_576,    poly_577,    poly_578,    poly_579,    poly_580, 
    poly_581,    poly_582,    poly_583,    poly_584,    poly_585, 
    poly_586,    poly_587,    poly_588,    poly_589,    ]) 

    return poly 



