import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3BC.monomials_MOL_3_1_1_5 import f_monomials as f_monos 

# File created from ./MOL_3_1_1_5.POLY 

N_POLYS = 636

# Total number of monomials = 636 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(636) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) + jnp.take(mono,3) + jnp.take(mono,4) 
    poly_3 = jnp.take(mono,5) + jnp.take(mono,6) + jnp.take(mono,7) 
    poly_4 = jnp.take(mono,8) + jnp.take(mono,9) + jnp.take(mono,10) 
    poly_5 = poly_1 * poly_2 
    poly_6 = jnp.take(mono,11) + jnp.take(mono,12) + jnp.take(mono,13) 
    poly_7 = poly_1 * poly_3 
    poly_8 = jnp.take(mono,14) + jnp.take(mono,15) + jnp.take(mono,16) + jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) 
    poly_9 = jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) 
    poly_10 = poly_2 * poly_3 - poly_8 
    poly_11 = poly_1 * poly_4 
    poly_12 = jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) 
    poly_13 = jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) 
    poly_14 = poly_2 * poly_4 - poly_12 
    poly_15 = poly_3 * poly_4 - poly_13 
    poly_16 = jnp.take(mono,29) + jnp.take(mono,30) + jnp.take(mono,31) 
    poly_17 = poly_1 * poly_1 
    poly_18 = poly_2 * poly_2 - poly_6 - poly_6 
    poly_19 = poly_3 * poly_3 - poly_9 - poly_9 
    poly_20 = poly_4 * poly_4 - poly_16 - poly_16 
    poly_21 = poly_1 * poly_6 
    poly_22 = jnp.take(mono,32) 
    poly_23 = poly_1 * poly_8 
    poly_24 = jnp.take(mono,33) + jnp.take(mono,34) + jnp.take(mono,35) 
    poly_25 = poly_1 * poly_9 
    poly_26 = jnp.take(mono,36) + jnp.take(mono,37) + jnp.take(mono,38) 
    poly_27 = jnp.take(mono,39) 
    poly_28 = poly_1 * poly_10 
    poly_29 = poly_3 * poly_6 - poly_24 
    poly_30 = poly_2 * poly_9 - poly_26 
    poly_31 = poly_1 * poly_12 
    poly_32 = poly_1 * poly_13 
    poly_33 = jnp.take(mono,40) + jnp.take(mono,41) + jnp.take(mono,42) 
    poly_34 = poly_1 * poly_14 
    poly_35 = jnp.take(mono,43) + jnp.take(mono,44) + jnp.take(mono,45) + jnp.take(mono,46) + jnp.take(mono,47) + jnp.take(mono,48) 
    poly_36 = poly_2 * poly_13 - poly_33 
    poly_37 = poly_4 * poly_6 - poly_35 
    poly_38 = poly_1 * poly_15 
    poly_39 = poly_3 * poly_12 - poly_33 
    poly_40 = jnp.take(mono,49) + jnp.take(mono,50) + jnp.take(mono,51) + jnp.take(mono,52) + jnp.take(mono,53) + jnp.take(mono,54) 
    poly_41 = poly_4 * poly_8 - poly_39 - poly_36 
    poly_42 = poly_4 * poly_9 - poly_40 
    poly_43 = poly_4 * poly_10 - poly_33 
    poly_44 = poly_1 * poly_16 
    poly_45 = jnp.take(mono,55) + jnp.take(mono,56) + jnp.take(mono,57) + jnp.take(mono,58) + jnp.take(mono,59) + jnp.take(mono,60) 
    poly_46 = jnp.take(mono,61) + jnp.take(mono,62) + jnp.take(mono,63) + jnp.take(mono,64) + jnp.take(mono,65) + jnp.take(mono,66) 
    poly_47 = jnp.take(mono,67) 
    poly_48 = poly_2 * poly_16 - poly_45 
    poly_49 = poly_3 * poly_16 - poly_46 
    poly_50 = poly_1 * poly_5 
    poly_51 = poly_1 * poly_18 
    poly_52 = poly_2 * poly_6 - poly_22 - poly_22 - poly_22 
    poly_53 = poly_1 * poly_7 
    poly_54 = poly_2 * poly_8 - poly_29 - poly_24 - poly_24 
    poly_55 = poly_2 * poly_10 - poly_29 
    poly_56 = poly_1 * poly_19 
    poly_57 = poly_3 * poly_8 - poly_30 - poly_26 - poly_26 
    poly_58 = poly_3 * poly_9 - poly_27 - poly_27 - poly_27 
    poly_59 = poly_2 * poly_19 - poly_57 
    poly_60 = poly_1 * poly_11 
    poly_61 = poly_2 * poly_12 - poly_35 
    poly_62 = poly_3 * poly_13 - poly_40 
    poly_63 = poly_4 * poly_18 - poly_61 
    poly_64 = poly_4 * poly_19 - poly_62 
    poly_65 = poly_1 * poly_20 
    poly_66 = poly_4 * poly_12 - poly_45 
    poly_67 = poly_4 * poly_13 - poly_46 
    poly_68 = poly_2 * poly_20 - poly_66 
    poly_69 = poly_3 * poly_20 - poly_67 
    poly_70 = poly_4 * poly_16 - poly_47 - poly_47 - poly_47 
    poly_71 = poly_1 * poly_17 
    poly_72 = poly_2 * poly_18 - poly_52 
    poly_73 = poly_3 * poly_19 - poly_58 
    poly_74 = poly_4 * poly_20 - poly_70 
    poly_75 = poly_1 * poly_22 
    poly_76 = poly_1 * poly_24 
    poly_77 = poly_1 * poly_26 
    poly_78 = poly_1 * poly_27 
    poly_79 = poly_1 * poly_29 
    poly_80 = poly_22 * poly_3 
    poly_81 = poly_1 * poly_30 
    poly_82 = jnp.take(mono,68) + jnp.take(mono,69) + jnp.take(mono,70) + jnp.take(mono,71) + jnp.take(mono,72) + jnp.take(mono,73) 
    poly_83 = poly_27 * poly_2 
    poly_84 = poly_6 * poly_9 - poly_82 
    poly_85 = poly_1 * poly_33 
    poly_86 = poly_1 * poly_35 
    poly_87 = poly_1 * poly_36 
    poly_88 = jnp.take(mono,74) + jnp.take(mono,75) + jnp.take(mono,76) + jnp.take(mono,77) + jnp.take(mono,78) + jnp.take(mono,79) 
    poly_89 = poly_1 * poly_37 
    poly_90 = poly_22 * poly_4 
    poly_91 = poly_6 * poly_13 - poly_88 
    poly_92 = poly_1 * poly_39 
    poly_93 = poly_1 * poly_40 
    poly_94 = jnp.take(mono,80) + jnp.take(mono,81) + jnp.take(mono,82) + jnp.take(mono,83) + jnp.take(mono,84) + jnp.take(mono,85) 
    poly_95 = poly_1 * poly_41 
    poly_96 = poly_4 * poly_24 - poly_91 
    poly_97 = jnp.take(mono,86) + jnp.take(mono,87) + jnp.take(mono,88) + jnp.take(mono,89) + jnp.take(mono,90) + jnp.take(mono,91) 
    poly_98 = poly_1 * poly_42 
    poly_99 = poly_4 * poly_26 - poly_97 
    poly_100 = poly_27 * poly_4 
    poly_101 = poly_1 * poly_43 
    poly_102 = poly_3 * poly_35 - poly_96 - poly_88 
    poly_103 = poly_2 * poly_40 - poly_97 - poly_94 
    poly_104 = poly_3 * poly_37 - poly_91 
    poly_105 = poly_2 * poly_42 - poly_99 
    poly_106 = poly_1 * poly_45 
    poly_107 = jnp.take(mono,92) + jnp.take(mono,93) + jnp.take(mono,94) 
    poly_108 = poly_1 * poly_46 
    poly_109 = jnp.take(mono,95) + jnp.take(mono,96) + jnp.take(mono,97) + jnp.take(mono,98) + jnp.take(mono,99) + jnp.take(mono,100) 
    poly_110 = jnp.take(mono,101) + jnp.take(mono,102) + jnp.take(mono,103) 
    poly_111 = jnp.take(mono,104) + jnp.take(mono,105) + jnp.take(mono,106) + jnp.take(mono,107) + jnp.take(mono,108) + jnp.take(mono,109) 
    poly_112 = poly_1 * poly_47 
    poly_113 = poly_1 * poly_48 
    poly_114 = poly_6 * poly_16 - poly_107 
    poly_115 = poly_2 * poly_46 - poly_111 - poly_109 
    poly_116 = poly_47 * poly_2 
    poly_117 = poly_1 * poly_49 
    poly_118 = poly_3 * poly_45 - poly_111 - poly_109 
    poly_119 = poly_9 * poly_16 - poly_110 
    poly_120 = poly_47 * poly_3 
    poly_121 = poly_2 * poly_49 - poly_118 
    poly_122 = poly_1 * poly_21 
    poly_123 = poly_1 * poly_52 
    poly_124 = poly_22 * poly_2 
    poly_125 = poly_1 * poly_23 
    poly_126 = poly_1 * poly_54 
    poly_127 = poly_2 * poly_24 - poly_80 
    poly_128 = poly_1 * poly_25 
    poly_129 = poly_2 * poly_26 - poly_82 
    poly_130 = poly_1 * poly_28 
    poly_131 = poly_6 * poly_8 - poly_80 - poly_127 - poly_80 
    poly_132 = poly_1 * poly_55 
    poly_133 = poly_6 * poly_10 - poly_80 
    poly_134 = poly_9 * poly_18 - poly_129 
    poly_135 = poly_1 * poly_57 
    poly_136 = poly_3 * poly_24 - poly_82 
    poly_137 = poly_1 * poly_58 
    poly_138 = poly_3 * poly_26 - poly_83 
    poly_139 = poly_27 * poly_3 
    poly_140 = poly_8 * poly_9 - poly_83 - poly_138 - poly_83 
    poly_141 = poly_1 * poly_59 
    poly_142 = poly_6 * poly_19 - poly_136 
    poly_143 = poly_9 * poly_10 - poly_83 
    poly_144 = poly_1 * poly_31 
    poly_145 = poly_1 * poly_61 
    poly_146 = poly_1 * poly_32 
    poly_147 = poly_2 * poly_33 - poly_88 
    poly_148 = poly_1 * poly_62 
    poly_149 = poly_3 * poly_33 - poly_94 
    poly_150 = poly_1 * poly_34 
    poly_151 = poly_6 * poly_12 - poly_90 
    poly_152 = poly_2 * poly_62 - poly_149 
    poly_153 = poly_1 * poly_63 
    poly_154 = poly_2 * poly_35 - poly_90 - poly_151 - poly_90 
    poly_155 = poly_13 * poly_18 - poly_147 
    poly_156 = poly_2 * poly_37 - poly_90 
    poly_157 = poly_1 * poly_38 
    poly_158 = poly_3 * poly_61 - poly_147 
    poly_159 = poly_9 * poly_13 - poly_100 
    poly_160 = poly_2 * poly_41 - poly_104 - poly_96 
    poly_161 = poly_4 * poly_55 - poly_147 
    poly_162 = poly_1 * poly_64 
    poly_163 = poly_12 * poly_19 - poly_149 
    poly_164 = poly_3 * poly_40 - poly_100 - poly_159 - poly_100 
    poly_165 = poly_3 * poly_41 - poly_105 - poly_97 
    poly_166 = poly_3 * poly_42 - poly_100 
    poly_167 = poly_4 * poly_59 - poly_149 
    poly_168 = poly_1 * poly_44 
    poly_169 = poly_2 * poly_45 - poly_114 - poly_107 - poly_107 
    poly_170 = poly_3 * poly_46 - poly_119 - poly_110 - poly_110 
    poly_171 = poly_2 * poly_48 - poly_114 
    poly_172 = poly_3 * poly_49 - poly_119 
    poly_173 = poly_1 * poly_66 
    poly_174 = poly_1 * poly_67 
    poly_175 = poly_4 * poly_33 - poly_111 
    poly_176 = poly_1 * poly_68 
    poly_177 = poly_12 * poly_14 - poly_114 - poly_169 
    poly_178 = poly_2 * poly_67 - poly_175 
    poly_179 = poly_4 * poly_37 - poly_114 
    poly_180 = poly_1 * poly_69 
    poly_181 = poly_3 * poly_66 - poly_175 
    poly_182 = poly_13 * poly_15 - poly_119 - poly_170 
    poly_183 = poly_4 * poly_41 - poly_118 - poly_115 
    poly_184 = poly_4 * poly_42 - poly_119 
    poly_185 = poly_10 * poly_20 - poly_175 
    poly_186 = poly_1 * poly_70 
    poly_187 = poly_12 * poly_16 - poly_116 
    poly_188 = poly_13 * poly_16 - poly_120 
    poly_189 = poly_4 * poly_45 - poly_116 - poly_187 - poly_116 
    poly_190 = poly_4 * poly_46 - poly_120 - poly_188 - poly_120 
    poly_191 = poly_47 * poly_4 
    poly_192 = poly_4 * poly_48 - poly_116 
    poly_193 = poly_4 * poly_49 - poly_120 
    poly_194 = poly_1 * poly_50 
    poly_195 = poly_1 * poly_51 
    poly_196 = poly_6 * poly_6 - poly_124 - poly_124 
    poly_197 = poly_1 * poly_72 
    poly_198 = poly_6 * poly_18 - poly_124 
    poly_199 = poly_1 * poly_53 
    poly_200 = poly_2 * poly_54 - poly_131 - poly_127 
    poly_201 = poly_2 * poly_55 - poly_133 
    poly_202 = poly_1 * poly_56 
    poly_203 = poly_2 * poly_57 - poly_142 - poly_136 - poly_136 
    poly_204 = poly_9 * poly_9 - poly_139 - poly_139 
    poly_205 = poly_2 * poly_59 - poly_142 
    poly_206 = poly_1 * poly_73 
    poly_207 = poly_3 * poly_57 - poly_140 - poly_138 
    poly_208 = poly_9 * poly_19 - poly_139 
    poly_209 = poly_2 * poly_73 - poly_207 
    poly_210 = poly_1 * poly_60 
    poly_211 = poly_2 * poly_61 - poly_151 
    poly_212 = poly_3 * poly_62 - poly_159 
    poly_213 = poly_4 * poly_72 - poly_211 
    poly_214 = poly_4 * poly_73 - poly_212 
    poly_215 = poly_1 * poly_65 
    poly_216 = poly_2 * poly_66 - poly_177 
    poly_217 = poly_3 * poly_67 - poly_182 
    poly_218 = poly_18 * poly_20 - poly_216 
    poly_219 = poly_19 * poly_20 - poly_217 
    poly_220 = poly_16 * poly_16 - poly_191 - poly_191 
    poly_221 = poly_1 * poly_74 
    poly_222 = poly_4 * poly_66 - poly_187 
    poly_223 = poly_4 * poly_67 - poly_188 
    poly_224 = poly_2 * poly_74 - poly_222 
    poly_225 = poly_3 * poly_74 - poly_223 
    poly_226 = poly_16 * poly_20 - poly_191 
    poly_227 = poly_1 * poly_71 
    poly_228 = poly_2 * poly_72 - poly_198 
    poly_229 = poly_3 * poly_73 - poly_208 
    poly_230 = poly_4 * poly_74 - poly_226 
    poly_231 = poly_1 * poly_80 
    poly_232 = poly_1 * poly_82 
    poly_233 = poly_1 * poly_83 
    poly_234 = poly_1 * poly_84 
    poly_235 = poly_22 * poly_9 
    poly_236 = poly_27 * poly_6 
    poly_237 = poly_1 * poly_88 
    poly_238 = poly_1 * poly_90 
    poly_239 = poly_1 * poly_91 
    poly_240 = poly_22 * poly_13 
    poly_241 = poly_1 * poly_94 
    poly_242 = poly_1 * poly_96 
    poly_243 = poly_1 * poly_97 
    poly_244 = jnp.take(mono,110) + jnp.take(mono,111) + jnp.take(mono,112) + jnp.take(mono,113) + jnp.take(mono,114) + jnp.take(mono,115) 
    poly_245 = poly_1 * poly_99 
    poly_246 = poly_1 * poly_100 
    poly_247 = poly_27 * poly_12 
    poly_248 = poly_1 * poly_102 
    poly_249 = poly_1 * poly_103 
    poly_250 = jnp.take(mono,116) + jnp.take(mono,117) + jnp.take(mono,118) + jnp.take(mono,119) + jnp.take(mono,120) + jnp.take(mono,121) 
    poly_251 = poly_1 * poly_104 
    poly_252 = poly_22 * poly_15 
    poly_253 = poly_6 * poly_40 - poly_250 - poly_244 
    poly_254 = poly_1 * poly_105 
    poly_255 = poly_4 * poly_82 - poly_253 - poly_244 
    poly_256 = poly_27 * poly_14 
    poly_257 = poly_4 * poly_84 - poly_250 
    poly_258 = poly_1 * poly_107 
    poly_259 = poly_1 * poly_109 
    poly_260 = poly_1 * poly_110 
    poly_261 = poly_1 * poly_111 
    poly_262 = jnp.take(mono,122) + jnp.take(mono,123) + jnp.take(mono,124) + jnp.take(mono,125) + jnp.take(mono,126) + jnp.take(mono,127) 
    poly_263 = jnp.take(mono,128) + jnp.take(mono,129) + jnp.take(mono,130) + jnp.take(mono,131) + jnp.take(mono,132) + jnp.take(mono,133) 
    poly_264 = poly_1 * poly_114 
    poly_265 = poly_22 * poly_16 
    poly_266 = poly_1 * poly_115 
    poly_267 = jnp.take(mono,134) + jnp.take(mono,135) + jnp.take(mono,136) + jnp.take(mono,137) + jnp.take(mono,138) + jnp.take(mono,139) 
    poly_268 = poly_2 * poly_110 - poly_263 
    poly_269 = poly_6 * poly_46 - poly_267 - poly_262 
    poly_270 = poly_1 * poly_116 
    poly_271 = poly_47 * poly_6 
    poly_272 = poly_1 * poly_118 
    poly_273 = poly_3 * poly_107 - poly_262 
    poly_274 = poly_1 * poly_119 
    poly_275 = poly_16 * poly_26 - poly_268 
    poly_276 = poly_27 * poly_16 
    poly_277 = poly_9 * poly_45 - poly_275 - poly_263 
    poly_278 = poly_1 * poly_120 
    poly_279 = poly_47 * poly_8 
    poly_280 = poly_47 * poly_9 
    poly_281 = poly_1 * poly_121 
    poly_282 = poly_6 * poly_49 - poly_273 
    poly_283 = poly_9 * poly_48 - poly_268 
    poly_284 = poly_47 * poly_10 
    poly_285 = poly_1 * poly_75 
    poly_286 = poly_1 * poly_124 
    poly_287 = poly_1 * poly_76 
    poly_288 = poly_1 * poly_127 
    poly_289 = poly_1 * poly_77 
    poly_290 = poly_1 * poly_129 
    poly_291 = poly_1 * poly_78 
    poly_292 = poly_1 * poly_79 
    poly_293 = poly_1 * poly_131 
    poly_294 = poly_22 * poly_8 
    poly_295 = poly_1 * poly_81 
    poly_296 = poly_6 * poly_26 - poly_235 
    poly_297 = poly_1 * poly_133 
    poly_298 = poly_22 * poly_10 
    poly_299 = poly_1 * poly_134 
    poly_300 = poly_2 * poly_82 - poly_235 - poly_296 - poly_235 
    poly_301 = poly_27 * poly_18 
    poly_302 = poly_2 * poly_84 - poly_235 
    poly_303 = poly_1 * poly_136 
    poly_304 = poly_1 * poly_138 
    poly_305 = poly_1 * poly_139 
    poly_306 = poly_1 * poly_140 
    poly_307 = poly_9 * poly_24 - poly_236 
    poly_308 = poly_27 * poly_8 
    poly_309 = poly_1 * poly_142 
    poly_310 = poly_22 * poly_19 
    poly_311 = poly_1 * poly_143 
    poly_312 = poly_10 * poly_26 - poly_301 
    poly_313 = poly_27 * poly_10 
    poly_314 = poly_3 * poly_84 - poly_236 
    poly_315 = poly_1 * poly_85 
    poly_316 = poly_1 * poly_147 
    poly_317 = poly_1 * poly_149 
    poly_318 = poly_1 * poly_86 
    poly_319 = poly_1 * poly_151 
    poly_320 = poly_1 * poly_87 
    poly_321 = poly_6 * poly_33 - poly_240 
    poly_322 = poly_1 * poly_152 
    poly_323 = poly_3 * poly_88 - poly_250 - poly_244 
    poly_324 = poly_1 * poly_89 
    poly_325 = poly_22 * poly_12 
    poly_326 = poly_3 * poly_91 - poly_253 
    poly_327 = poly_1 * poly_154 
    poly_328 = poly_1 * poly_155 
    poly_329 = poly_2 * poly_88 - poly_240 - poly_321 - poly_240 
    poly_330 = poly_1 * poly_156 
    poly_331 = poly_22 * poly_14 
    poly_332 = poly_2 * poly_91 - poly_240 
    poly_333 = poly_1 * poly_92 
    poly_334 = poly_1 * poly_158 
    poly_335 = poly_1 * poly_93 
    poly_336 = poly_2 * poly_94 - poly_250 - poly_244 
    poly_337 = poly_1 * poly_159 
    poly_338 = poly_9 * poly_33 - poly_247 
    poly_339 = poly_1 * poly_95 
    poly_340 = poly_12 * poly_24 - poly_240 
    poly_341 = poly_13 * poly_26 - poly_247 
    poly_342 = poly_1 * poly_160 
    poly_343 = poly_2 * poly_96 - poly_252 - poly_340 
    poly_344 = poly_2 * poly_97 - poly_253 - poly_244 
    poly_345 = poly_1 * poly_98 
    poly_346 = poly_2 * poly_99 - poly_255 
    poly_347 = poly_27 * poly_13 
    poly_348 = poly_1 * poly_101 
    poly_349 = poly_3 * poly_151 - poly_340 - poly_321 
    poly_350 = poly_2 * poly_159 - poly_341 - poly_338 
    poly_351 = poly_4 * poly_131 - poly_349 - poly_329 
    poly_352 = poly_1 * poly_161 
    poly_353 = poly_2 * poly_102 - poly_252 - poly_349 
    poly_354 = poly_2 * poly_103 - poly_253 - poly_250 
    poly_355 = poly_10 * poly_37 - poly_240 
    poly_356 = poly_18 * poly_42 - poly_346 
    poly_357 = poly_1 * poly_163 
    poly_358 = poly_1 * poly_164 
    poly_359 = poly_3 * poly_94 - poly_247 - poly_338 - poly_247 
    poly_360 = poly_1 * poly_165 
    poly_361 = poly_4 * poly_136 - poly_326 
    poly_362 = poly_3 * poly_97 - poly_256 - poly_341 
    poly_363 = poly_1 * poly_166 
    poly_364 = poly_3 * poly_99 - poly_247 
    poly_365 = poly_27 * poly_15 
    poly_366 = poly_4 * poly_140 - poly_359 - poly_350 
    poly_367 = poly_1 * poly_167 
    poly_368 = poly_3 * poly_102 - poly_255 - poly_250 
    poly_369 = poly_2 * poly_164 - poly_362 - poly_359 
    poly_370 = poly_19 * poly_37 - poly_326 
    poly_371 = poly_10 * poly_42 - poly_247 
    poly_372 = poly_1 * poly_106 
    poly_373 = poly_1 * poly_169 
    poly_374 = poly_2 * poly_107 - poly_265 
    poly_375 = poly_1 * poly_108 
    poly_376 = poly_2 * poly_109 - poly_267 - poly_262 
    poly_377 = poly_2 * poly_111 - poly_269 - poly_262 
    poly_378 = poly_1 * poly_170 
    poly_379 = poly_3 * poly_109 - poly_275 - poly_263 
    poly_380 = poly_3 * poly_110 - poly_276 
    poly_381 = poly_3 * poly_111 - poly_277 - poly_263 
    poly_382 = poly_1 * poly_112 
    poly_383 = poly_1 * poly_113 
    poly_384 = poly_6 * poly_45 - poly_265 - poly_374 - poly_265 
    poly_385 = poly_2 * poly_170 - poly_381 - poly_379 
    poly_386 = poly_1 * poly_171 
    poly_387 = poly_6 * poly_48 - poly_265 
    poly_388 = poly_2 * poly_115 - poly_269 - poly_267 
    poly_389 = poly_47 * poly_18 
    poly_390 = poly_1 * poly_117 
    poly_391 = poly_3 * poly_169 - poly_377 - poly_376 
    poly_392 = poly_9 * poly_46 - poly_276 - poly_380 - poly_276 
    poly_393 = poly_2 * poly_121 - poly_282 
    poly_394 = poly_1 * poly_172 
    poly_395 = poly_3 * poly_118 - poly_277 - poly_275 
    poly_396 = poly_9 * poly_49 - poly_276 
    poly_397 = poly_47 * poly_19 
    poly_398 = poly_2 * poly_172 - poly_395 
    poly_399 = poly_1 * poly_175 
    poly_400 = poly_1 * poly_177 
    poly_401 = poly_1 * poly_178 
    poly_402 = poly_4 * poly_88 - poly_269 - poly_262 
    poly_403 = poly_1 * poly_179 
    poly_404 = poly_22 * poly_20 
    poly_405 = poly_4 * poly_91 - poly_267 
    poly_406 = poly_1 * poly_181 
    poly_407 = poly_1 * poly_182 
    poly_408 = poly_4 * poly_94 - poly_277 - poly_263 
    poly_409 = poly_1 * poly_183 
    poly_410 = poly_20 * poly_24 - poly_405 
    poly_411 = poly_13 * poly_41 - poly_277 - poly_385 
    poly_412 = poly_1 * poly_184 
    poly_413 = poly_4 * poly_99 - poly_275 
    poly_414 = poly_27 * poly_20 
    poly_415 = poly_1 * poly_185 
    poly_416 = poly_3 * poly_177 - poly_410 - poly_402 
    poly_417 = poly_2 * poly_182 - poly_411 - poly_408 
    poly_418 = poly_3 * poly_179 - poly_405 
    poly_419 = poly_2 * poly_184 - poly_413 
    poly_420 = poly_1 * poly_187 
    poly_421 = poly_1 * poly_188 
    poly_422 = poly_16 * poly_33 - poly_284 
    poly_423 = poly_1 * poly_189 
    poly_424 = poly_4 * poly_107 - poly_271 
    poly_425 = poly_13 * poly_45 - poly_279 - poly_422 
    poly_426 = poly_1 * poly_190 
    poly_427 = poly_4 * poly_109 - poly_279 - poly_425 
    poly_428 = poly_4 * poly_110 - poly_280 
    poly_429 = poly_4 * poly_111 - poly_284 - poly_422 - poly_284 
    poly_430 = poly_1 * poly_191 
    poly_431 = poly_47 * poly_12 
    poly_432 = poly_47 * poly_13 
    poly_433 = poly_1 * poly_192 
    poly_434 = poly_12 * poly_48 - poly_389 
    poly_435 = poly_13 * poly_48 - poly_284 
    poly_436 = poly_16 * poly_37 - poly_271 
    poly_437 = poly_2 * poly_190 - poly_429 - poly_427 
    poly_438 = poly_47 * poly_14 
    poly_439 = poly_1 * poly_193 
    poly_440 = poly_12 * poly_49 - poly_284 
    poly_441 = poly_13 * poly_49 - poly_397 
    poly_442 = poly_3 * poly_189 - poly_429 - poly_425 
    poly_443 = poly_16 * poly_42 - poly_280 
    poly_444 = poly_47 * poly_15 
    poly_445 = poly_4 * poly_121 - poly_284 
    poly_446 = poly_1 * poly_122 
    poly_447 = poly_1 * poly_123 
    poly_448 = poly_1 * poly_196 
    poly_449 = poly_22 * poly_6 
    poly_450 = poly_1 * poly_198 
    poly_451 = poly_22 * poly_18 
    poly_452 = poly_1 * poly_125 
    poly_453 = poly_1 * poly_126 
    poly_454 = poly_6 * poly_24 - poly_294 
    poly_455 = poly_1 * poly_200 
    poly_456 = poly_18 * poly_24 - poly_298 
    poly_457 = poly_1 * poly_128 
    poly_458 = poly_2 * poly_129 - poly_296 
    poly_459 = poly_1 * poly_130 
    poly_460 = poly_6 * poly_54 - poly_294 - poly_456 
    poly_461 = poly_1 * poly_132 
    poly_462 = poly_3 * poly_196 - poly_454 
    poly_463 = poly_1 * poly_201 
    poly_464 = poly_6 * poly_55 - poly_298 
    poly_465 = poly_9 * poly_72 - poly_458 
    poly_466 = poly_1 * poly_135 
    poly_467 = poly_1 * poly_203 
    poly_468 = poly_2 * poly_136 - poly_310 
    poly_469 = poly_1 * poly_137 
    poly_470 = poly_3 * poly_129 - poly_301 
    poly_471 = poly_2 * poly_140 - poly_314 - poly_307 
    poly_472 = poly_1 * poly_204 
    poly_473 = poly_9 * poly_26 - poly_308 
    poly_474 = poly_27 * poly_9 
    poly_475 = poly_1 * poly_141 
    poly_476 = poly_3 * poly_131 - poly_302 - poly_296 
    poly_477 = poly_2 * poly_204 - poly_473 
    poly_478 = poly_1 * poly_205 
    poly_479 = poly_6 * poly_59 - poly_310 
    poly_480 = poly_9 * poly_55 - poly_301 
    poly_481 = poly_1 * poly_207 
    poly_482 = poly_3 * poly_136 - poly_307 
    poly_483 = poly_1 * poly_208 
    poly_484 = poly_19 * poly_26 - poly_313 
    poly_485 = poly_27 * poly_19 
    poly_486 = poly_3 * poly_140 - poly_308 - poly_477 
    poly_487 = poly_1 * poly_209 
    poly_488 = poly_6 * poly_73 - poly_482 
    poly_489 = poly_9 * poly_59 - poly_313 
    poly_490 = poly_1 * poly_144 
    poly_491 = poly_1 * poly_145 
    poly_492 = poly_1 * poly_211 
    poly_493 = poly_1 * poly_146 
    poly_494 = poly_2 * poly_147 - poly_321 
    poly_495 = poly_1 * poly_148 
    poly_496 = poly_2 * poly_149 - poly_323 
    poly_497 = poly_1 * poly_212 
    poly_498 = poly_3 * poly_149 - poly_338 
    poly_499 = poly_1 * poly_150 
    poly_500 = poly_6 * poly_61 - poly_325 
    poly_501 = poly_2 * poly_212 - poly_498 
    poly_502 = poly_1 * poly_153 
    poly_503 = poly_12 * poly_52 - poly_331 - poly_500 
    poly_504 = poly_18 * poly_62 - poly_496 
    poly_505 = poly_4 * poly_196 - poly_503 
    poly_506 = poly_1 * poly_213 
    poly_507 = poly_2 * poly_154 - poly_331 - poly_503 
    poly_508 = poly_13 * poly_72 - poly_494 
    poly_509 = poly_18 * poly_37 - poly_325 
    poly_510 = poly_1 * poly_157 
    poly_511 = poly_3 * poly_211 - poly_494 
    poly_512 = poly_9 * poly_62 - poly_347 
    poly_513 = poly_2 * poly_160 - poly_351 - poly_343 
    poly_514 = poly_4 * poly_201 - poly_494 
    poly_515 = poly_1 * poly_162 
    poly_516 = poly_19 * poly_61 - poly_496 
    poly_517 = poly_13 * poly_58 - poly_365 - poly_512 
    poly_518 = poly_2 * poly_165 - poly_370 - poly_361 
    poly_519 = poly_4 * poly_204 - poly_517 
    poly_520 = poly_4 * poly_205 - poly_496 
    poly_521 = poly_1 * poly_214 
    poly_522 = poly_12 * poly_73 - poly_498 
    poly_523 = poly_3 * poly_164 - poly_365 - poly_517 
    poly_524 = poly_3 * poly_165 - poly_366 - poly_362 
    poly_525 = poly_19 * poly_42 - poly_347 
    poly_526 = poly_4 * poly_209 - poly_498 
    poly_527 = poly_1 * poly_168 
    poly_528 = poly_2 * poly_169 - poly_384 - poly_374 
    poly_529 = poly_3 * poly_170 - poly_392 - poly_380 
    poly_530 = poly_2 * poly_171 - poly_387 
    poly_531 = poly_3 * poly_172 - poly_396 
    poly_532 = poly_1 * poly_173 
    poly_533 = poly_1 * poly_216 
    poly_534 = poly_1 * poly_174 
    poly_535 = poly_2 * poly_175 - poly_402 
    poly_536 = poly_1 * poly_217 
    poly_537 = poly_3 * poly_175 - poly_408 
    poly_538 = poly_1 * poly_176 
    poly_539 = poly_6 * poly_66 - poly_404 
    poly_540 = poly_2 * poly_217 - poly_537 
    poly_541 = poly_1 * poly_218 
    poly_542 = poly_4 * poly_154 - poly_387 - poly_374 
    poly_543 = poly_18 * poly_67 - poly_535 
    poly_544 = poly_2 * poly_179 - poly_404 
    poly_545 = poly_1 * poly_180 
    poly_546 = poly_3 * poly_216 - poly_535 
    poly_547 = poly_9 * poly_67 - poly_414 
    poly_548 = poly_2 * poly_183 - poly_418 - poly_410 
    poly_549 = poly_20 * poly_55 - poly_535 
    poly_550 = poly_1 * poly_219 
    poly_551 = poly_19 * poly_66 - poly_537 
    poly_552 = poly_4 * poly_164 - poly_396 - poly_380 
    poly_553 = poly_3 * poly_183 - poly_419 - poly_411 
    poly_554 = poly_3 * poly_184 - poly_414 
    poly_555 = poly_20 * poly_59 - poly_537 
    poly_556 = poly_1 * poly_186 
    poly_557 = poly_16 * poly_61 - poly_389 
    poly_558 = poly_16 * poly_62 - poly_397 
    poly_559 = poly_2 * poly_189 - poly_436 - poly_424 
    poly_560 = poly_3 * poly_190 - poly_443 - poly_428 
    poly_561 = poly_4 * poly_171 - poly_389 
    poly_562 = poly_4 * poly_172 - poly_397 
    poly_563 = poly_1 * poly_220 
    poly_564 = poly_16 * poly_45 - poly_438 - poly_431 - poly_431 
    poly_565 = poly_16 * poly_46 - poly_444 - poly_432 - poly_432 
    poly_566 = poly_47 * poly_16 
    poly_567 = poly_2 * poly_220 - poly_564 
    poly_568 = poly_3 * poly_220 - poly_565 
    poly_569 = poly_1 * poly_222 
    poly_570 = poly_1 * poly_223 
    poly_571 = poly_4 * poly_175 - poly_422 
    poly_572 = poly_1 * poly_224 
    poly_573 = poly_4 * poly_177 - poly_434 - poly_424 
    poly_574 = poly_2 * poly_223 - poly_571 
    poly_575 = poly_4 * poly_179 - poly_436 
    poly_576 = poly_1 * poly_225 
    poly_577 = poly_3 * poly_222 - poly_571 
    poly_578 = poly_4 * poly_182 - poly_441 - poly_428 
    poly_579 = poly_4 * poly_183 - poly_442 - poly_437 
    poly_580 = poly_4 * poly_184 - poly_443 
    poly_581 = poly_10 * poly_74 - poly_571 
    poly_582 = poly_1 * poly_226 
    poly_583 = poly_16 * poly_66 - poly_431 
    poly_584 = poly_16 * poly_67 - poly_432 
    poly_585 = poly_4 * poly_189 - poly_438 - poly_564 
    poly_586 = poly_4 * poly_190 - poly_444 - poly_565 
    poly_587 = poly_47 * poly_20 
    poly_588 = poly_20 * poly_48 - poly_431 
    poly_589 = poly_20 * poly_49 - poly_432 
    poly_590 = poly_1 * poly_194 
    poly_591 = poly_1 * poly_195 
    poly_592 = poly_1 * poly_197 
    poly_593 = poly_2 * poly_196 - poly_449 
    poly_594 = poly_1 * poly_228 
    poly_595 = poly_6 * poly_72 - poly_451 
    poly_596 = poly_1 * poly_199 
    poly_597 = poly_2 * poly_200 - poly_460 - poly_456 
    poly_598 = poly_2 * poly_201 - poly_464 
    poly_599 = poly_1 * poly_202 
    poly_600 = poly_2 * poly_203 - poly_476 - poly_468 
    poly_601 = poly_2 * poly_205 - poly_479 
    poly_602 = poly_1 * poly_206 
    poly_603 = poly_3 * poly_203 - poly_471 - poly_470 
    poly_604 = poly_3 * poly_204 - poly_474 
    poly_605 = poly_2 * poly_209 - poly_488 
    poly_606 = poly_1 * poly_229 
    poly_607 = poly_3 * poly_207 - poly_486 - poly_484 
    poly_608 = poly_9 * poly_73 - poly_485 
    poly_609 = poly_2 * poly_229 - poly_607 
    poly_610 = poly_1 * poly_210 
    poly_611 = poly_2 * poly_211 - poly_500 
    poly_612 = poly_3 * poly_212 - poly_512 
    poly_613 = poly_4 * poly_228 - poly_611 
    poly_614 = poly_4 * poly_229 - poly_612 
    poly_615 = poly_1 * poly_215 
    poly_616 = poly_2 * poly_216 - poly_539 
    poly_617 = poly_3 * poly_217 - poly_547 
    poly_618 = poly_20 * poly_72 - poly_616 
    poly_619 = poly_20 * poly_73 - poly_617 
    poly_620 = poly_1 * poly_221 
    poly_621 = poly_2 * poly_222 - poly_573 
    poly_622 = poly_3 * poly_223 - poly_578 
    poly_623 = poly_18 * poly_74 - poly_621 
    poly_624 = poly_19 * poly_74 - poly_622 
    poly_625 = poly_4 * poly_220 - poly_566 
    poly_626 = poly_1 * poly_230 
    poly_627 = poly_4 * poly_222 - poly_583 
    poly_628 = poly_4 * poly_223 - poly_584 
    poly_629 = poly_2 * poly_230 - poly_627 
    poly_630 = poly_3 * poly_230 - poly_628 
    poly_631 = poly_16 * poly_74 - poly_587 
    poly_632 = poly_1 * poly_227 
    poly_633 = poly_2 * poly_228 - poly_595 
    poly_634 = poly_3 * poly_229 - poly_608 
    poly_635 = poly_4 * poly_230 - poly_631 

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
    ]) 

    return poly 



