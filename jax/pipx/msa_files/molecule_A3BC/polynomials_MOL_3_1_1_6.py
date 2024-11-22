import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3BC.monomials_MOL_3_1_1_6 import f_monomials as f_monos 

# File created from ./MOL_3_1_1_6.POLY 

N_POLYS = 1603

# Total number of monomials = 1603 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(1603) 

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
    poly_636 = poly_1 * poly_235 
    poly_637 = poly_1 * poly_236 
    poly_638 = poly_22 * poly_27 
    poly_639 = poly_1 * poly_240 
    poly_640 = poly_1 * poly_244 
    poly_641 = poly_1 * poly_247 
    poly_642 = poly_1 * poly_250 
    poly_643 = poly_1 * poly_252 
    poly_644 = poly_1 * poly_253 
    poly_645 = poly_22 * poly_40 
    poly_646 = poly_1 * poly_255 
    poly_647 = poly_1 * poly_256 
    poly_648 = poly_27 * poly_35 
    poly_649 = poly_1 * poly_257 
    poly_650 = poly_22 * poly_42 
    poly_651 = poly_27 * poly_37 
    poly_652 = poly_1 * poly_262 
    poly_653 = poly_1 * poly_263 
    poly_654 = jnp.take(mono,140) + jnp.take(mono,141) + jnp.take(mono,142) 
    poly_655 = poly_1 * poly_265 
    poly_656 = poly_1 * poly_267 
    poly_657 = poly_1 * poly_268 
    poly_658 = poly_1 * poly_269 
    poly_659 = poly_22 * poly_46 
    poly_660 = poly_6 * poly_110 - poly_654 
    poly_661 = poly_1 * poly_271 
    poly_662 = poly_22 * poly_47 
    poly_663 = poly_1 * poly_273 
    poly_664 = poly_1 * poly_275 
    poly_665 = poly_1 * poly_276 
    poly_666 = poly_1 * poly_277 
    poly_667 = poly_9 * poly_107 - poly_654 
    poly_668 = poly_27 * poly_45 
    poly_669 = poly_1 * poly_279 
    poly_670 = poly_47 * poly_24 
    poly_671 = poly_1 * poly_280 
    poly_672 = poly_47 * poly_26 
    poly_673 = poly_27 * poly_47 
    poly_674 = poly_1 * poly_282 
    poly_675 = poly_22 * poly_49 
    poly_676 = poly_1 * poly_283 
    poly_677 = poly_16 * poly_82 - poly_667 - poly_660 
    poly_678 = poly_27 * poly_48 
    poly_679 = poly_16 * poly_84 - poly_654 
    poly_680 = poly_1 * poly_284 
    poly_681 = poly_47 * poly_29 
    poly_682 = poly_47 * poly_30 
    poly_683 = poly_1 * poly_231 
    poly_684 = poly_1 * poly_294 
    poly_685 = poly_1 * poly_232 
    poly_686 = poly_1 * poly_296 
    poly_687 = poly_1 * poly_233 
    poly_688 = poly_1 * poly_234 
    poly_689 = poly_22 * poly_26 
    poly_690 = poly_1 * poly_298 
    poly_691 = poly_1 * poly_300 
    poly_692 = poly_1 * poly_301 
    poly_693 = poly_1 * poly_302 
    poly_694 = poly_22 * poly_30 
    poly_695 = poly_27 * poly_52 
    poly_696 = poly_1 * poly_307 
    poly_697 = poly_1 * poly_308 
    poly_698 = poly_27 * poly_24 
    poly_699 = poly_1 * poly_310 
    poly_700 = poly_1 * poly_312 
    poly_701 = poly_1 * poly_313 
    poly_702 = poly_1 * poly_314 
    poly_703 = poly_22 * poly_58 
    poly_704 = poly_27 * poly_29 
    poly_705 = poly_1 * poly_237 
    poly_706 = poly_1 * poly_321 
    poly_707 = poly_1 * poly_323 
    poly_708 = poly_1 * poly_238 
    poly_709 = poly_1 * poly_325 
    poly_710 = poly_1 * poly_239 
    poly_711 = poly_22 * poly_33 
    poly_712 = poly_1 * poly_326 
    poly_713 = poly_22 * poly_62 
    poly_714 = poly_1 * poly_329 
    poly_715 = poly_1 * poly_331 
    poly_716 = poly_1 * poly_332 
    poly_717 = poly_22 * poly_36 
    poly_718 = poly_1 * poly_241 
    poly_719 = poly_1 * poly_336 
    poly_720 = poly_1 * poly_338 
    poly_721 = poly_1 * poly_242 
    poly_722 = poly_1 * poly_340 
    poly_723 = poly_1 * poly_243 
    poly_724 = poly_24 * poly_33 - poly_713 
    poly_725 = poly_1 * poly_341 
    poly_726 = jnp.take(mono,143) + jnp.take(mono,144) + jnp.take(mono,145) + jnp.take(mono,146) + jnp.take(mono,147) + jnp.take(mono,148) 
    poly_727 = poly_1 * poly_343 
    poly_728 = poly_1 * poly_344 
    poly_729 = poly_2 * poly_244 - poly_645 - poly_724 
    poly_730 = poly_1 * poly_245 
    poly_731 = poly_1 * poly_346 
    poly_732 = poly_1 * poly_246 
    poly_733 = poly_27 * poly_61 
    poly_734 = poly_1 * poly_347 
    poly_735 = poly_27 * poly_33 
    poly_736 = poly_1 * poly_248 
    poly_737 = poly_1 * poly_349 
    poly_738 = poly_1 * poly_249 
    poly_739 = poly_12 * poly_84 - poly_650 
    poly_740 = poly_1 * poly_350 
    poly_741 = poly_13 * poly_84 - poly_651 
    poly_742 = poly_1 * poly_251 
    poly_743 = poly_22 * poly_39 
    poly_744 = poly_9 * poly_91 - poly_651 
    poly_745 = poly_1 * poly_351 
    poly_746 = poly_22 * poly_41 
    poly_747 = poly_26 * poly_37 - poly_650 
    poly_748 = poly_1 * poly_254 
    poly_749 = poly_6 * poly_99 - poly_650 
    poly_750 = poly_27 * poly_36 
    poly_751 = poly_1 * poly_353 
    poly_752 = poly_1 * poly_354 
    poly_753 = poly_2 * poly_250 - poly_645 - poly_739 
    poly_754 = poly_1 * poly_355 
    poly_755 = poly_22 * poly_43 
    poly_756 = poly_10 * poly_91 - poly_713 
    poly_757 = poly_1 * poly_356 
    poly_758 = poly_4 * poly_300 - poly_756 - poly_724 
    poly_759 = poly_27 * poly_63 
    poly_760 = poly_2 * poly_257 - poly_650 
    poly_761 = poly_1 * poly_359 
    poly_762 = poly_1 * poly_361 
    poly_763 = poly_1 * poly_362 
    poly_764 = poly_3 * poly_244 - poly_648 - poly_726 
    poly_765 = poly_1 * poly_364 
    poly_766 = poly_1 * poly_365 
    poly_767 = poly_27 * poly_39 
    poly_768 = poly_1 * poly_366 
    poly_769 = poly_24 * poly_42 - poly_651 
    poly_770 = poly_27 * poly_41 
    poly_771 = poly_1 * poly_368 
    poly_772 = poly_1 * poly_369 
    poly_773 = poly_3 * poly_250 - poly_648 - poly_741 
    poly_774 = poly_1 * poly_370 
    poly_775 = poly_22 * poly_64 
    poly_776 = poly_6 * poly_164 - poly_773 - poly_764 
    poly_777 = poly_1 * poly_371 
    poly_778 = poly_10 * poly_99 - poly_733 
    poly_779 = poly_27 * poly_43 
    poly_780 = poly_3 * poly_257 - poly_651 
    poly_781 = poly_1 * poly_258 
    poly_782 = poly_1 * poly_374 
    poly_783 = poly_1 * poly_259 
    poly_784 = poly_1 * poly_376 
    poly_785 = poly_1 * poly_260 
    poly_786 = poly_1 * poly_261 
    poly_787 = jnp.take(mono,149) + jnp.take(mono,150) + jnp.take(mono,151) + jnp.take(mono,152) + jnp.take(mono,153) + jnp.take(mono,154) 
    poly_788 = poly_1 * poly_377 
    poly_789 = poly_10 * poly_107 - poly_675 
    poly_790 = poly_2 * poly_263 - poly_660 - poly_654 - poly_654 
    poly_791 = poly_1 * poly_379 
    poly_792 = poly_1 * poly_380 
    poly_793 = jnp.take(mono,155) + jnp.take(mono,156) + jnp.take(mono,157) + jnp.take(mono,158) + jnp.take(mono,159) + jnp.take(mono,160) 
    poly_794 = poly_1 * poly_381 
    poly_795 = poly_10 * poly_109 - poly_677 - poly_790 
    poly_796 = poly_10 * poly_110 - poly_678 
    poly_797 = poly_1 * poly_264 
    poly_798 = poly_1 * poly_384 
    poly_799 = poly_22 * poly_45 
    poly_800 = poly_1 * poly_266 
    poly_801 = poly_6 * poly_109 - poly_659 - poly_787 
    poly_802 = poly_6 * poly_111 - poly_659 - poly_789 
    poly_803 = poly_1 * poly_385 
    poly_804 = poly_3 * poly_267 - poly_677 - poly_660 
    poly_805 = poly_3 * poly_268 - poly_678 
    poly_806 = poly_3 * poly_269 - poly_679 - poly_660 
    poly_807 = poly_1 * poly_270 
    poly_808 = poly_1 * poly_387 
    poly_809 = poly_22 * poly_48 
    poly_810 = poly_1 * poly_388 
    poly_811 = poly_24 * poly_48 - poly_675 
    poly_812 = poly_2 * poly_268 - poly_660 
    poly_813 = poly_2 * poly_269 - poly_659 - poly_802 
    poly_814 = poly_1 * poly_389 
    poly_815 = poly_47 * poly_52 
    poly_816 = poly_1 * poly_272 
    poly_817 = poly_1 * poly_391 
    poly_818 = poly_2 * poly_273 - poly_675 
    poly_819 = poly_1 * poly_274 
    poly_820 = poly_16 * poly_129 - poly_812 
    poly_821 = poly_2 * poly_277 - poly_679 - poly_667 
    poly_822 = poly_1 * poly_392 
    poly_823 = poly_9 * poly_109 - poly_668 - poly_793 
    poly_824 = poly_27 * poly_46 
    poly_825 = poly_9 * poly_111 - poly_668 - poly_796 
    poly_826 = poly_1 * poly_278 
    poly_827 = poly_47 * poly_54 
    poly_828 = poly_1 * poly_281 
    poly_829 = poly_3 * poly_384 - poly_802 - poly_801 
    poly_830 = poly_2 * poly_392 - poly_825 - poly_823 
    poly_831 = poly_1 * poly_393 
    poly_832 = poly_6 * poly_121 - poly_675 
    poly_833 = poly_9 * poly_171 - poly_812 
    poly_834 = poly_47 * poly_55 
    poly_835 = poly_1 * poly_395 
    poly_836 = poly_3 * poly_273 - poly_667 
    poly_837 = poly_1 * poly_396 
    poly_838 = poly_26 * poly_49 - poly_678 
    poly_839 = poly_27 * poly_49 
    poly_840 = poly_3 * poly_277 - poly_668 - poly_825 
    poly_841 = poly_1 * poly_397 
    poly_842 = poly_47 * poly_57 
    poly_843 = poly_47 * poly_58 
    poly_844 = poly_1 * poly_398 
    poly_845 = poly_6 * poly_172 - poly_836 
    poly_846 = poly_9 * poly_121 - poly_678 
    poly_847 = poly_47 * poly_59 
    poly_848 = poly_1 * poly_402 
    poly_849 = poly_1 * poly_404 
    poly_850 = poly_1 * poly_405 
    poly_851 = poly_22 * poly_67 
    poly_852 = poly_1 * poly_408 
    poly_853 = poly_1 * poly_410 
    poly_854 = poly_1 * poly_411 
    poly_855 = poly_4 * poly_244 - poly_667 - poly_660 
    poly_856 = poly_1 * poly_413 
    poly_857 = poly_1 * poly_414 
    poly_858 = poly_27 * poly_66 
    poly_859 = poly_1 * poly_416 
    poly_860 = poly_1 * poly_417 
    poly_861 = poly_12 * poly_103 - poly_677 - poly_790 
    poly_862 = poly_1 * poly_418 
    poly_863 = poly_22 * poly_69 
    poly_864 = poly_4 * poly_253 - poly_677 - poly_660 
    poly_865 = poly_1 * poly_419 
    poly_866 = poly_4 * poly_255 - poly_677 - poly_667 
    poly_867 = poly_27 * poly_68 
    poly_868 = poly_4 * poly_257 - poly_679 
    poly_869 = poly_1 * poly_422 
    poly_870 = poly_1 * poly_424 
    poly_871 = poly_1 * poly_425 
    poly_872 = poly_13 * poly_107 - poly_670 
    poly_873 = poly_1 * poly_427 
    poly_874 = poly_1 * poly_428 
    poly_875 = poly_12 * poly_110 - poly_672 
    poly_876 = poly_1 * poly_429 
    poly_877 = poly_4 * poly_262 - poly_681 - poly_872 
    poly_878 = poly_4 * poly_263 - poly_682 - poly_875 
    poly_879 = poly_1 * poly_431 
    poly_880 = poly_1 * poly_432 
    poly_881 = poly_47 * poly_33 
    poly_882 = poly_1 * poly_434 
    poly_883 = poly_1 * poly_435 
    poly_884 = poly_33 * poly_48 - poly_834 
    poly_885 = poly_1 * poly_436 
    poly_886 = poly_22 * poly_70 
    poly_887 = poly_16 * poly_91 - poly_670 
    poly_888 = poly_1 * poly_437 
    poly_889 = poly_12 * poly_115 - poly_884 - poly_827 
    poly_890 = poly_4 * poly_268 - poly_672 
    poly_891 = poly_4 * poly_269 - poly_681 - poly_884 
    poly_892 = poly_1 * poly_438 
    poly_893 = poly_47 * poly_35 
    poly_894 = poly_47 * poly_36 
    poly_895 = poly_47 * poly_37 
    poly_896 = poly_1 * poly_440 
    poly_897 = poly_1 * poly_441 
    poly_898 = poly_33 * poly_49 - poly_847 
    poly_899 = poly_1 * poly_442 
    poly_900 = poly_4 * poly_273 - poly_670 
    poly_901 = poly_13 * poly_118 - poly_898 - poly_842 
    poly_902 = poly_1 * poly_443 
    poly_903 = poly_16 * poly_99 - poly_672 
    poly_904 = poly_27 * poly_70 
    poly_905 = poly_4 * poly_277 - poly_682 - poly_898 
    poly_906 = poly_1 * poly_444 
    poly_907 = poly_47 * poly_39 
    poly_908 = poly_47 * poly_40 
    poly_909 = poly_47 * poly_41 
    poly_910 = poly_47 * poly_42 
    poly_911 = poly_1 * poly_445 
    poly_912 = poly_12 * poly_121 - poly_834 
    poly_913 = poly_13 * poly_121 - poly_847 
    poly_914 = poly_37 * poly_49 - poly_670 
    poly_915 = poly_42 * poly_48 - poly_672 
    poly_916 = poly_47 * poly_43 
    poly_917 = poly_1 * poly_285 
    poly_918 = poly_1 * poly_286 
    poly_919 = poly_1 * poly_449 
    poly_920 = poly_1 * poly_451 
    poly_921 = poly_1 * poly_287 
    poly_922 = poly_1 * poly_288 
    poly_923 = poly_1 * poly_454 
    poly_924 = poly_1 * poly_456 
    poly_925 = poly_1 * poly_289 
    poly_926 = poly_1 * poly_290 
    poly_927 = poly_1 * poly_458 
    poly_928 = poly_1 * poly_291 
    poly_929 = poly_1 * poly_292 
    poly_930 = poly_1 * poly_293 
    poly_931 = poly_22 * poly_24 
    poly_932 = poly_1 * poly_460 
    poly_933 = poly_22 * poly_54 
    poly_934 = poly_1 * poly_295 
    poly_935 = poly_6 * poly_129 - poly_689 
    poly_936 = poly_1 * poly_297 
    poly_937 = poly_1 * poly_462 
    poly_938 = poly_22 * poly_29 
    poly_939 = poly_1 * poly_299 
    poly_940 = poly_26 * poly_52 - poly_694 - poly_935 
    poly_941 = poly_6 * poly_84 - poly_694 
    poly_942 = poly_1 * poly_464 
    poly_943 = poly_22 * poly_55 
    poly_944 = poly_1 * poly_465 
    poly_945 = poly_2 * poly_300 - poly_694 - poly_940 
    poly_946 = poly_27 * poly_72 
    poly_947 = poly_18 * poly_84 - poly_689 
    poly_948 = poly_1 * poly_303 
    poly_949 = poly_1 * poly_468 
    poly_950 = poly_1 * poly_304 
    poly_951 = poly_1 * poly_470 
    poly_952 = poly_1 * poly_305 
    poly_953 = poly_1 * poly_306 
    poly_954 = poly_24 * poly_26 - poly_638 - poly_638 - poly_638 
    poly_955 = poly_1 * poly_471 
    poly_956 = poly_2 * poly_307 - poly_703 - poly_954 
    poly_957 = poly_27 * poly_54 
    poly_958 = poly_1 * poly_473 
    poly_959 = poly_1 * poly_474 
    poly_960 = poly_27 * poly_26 
    poly_961 = poly_1 * poly_309 
    poly_962 = poly_1 * poly_476 
    poly_963 = poly_22 * poly_57 
    poly_964 = poly_1 * poly_311 
    poly_965 = poly_10 * poly_129 - poly_946 
    poly_966 = poly_6 * poly_140 - poly_703 - poly_956 
    poly_967 = poly_1 * poly_477 
    poly_968 = poly_26 * poly_30 - poly_704 - poly_957 
    poly_969 = poly_27 * poly_30 
    poly_970 = poly_6 * poly_204 - poly_968 
    poly_971 = poly_1 * poly_479 
    poly_972 = poly_22 * poly_59 
    poly_973 = poly_1 * poly_480 
    poly_974 = poly_26 * poly_55 - poly_946 
    poly_975 = poly_27 * poly_55 
    poly_976 = poly_2 * poly_314 - poly_703 - poly_966 
    poly_977 = poly_1 * poly_482 
    poly_978 = poly_1 * poly_484 
    poly_979 = poly_1 * poly_485 
    poly_980 = poly_1 * poly_486 
    poly_981 = poly_9 * poly_136 - poly_698 
    poly_982 = poly_27 * poly_57 
    poly_983 = poly_1 * poly_488 
    poly_984 = poly_22 * poly_73 
    poly_985 = poly_1 * poly_489 
    poly_986 = poly_26 * poly_59 - poly_975 
    poly_987 = poly_27 * poly_59 
    poly_988 = poly_19 * poly_84 - poly_698 
    poly_989 = poly_1 * poly_315 
    poly_990 = poly_1 * poly_316 
    poly_991 = poly_1 * poly_494 
    poly_992 = poly_1 * poly_317 
    poly_993 = poly_1 * poly_496 
    poly_994 = poly_1 * poly_498 
    poly_995 = poly_1 * poly_318 
    poly_996 = poly_1 * poly_319 
    poly_997 = poly_1 * poly_500 
    poly_998 = poly_1 * poly_320 
    poly_999 = poly_6 * poly_147 - poly_711 
    poly_1000 = poly_1 * poly_322 
    poly_1001 = poly_6 * poly_149 - poly_713 
    poly_1002 = poly_1 * poly_501 
    poly_1003 = poly_3 * poly_323 - poly_741 - poly_726 
    poly_1004 = poly_1 * poly_324 
    poly_1005 = poly_22 * poly_61 
    poly_1006 = poly_3 * poly_326 - poly_744 
    poly_1007 = poly_1 * poly_327 
    poly_1008 = poly_1 * poly_503 
    poly_1009 = poly_1 * poly_328 
    poly_1010 = poly_33 * poly_52 - poly_717 - poly_999 
    poly_1011 = poly_1 * poly_504 
    poly_1012 = poly_3 * poly_329 - poly_753 - poly_729 
    poly_1013 = poly_1 * poly_330 
    poly_1014 = poly_22 * poly_35 
    poly_1015 = poly_2 * poly_326 - poly_713 
    poly_1016 = poly_1 * poly_505 
    poly_1017 = poly_22 * poly_37 
    poly_1018 = poly_6 * poly_91 - poly_717 
    poly_1019 = poly_1 * poly_507 
    poly_1020 = poly_1 * poly_508 
    poly_1021 = poly_2 * poly_329 - poly_717 - poly_1010 
    poly_1022 = poly_1 * poly_509 
    poly_1023 = poly_22 * poly_63 
    poly_1024 = poly_18 * poly_91 - poly_711 
    poly_1025 = poly_1 * poly_333 
    poly_1026 = poly_1 * poly_334 
    poly_1027 = poly_1 * poly_511 
    poly_1028 = poly_1 * poly_335 
    poly_1029 = poly_2 * poly_336 - poly_739 - poly_724 
    poly_1030 = poly_1 * poly_337 
    poly_1031 = poly_9 * poly_147 - poly_733 
    poly_1032 = poly_1 * poly_512 
    poly_1033 = poly_9 * poly_149 - poly_735 
    poly_1034 = poly_1 * poly_339 
    poly_1035 = poly_24 * poly_61 - poly_711 
    poly_1036 = poly_26 * poly_62 - poly_735 
    poly_1037 = poly_1 * poly_342 
    poly_1038 = poly_4 * poly_454 - poly_1018 
    poly_1039 = poly_13 * poly_129 - poly_733 
    poly_1040 = poly_1 * poly_513 
    poly_1041 = poly_2 * poly_343 - poly_746 - poly_1038 
    poly_1042 = poly_2 * poly_344 - poly_747 - poly_729 
    poly_1043 = poly_1 * poly_345 
    poly_1044 = poly_2 * poly_346 - poly_749 
    poly_1045 = poly_27 * poly_62 
    poly_1046 = poly_1 * poly_348 
    poly_1047 = poly_3 * poly_500 - poly_1035 - poly_999 
    poly_1048 = poly_2 * poly_512 - poly_1036 - poly_1033 
    poly_1049 = poly_4 * poly_460 - poly_1047 - poly_1021 
    poly_1050 = poly_1 * poly_352 
    poly_1051 = poly_2 * poly_349 - poly_743 - poly_1047 
    poly_1052 = poly_2 * poly_350 - poly_744 - poly_741 
    poly_1053 = poly_3 * poly_505 - poly_1018 
    poly_1054 = poly_1 * poly_514 
    poly_1055 = poly_2 * poly_353 - poly_755 - poly_1051 
    poly_1056 = poly_2 * poly_354 - poly_756 - poly_753 
    poly_1057 = poly_37 * poly_55 - poly_711 
    poly_1058 = poly_42 * poly_72 - poly_1044 
    poly_1059 = poly_1 * poly_357 
    poly_1060 = poly_1 * poly_516 
    poly_1061 = poly_1 * poly_358 
    poly_1062 = poly_2 * poly_359 - poly_773 - poly_764 
    poly_1063 = poly_1 * poly_517 
    poly_1064 = poly_13 * poly_140 - poly_770 - poly_1048 
    poly_1065 = poly_1 * poly_360 
    poly_1066 = poly_12 * poly_136 - poly_713 
    poly_1067 = poly_3 * poly_341 - poly_750 - poly_1036 
    poly_1068 = poly_1 * poly_518 
    poly_1069 = poly_2 * poly_361 - poly_775 - poly_1066 
    poly_1070 = poly_2 * poly_362 - poly_776 - poly_764 
    poly_1071 = poly_1 * poly_363 
    poly_1072 = poly_3 * poly_346 - poly_733 
    poly_1073 = poly_27 * poly_40 
    poly_1074 = poly_2 * poly_366 - poly_780 - poly_769 
    poly_1075 = poly_1 * poly_519 
    poly_1076 = poly_4 * poly_473 - poly_1067 
    poly_1077 = poly_27 * poly_42 
    poly_1078 = poly_1 * poly_367 
    poly_1079 = poly_3 * poly_349 - poly_749 - poly_739 
    poly_1080 = poly_2 * poly_517 - poly_1067 - poly_1064 
    poly_1081 = poly_3 * poly_351 - poly_760 - poly_747 
    poly_1082 = poly_2 * poly_519 - poly_1076 
    poly_1083 = poly_1 * poly_520 
    poly_1084 = poly_2 * poly_368 - poly_775 - poly_1079 
    poly_1085 = poly_2 * poly_369 - poly_776 - poly_773 
    poly_1086 = poly_37 * poly_59 - poly_713 
    poly_1087 = poly_42 * poly_55 - poly_733 
    poly_1088 = poly_1 * poly_522 
    poly_1089 = poly_1 * poly_523 
    poly_1090 = poly_3 * poly_359 - poly_767 - poly_1064 
    poly_1091 = poly_1 * poly_524 
    poly_1092 = poly_4 * poly_482 - poly_1006 
    poly_1093 = poly_3 * poly_362 - poly_770 - poly_1067 
    poly_1094 = poly_1 * poly_525 
    poly_1095 = poly_19 * poly_99 - poly_735 
    poly_1096 = poly_27 * poly_64 
    poly_1097 = poly_3 * poly_366 - poly_770 - poly_1082 
    poly_1098 = poly_1 * poly_526 
    poly_1099 = poly_3 * poly_368 - poly_778 - poly_773 
    poly_1100 = poly_2 * poly_523 - poly_1093 - poly_1090 
    poly_1101 = poly_37 * poly_73 - poly_1006 
    poly_1102 = poly_42 * poly_59 - poly_735 
    poly_1103 = poly_1 * poly_372 
    poly_1104 = poly_1 * poly_373 
    poly_1105 = poly_6 * poly_107 - poly_799 
    poly_1106 = poly_1 * poly_528 
    poly_1107 = poly_18 * poly_107 - poly_809 
    poly_1108 = poly_1 * poly_375 
    poly_1109 = poly_2 * poly_376 - poly_801 - poly_787 
    poly_1110 = poly_2 * poly_377 - poly_802 - poly_789 
    poly_1111 = poly_1 * poly_378 
    poly_1112 = poly_2 * poly_379 - poly_804 - poly_795 
    poly_1113 = poly_9 * poly_110 - poly_824 
    poly_1114 = poly_2 * poly_381 - poly_806 - poly_795 
    poly_1115 = poly_1 * poly_529 
    poly_1116 = poly_3 * poly_379 - poly_823 - poly_793 
    poly_1117 = poly_19 * poly_110 - poly_839 
    poly_1118 = poly_3 * poly_381 - poly_825 - poly_796 
    poly_1119 = poly_1 * poly_382 
    poly_1120 = poly_1 * poly_383 
    poly_1121 = poly_6 * poly_169 - poly_799 - poly_1107 
    poly_1122 = poly_2 * poly_529 - poly_1118 - poly_1116 
    poly_1123 = poly_1 * poly_386 
    poly_1124 = poly_16 * poly_196 - poly_1105 
    poly_1125 = poly_2 * poly_385 - poly_806 - poly_804 
    poly_1126 = poly_1 * poly_530 
    poly_1127 = poly_6 * poly_171 - poly_809 
    poly_1128 = poly_2 * poly_388 - poly_813 - poly_811 
    poly_1129 = poly_47 * poly_72 
    poly_1130 = poly_1 * poly_390 
    poly_1131 = poly_2 * poly_391 - poly_829 - poly_818 
    poly_1132 = poly_9 * poly_170 - poly_824 - poly_1117 
    poly_1133 = poly_2 * poly_393 - poly_832 
    poly_1134 = poly_1 * poly_394 
    poly_1135 = poly_3 * poly_391 - poly_821 - poly_820 
    poly_1136 = poly_16 * poly_204 - poly_1113 
    poly_1137 = poly_2 * poly_398 - poly_845 
    poly_1138 = poly_1 * poly_531 
    poly_1139 = poly_3 * poly_395 - poly_840 - poly_838 
    poly_1140 = poly_9 * poly_172 - poly_839 
    poly_1141 = poly_47 * poly_73 
    poly_1142 = poly_2 * poly_531 - poly_1139 
    poly_1143 = poly_1 * poly_399 
    poly_1144 = poly_1 * poly_535 
    poly_1145 = poly_1 * poly_537 
    poly_1146 = poly_1 * poly_400 
    poly_1147 = poly_1 * poly_539 
    poly_1148 = poly_1 * poly_401 
    poly_1149 = poly_6 * poly_175 - poly_851 
    poly_1150 = poly_1 * poly_540 
    poly_1151 = poly_3 * poly_402 - poly_861 - poly_855 
    poly_1152 = poly_1 * poly_403 
    poly_1153 = poly_22 * poly_66 
    poly_1154 = poly_3 * poly_405 - poly_864 
    poly_1155 = poly_1 * poly_542 
    poly_1156 = poly_1 * poly_543 
    poly_1157 = poly_4 * poly_329 - poly_813 - poly_787 
    poly_1158 = poly_1 * poly_544 
    poly_1159 = poly_22 * poly_68 
    poly_1160 = poly_2 * poly_405 - poly_851 
    poly_1161 = poly_1 * poly_406 
    poly_1162 = poly_1 * poly_546 
    poly_1163 = poly_1 * poly_407 
    poly_1164 = poly_2 * poly_408 - poly_861 - poly_855 
    poly_1165 = poly_1 * poly_547 
    poly_1166 = poly_9 * poly_175 - poly_858 
    poly_1167 = poly_1 * poly_409 
    poly_1168 = poly_24 * poly_66 - poly_851 
    poly_1169 = poly_26 * poly_67 - poly_858 
    poly_1170 = poly_1 * poly_548 
    poly_1171 = poly_2 * poly_410 - poly_863 - poly_1168 
    poly_1172 = poly_2 * poly_411 - poly_864 - poly_855 
    poly_1173 = poly_1 * poly_412 
    poly_1174 = poly_2 * poly_413 - poly_866 
    poly_1175 = poly_27 * poly_67 
    poly_1176 = poly_1 * poly_415 
    poly_1177 = poly_3 * poly_539 - poly_1168 - poly_1149 
    poly_1178 = poly_2 * poly_547 - poly_1169 - poly_1166 
    poly_1179 = poly_4 * poly_351 - poly_829 - poly_813 
    poly_1180 = poly_1 * poly_549 
    poly_1181 = poly_2 * poly_416 - poly_863 - poly_1177 
    poly_1182 = poly_2 * poly_417 - poly_864 - poly_861 
    poly_1183 = poly_10 * poly_179 - poly_851 
    poly_1184 = poly_18 * poly_184 - poly_1174 
    poly_1185 = poly_1 * poly_551 
    poly_1186 = poly_1 * poly_552 
    poly_1187 = poly_4 * poly_359 - poly_840 - poly_793 
    poly_1188 = poly_1 * poly_553 
    poly_1189 = poly_20 * poly_136 - poly_1154 
    poly_1190 = poly_3 * poly_411 - poly_867 - poly_1169 
    poly_1191 = poly_1 * poly_554 
    poly_1192 = poly_3 * poly_413 - poly_858 
    poly_1193 = poly_27 * poly_69 
    poly_1194 = poly_4 * poly_366 - poly_840 - poly_830 
    poly_1195 = poly_1 * poly_555 
    poly_1196 = poly_3 * poly_416 - poly_866 - poly_861 
    poly_1197 = poly_2 * poly_552 - poly_1190 - poly_1187 
    poly_1198 = poly_19 * poly_179 - poly_1154 
    poly_1199 = poly_10 * poly_184 - poly_858 
    poly_1200 = poly_1 * poly_420 
    poly_1201 = poly_1 * poly_557 
    poly_1202 = poly_1 * poly_421 
    poly_1203 = poly_16 * poly_147 - poly_834 
    poly_1204 = poly_1 * poly_558 
    poly_1205 = poly_16 * poly_149 - poly_847 
    poly_1206 = poly_1 * poly_423 
    poly_1207 = poly_12 * poly_107 - poly_662 - poly_662 - poly_662 
    poly_1208 = poly_3 * poly_425 - poly_901 - poly_878 
    poly_1209 = poly_1 * poly_559 
    poly_1210 = poly_2 * poly_424 - poly_886 - poly_1207 
    poly_1211 = poly_2 * poly_425 - poly_887 - poly_872 
    poly_1212 = poly_1 * poly_426 
    poly_1213 = poly_2 * poly_427 - poly_889 - poly_877 
    poly_1214 = poly_13 * poly_110 - poly_673 - poly_673 - poly_673 
    poly_1215 = poly_2 * poly_429 - poly_891 - poly_877 
    poly_1216 = poly_1 * poly_560 
    poly_1217 = poly_3 * poly_427 - poly_903 - poly_875 
    poly_1218 = poly_3 * poly_428 - poly_904 - poly_1214 
    poly_1219 = poly_3 * poly_429 - poly_905 - poly_878 
    poly_1220 = poly_1 * poly_430 
    poly_1221 = poly_47 * poly_61 
    poly_1222 = poly_47 * poly_62 
    poly_1223 = poly_1 * poly_433 
    poly_1224 = poly_48 * poly_61 - poly_1129 
    poly_1225 = poly_48 * poly_62 - poly_847 
    poly_1226 = poly_4 * poly_384 - poly_815 - poly_1224 
    poly_1227 = poly_2 * poly_560 - poly_1219 - poly_1217 
    poly_1228 = poly_1 * poly_561 
    poly_1229 = poly_12 * poly_171 - poly_1129 
    poly_1230 = poly_13 * poly_171 - poly_834 
    poly_1231 = poly_2 * poly_436 - poly_886 - poly_1226 
    poly_1232 = poly_2 * poly_437 - poly_891 - poly_889 
    poly_1233 = poly_47 * poly_63 
    poly_1234 = poly_1 * poly_439 
    poly_1235 = poly_49 * poly_61 - poly_834 
    poly_1236 = poly_49 * poly_62 - poly_1141 
    poly_1237 = poly_2 * poly_442 - poly_914 - poly_900 
    poly_1238 = poly_4 * poly_392 - poly_843 - poly_1236 
    poly_1239 = poly_4 * poly_393 - poly_834 
    poly_1240 = poly_1 * poly_562 
    poly_1241 = poly_12 * poly_172 - poly_847 
    poly_1242 = poly_13 * poly_172 - poly_1141 
    poly_1243 = poly_3 * poly_442 - poly_905 - poly_901 
    poly_1244 = poly_3 * poly_443 - poly_904 - poly_1238 
    poly_1245 = poly_47 * poly_64 
    poly_1246 = poly_4 * poly_398 - poly_847 
    poly_1247 = poly_1 * poly_564 
    poly_1248 = poly_16 * poly_107 - poly_893 
    poly_1249 = poly_1 * poly_565 
    poly_1250 = poly_16 * poly_109 - poly_907 - poly_894 
    poly_1251 = poly_16 * poly_110 - poly_908 
    poly_1252 = poly_16 * poly_111 - poly_916 - poly_881 - poly_881 
    poly_1253 = poly_1 * poly_566 
    poly_1254 = poly_47 * poly_45 
    poly_1255 = poly_47 * poly_46 
    poly_1256 = poly_1 * poly_567 
    poly_1257 = poly_6 * poly_220 - poly_1248 
    poly_1258 = poly_2 * poly_565 - poly_1252 - poly_1250 
    poly_1259 = poly_47 * poly_48 
    poly_1260 = poly_1 * poly_568 
    poly_1261 = poly_3 * poly_564 - poly_1252 - poly_1250 
    poly_1262 = poly_9 * poly_220 - poly_1251 
    poly_1263 = poly_47 * poly_49 
    poly_1264 = poly_2 * poly_568 - poly_1261 
    poly_1265 = poly_1 * poly_571 
    poly_1266 = poly_1 * poly_573 
    poly_1267 = poly_1 * poly_574 
    poly_1268 = poly_4 * poly_402 - poly_884 - poly_872 
    poly_1269 = poly_1 * poly_575 
    poly_1270 = poly_22 * poly_74 
    poly_1271 = poly_4 * poly_405 - poly_887 
    poly_1272 = poly_1 * poly_577 
    poly_1273 = poly_1 * poly_578 
    poly_1274 = poly_4 * poly_408 - poly_898 - poly_875 
    poly_1275 = poly_1 * poly_579 
    poly_1276 = poly_24 * poly_74 - poly_1271 
    poly_1277 = poly_4 * poly_411 - poly_901 - poly_890 
    poly_1278 = poly_1 * poly_580 
    poly_1279 = poly_4 * poly_413 - poly_903 
    poly_1280 = poly_27 * poly_74 
    poly_1281 = poly_1 * poly_581 
    poly_1282 = poly_3 * poly_573 - poly_1276 - poly_1268 
    poly_1283 = poly_2 * poly_578 - poly_1277 - poly_1274 
    poly_1284 = poly_3 * poly_575 - poly_1271 
    poly_1285 = poly_2 * poly_580 - poly_1279 
    poly_1286 = poly_1 * poly_583 
    poly_1287 = poly_1 * poly_584 
    poly_1288 = poly_16 * poly_175 - poly_881 
    poly_1289 = poly_1 * poly_585 
    poly_1290 = poly_20 * poly_107 - poly_895 
    poly_1291 = poly_4 * poly_425 - poly_894 - poly_1250 
    poly_1292 = poly_1 * poly_586 
    poly_1293 = poly_4 * poly_427 - poly_907 - poly_1250 
    poly_1294 = poly_20 * poly_110 - poly_910 
    poly_1295 = poly_4 * poly_429 - poly_916 - poly_1252 
    poly_1296 = poly_1 * poly_587 
    poly_1297 = poly_47 * poly_66 
    poly_1298 = poly_47 * poly_67 
    poly_1299 = poly_1 * poly_588 
    poly_1300 = poly_48 * poly_66 - poly_1221 
    poly_1301 = poly_48 * poly_67 - poly_881 
    poly_1302 = poly_16 * poly_179 - poly_895 
    poly_1303 = poly_2 * poly_586 - poly_1295 - poly_1293 
    poly_1304 = poly_47 * poly_68 
    poly_1305 = poly_1 * poly_589 
    poly_1306 = poly_49 * poly_66 - poly_881 
    poly_1307 = poly_49 * poly_67 - poly_1222 
    poly_1308 = poly_3 * poly_585 - poly_1295 - poly_1291 
    poly_1309 = poly_16 * poly_184 - poly_910 
    poly_1310 = poly_47 * poly_69 
    poly_1311 = poly_20 * poly_121 - poly_881 
    poly_1312 = poly_1 * poly_446 
    poly_1313 = poly_1 * poly_447 
    poly_1314 = poly_1 * poly_448 
    poly_1315 = poly_22 * poly_22 
    poly_1316 = poly_1 * poly_450 
    poly_1317 = poly_1 * poly_593 
    poly_1318 = poly_22 * poly_52 
    poly_1319 = poly_1 * poly_595 
    poly_1320 = poly_22 * poly_72 
    poly_1321 = poly_1 * poly_452 
    poly_1322 = poly_1 * poly_453 
    poly_1323 = poly_1 * poly_455 
    poly_1324 = poly_2 * poly_454 - poly_931 
    poly_1325 = poly_1 * poly_597 
    poly_1326 = poly_24 * poly_72 - poly_943 
    poly_1327 = poly_1 * poly_457 
    poly_1328 = poly_2 * poly_458 - poly_935 
    poly_1329 = poly_1 * poly_459 
    poly_1330 = poly_6 * poly_200 - poly_933 - poly_1326 
    poly_1331 = poly_1 * poly_461 
    poly_1332 = poly_2 * poly_460 - poly_933 - poly_1330 
    poly_1333 = poly_1 * poly_463 
    poly_1334 = poly_10 * poly_196 - poly_931 
    poly_1335 = poly_1 * poly_598 
    poly_1336 = poly_6 * poly_201 - poly_943 
    poly_1337 = poly_9 * poly_228 - poly_1328 
    poly_1338 = poly_1 * poly_466 
    poly_1339 = poly_1 * poly_467 
    poly_1340 = poly_3 * poly_454 - poly_940 
    poly_1341 = poly_1 * poly_600 
    poly_1342 = poly_18 * poly_136 - poly_972 
    poly_1343 = poly_1 * poly_469 
    poly_1344 = poly_3 * poly_458 - poly_946 
    poly_1345 = poly_2 * poly_471 - poly_966 - poly_956 
    poly_1346 = poly_1 * poly_472 
    poly_1347 = poly_2 * poly_473 - poly_968 
    poly_1348 = poly_27 * poly_27 
    poly_1349 = poly_1 * poly_475 
    poly_1350 = poly_3 * poly_460 - poly_947 - poly_935 
    poly_1351 = poly_1 * poly_478 
    poly_1352 = poly_19 * poly_196 - poly_1340 
    poly_1353 = poly_18 * poly_204 - poly_1347 
    poly_1354 = poly_1 * poly_601 
    poly_1355 = poly_6 * poly_205 - poly_972 
    poly_1356 = poly_9 * poly_201 - poly_946 
    poly_1357 = poly_1 * poly_481 
    poly_1358 = poly_1 * poly_603 
    poly_1359 = poly_2 * poly_482 - poly_984 
    poly_1360 = poly_1 * poly_483 
    poly_1361 = poly_19 * poly_129 - poly_975 
    poly_1362 = poly_2 * poly_486 - poly_988 - poly_981 
    poly_1363 = poly_1 * poly_604 
    poly_1364 = poly_3 * poly_473 - poly_960 
    poly_1365 = poly_27 * poly_58 
    poly_1366 = poly_8 * poly_204 - poly_969 - poly_1364 
    poly_1367 = poly_1 * poly_487 
    poly_1368 = poly_3 * poly_476 - poly_966 - poly_965 
    poly_1369 = poly_10 * poly_204 - poly_960 
    poly_1370 = poly_1 * poly_605 
    poly_1371 = poly_6 * poly_209 - poly_984 
    poly_1372 = poly_9 * poly_205 - poly_975 
    poly_1373 = poly_1 * poly_607 
    poly_1374 = poly_3 * poly_482 - poly_981 
    poly_1375 = poly_1 * poly_608 
    poly_1376 = poly_26 * poly_73 - poly_987 
    poly_1377 = poly_27 * poly_73 
    poly_1378 = poly_3 * poly_486 - poly_982 - poly_1366 
    poly_1379 = poly_1 * poly_609 
    poly_1380 = poly_6 * poly_229 - poly_1374 
    poly_1381 = poly_9 * poly_209 - poly_987 
    poly_1382 = poly_1 * poly_490 
    poly_1383 = poly_1 * poly_491 
    poly_1384 = poly_1 * poly_492 
    poly_1385 = poly_1 * poly_611 
    poly_1386 = poly_1 * poly_493 
    poly_1387 = poly_2 * poly_494 - poly_999 
    poly_1388 = poly_1 * poly_495 
    poly_1389 = poly_2 * poly_496 - poly_1001 
    poly_1390 = poly_1 * poly_497 
    poly_1391 = poly_2 * poly_498 - poly_1003 
    poly_1392 = poly_1 * poly_612 
    poly_1393 = poly_3 * poly_498 - poly_1033 
    poly_1394 = poly_1 * poly_499 
    poly_1395 = poly_6 * poly_211 - poly_1005 
    poly_1396 = poly_2 * poly_612 - poly_1393 
    poly_1397 = poly_1 * poly_502 
    poly_1398 = poly_12 * poly_196 - poly_1017 
    poly_1399 = poly_18 * poly_212 - poly_1391 
    poly_1400 = poly_1 * poly_506 
    poly_1401 = poly_2 * poly_503 - poly_1014 - poly_1398 
    poly_1402 = poly_62 * poly_72 - poly_1389 
    poly_1403 = poly_2 * poly_505 - poly_1017 
    poly_1404 = poly_1 * poly_613 
    poly_1405 = poly_2 * poly_507 - poly_1023 - poly_1401 
    poly_1406 = poly_13 * poly_228 - poly_1387 
    poly_1407 = poly_37 * poly_72 - poly_1005 
    poly_1408 = poly_1 * poly_510 
    poly_1409 = poly_3 * poly_611 - poly_1387 
    poly_1410 = poly_9 * poly_212 - poly_1045 
    poly_1411 = poly_2 * poly_513 - poly_1049 - poly_1041 
    poly_1412 = poly_4 * poly_598 - poly_1387 
    poly_1413 = poly_1 * poly_515 
    poly_1414 = poly_19 * poly_211 - poly_1389 
    poly_1415 = poly_13 * poly_204 - poly_1077 
    poly_1416 = poly_2 * poly_518 - poly_1081 - poly_1069 
    poly_1417 = poly_4 * poly_601 - poly_1389 
    poly_1418 = poly_1 * poly_521 
    poly_1419 = poly_61 * poly_73 - poly_1391 
    poly_1420 = poly_3 * poly_517 - poly_1073 - poly_1415 
    poly_1421 = poly_2 * poly_524 - poly_1101 - poly_1092 
    poly_1422 = poly_3 * poly_519 - poly_1077 
    poly_1423 = poly_4 * poly_605 - poly_1391 
    poly_1424 = poly_1 * poly_614 
    poly_1425 = poly_12 * poly_229 - poly_1393 
    poly_1426 = poly_3 * poly_523 - poly_1096 - poly_1420 
    poly_1427 = poly_3 * poly_524 - poly_1097 - poly_1093 
    poly_1428 = poly_42 * poly_73 - poly_1045 
    poly_1429 = poly_4 * poly_609 - poly_1393 
    poly_1430 = poly_1 * poly_527 
    poly_1431 = poly_2 * poly_528 - poly_1121 - poly_1107 
    poly_1432 = poly_3 * poly_529 - poly_1132 - poly_1117 
    poly_1433 = poly_2 * poly_530 - poly_1127 
    poly_1434 = poly_3 * poly_531 - poly_1140 
    poly_1435 = poly_1 * poly_532 
    poly_1436 = poly_1 * poly_533 
    poly_1437 = poly_1 * poly_616 
    poly_1438 = poly_1 * poly_534 
    poly_1439 = poly_2 * poly_535 - poly_1149 
    poly_1440 = poly_1 * poly_536 
    poly_1441 = poly_2 * poly_537 - poly_1151 
    poly_1442 = poly_1 * poly_617 
    poly_1443 = poly_3 * poly_537 - poly_1166 
    poly_1444 = poly_1 * poly_538 
    poly_1445 = poly_6 * poly_216 - poly_1153 
    poly_1446 = poly_2 * poly_617 - poly_1443 
    poly_1447 = poly_1 * poly_541 
    poly_1448 = poly_52 * poly_66 - poly_1159 - poly_1445 
    poly_1449 = poly_18 * poly_217 - poly_1441 
    poly_1450 = poly_4 * poly_505 - poly_1124 
    poly_1451 = poly_1 * poly_618 
    poly_1452 = poly_2 * poly_542 - poly_1159 - poly_1448 
    poly_1453 = poly_67 * poly_72 - poly_1439 
    poly_1454 = poly_18 * poly_179 - poly_1153 
    poly_1455 = poly_1 * poly_545 
    poly_1456 = poly_3 * poly_616 - poly_1439 
    poly_1457 = poly_9 * poly_217 - poly_1175 
    poly_1458 = poly_2 * poly_548 - poly_1179 - poly_1171 
    poly_1459 = poly_20 * poly_201 - poly_1439 
    poly_1460 = poly_1 * poly_550 
    poly_1461 = poly_19 * poly_216 - poly_1441 
    poly_1462 = poly_58 * poly_67 - poly_1193 - poly_1457 
    poly_1463 = poly_2 * poly_553 - poly_1198 - poly_1189 
    poly_1464 = poly_4 * poly_519 - poly_1136 
    poly_1465 = poly_20 * poly_205 - poly_1441 
    poly_1466 = poly_1 * poly_619 
    poly_1467 = poly_66 * poly_73 - poly_1443 
    poly_1468 = poly_3 * poly_552 - poly_1193 - poly_1462 
    poly_1469 = poly_3 * poly_553 - poly_1194 - poly_1190 
    poly_1470 = poly_19 * poly_184 - poly_1175 
    poly_1471 = poly_20 * poly_209 - poly_1443 
    poly_1472 = poly_1 * poly_556 
    poly_1473 = poly_16 * poly_211 - poly_1129 
    poly_1474 = poly_16 * poly_212 - poly_1141 
    poly_1475 = poly_2 * poly_559 - poly_1226 - poly_1210 
    poly_1476 = poly_3 * poly_560 - poly_1238 - poly_1218 
    poly_1477 = poly_4 * poly_530 - poly_1129 
    poly_1478 = poly_4 * poly_531 - poly_1141 
    poly_1479 = poly_1 * poly_563 
    poly_1480 = poly_2 * poly_564 - poly_1257 - poly_1248 - poly_1248 
    poly_1481 = poly_3 * poly_565 - poly_1262 - poly_1251 - poly_1251 
    poly_1482 = poly_47 * poly_47 
    poly_1483 = poly_2 * poly_567 - poly_1257 
    poly_1484 = poly_3 * poly_568 - poly_1262 
    poly_1485 = poly_1 * poly_569 
    poly_1486 = poly_1 * poly_621 
    poly_1487 = poly_1 * poly_570 
    poly_1488 = poly_2 * poly_571 - poly_1268 
    poly_1489 = poly_1 * poly_622 
    poly_1490 = poly_3 * poly_571 - poly_1274 
    poly_1491 = poly_1 * poly_572 
    poly_1492 = poly_6 * poly_222 - poly_1270 
    poly_1493 = poly_2 * poly_622 - poly_1490 
    poly_1494 = poly_1 * poly_623 
    poly_1495 = poly_4 * poly_542 - poly_1229 - poly_1210 
    poly_1496 = poly_18 * poly_223 - poly_1488 
    poly_1497 = poly_2 * poly_575 - poly_1270 
    poly_1498 = poly_1 * poly_576 
    poly_1499 = poly_3 * poly_621 - poly_1488 
    poly_1500 = poly_9 * poly_223 - poly_1280 
    poly_1501 = poly_2 * poly_579 - poly_1284 - poly_1276 
    poly_1502 = poly_55 * poly_74 - poly_1488 
    poly_1503 = poly_1 * poly_624 
    poly_1504 = poly_19 * poly_222 - poly_1490 
    poly_1505 = poly_4 * poly_552 - poly_1242 - poly_1218 
    poly_1506 = poly_3 * poly_579 - poly_1285 - poly_1277 
    poly_1507 = poly_3 * poly_580 - poly_1280 
    poly_1508 = poly_59 * poly_74 - poly_1490 
    poly_1509 = poly_1 * poly_582 
    poly_1510 = poly_16 * poly_216 - poly_1221 
    poly_1511 = poly_16 * poly_217 - poly_1222 
    poly_1512 = poly_2 * poly_585 - poly_1302 - poly_1290 
    poly_1513 = poly_3 * poly_586 - poly_1309 - poly_1294 
    poly_1514 = poly_20 * poly_171 - poly_1221 
    poly_1515 = poly_20 * poly_172 - poly_1222 
    poly_1516 = poly_1 * poly_625 
    poly_1517 = poly_12 * poly_220 - poly_1259 
    poly_1518 = poly_13 * poly_220 - poly_1263 
    poly_1519 = poly_4 * poly_564 - poly_1254 - poly_1517 
    poly_1520 = poly_4 * poly_565 - poly_1255 - poly_1518 
    poly_1521 = poly_47 * poly_70 
    poly_1522 = poly_4 * poly_567 - poly_1259 
    poly_1523 = poly_4 * poly_568 - poly_1263 
    poly_1524 = poly_1 * poly_627 
    poly_1525 = poly_1 * poly_628 
    poly_1526 = poly_4 * poly_571 - poly_1288 
    poly_1527 = poly_1 * poly_629 
    poly_1528 = poly_4 * poly_573 - poly_1300 - poly_1290 
    poly_1529 = poly_2 * poly_628 - poly_1526 
    poly_1530 = poly_4 * poly_575 - poly_1302 
    poly_1531 = poly_1 * poly_630 
    poly_1532 = poly_3 * poly_627 - poly_1526 
    poly_1533 = poly_4 * poly_578 - poly_1307 - poly_1294 
    poly_1534 = poly_4 * poly_579 - poly_1308 - poly_1303 
    poly_1535 = poly_4 * poly_580 - poly_1309 
    poly_1536 = poly_10 * poly_230 - poly_1526 
    poly_1537 = poly_1 * poly_631 
    poly_1538 = poly_16 * poly_222 - poly_1297 
    poly_1539 = poly_16 * poly_223 - poly_1298 
    poly_1540 = poly_4 * poly_585 - poly_1304 - poly_1519 
    poly_1541 = poly_4 * poly_586 - poly_1310 - poly_1520 
    poly_1542 = poly_47 * poly_74 
    poly_1543 = poly_48 * poly_74 - poly_1297 
    poly_1544 = poly_49 * poly_74 - poly_1298 
    poly_1545 = poly_1 * poly_590 
    poly_1546 = poly_1 * poly_591 
    poly_1547 = poly_1 * poly_592 
    poly_1548 = poly_6 * poly_196 - poly_1318 
    poly_1549 = poly_1 * poly_594 
    poly_1550 = poly_2 * poly_593 - poly_1318 - poly_1548 - poly_1548 
    poly_1551 = poly_1 * poly_633 
    poly_1552 = poly_6 * poly_228 - poly_1320 
    poly_1553 = poly_1 * poly_596 
    poly_1554 = poly_2 * poly_597 - poly_1330 - poly_1326 
    poly_1555 = poly_2 * poly_598 - poly_1336 
    poly_1556 = poly_1 * poly_599 
    poly_1557 = poly_2 * poly_600 - poly_1350 - poly_1342 
    poly_1558 = poly_2 * poly_601 - poly_1355 
    poly_1559 = poly_1 * poly_602 
    poly_1560 = poly_2 * poly_603 - poly_1368 - poly_1359 
    poly_1561 = poly_9 * poly_204 - poly_1365 
    poly_1562 = poly_2 * poly_605 - poly_1371 
    poly_1563 = poly_1 * poly_606 
    poly_1564 = poly_3 * poly_603 - poly_1362 - poly_1361 
    poly_1565 = poly_3 * poly_604 - poly_1365 - poly_1561 - poly_1561 
    poly_1566 = poly_2 * poly_609 - poly_1380 
    poly_1567 = poly_1 * poly_634 
    poly_1568 = poly_3 * poly_607 - poly_1378 - poly_1376 
    poly_1569 = poly_9 * poly_229 - poly_1377 
    poly_1570 = poly_2 * poly_634 - poly_1568 
    poly_1571 = poly_1 * poly_610 
    poly_1572 = poly_2 * poly_611 - poly_1395 
    poly_1573 = poly_3 * poly_612 - poly_1410 
    poly_1574 = poly_4 * poly_633 - poly_1572 
    poly_1575 = poly_4 * poly_634 - poly_1573 
    poly_1576 = poly_1 * poly_615 
    poly_1577 = poly_2 * poly_616 - poly_1445 
    poly_1578 = poly_3 * poly_617 - poly_1457 
    poly_1579 = poly_20 * poly_228 - poly_1577 
    poly_1580 = poly_20 * poly_229 - poly_1578 
    poly_1581 = poly_1 * poly_620 
    poly_1582 = poly_2 * poly_621 - poly_1492 
    poly_1583 = poly_3 * poly_622 - poly_1500 
    poly_1584 = poly_72 * poly_74 - poly_1582 
    poly_1585 = poly_73 * poly_74 - poly_1583 
    poly_1586 = poly_16 * poly_220 - poly_1521 
    poly_1587 = poly_1 * poly_626 
    poly_1588 = poly_2 * poly_627 - poly_1528 
    poly_1589 = poly_3 * poly_628 - poly_1533 
    poly_1590 = poly_18 * poly_230 - poly_1588 
    poly_1591 = poly_19 * poly_230 - poly_1589 
    poly_1592 = poly_4 * poly_625 - poly_1521 - poly_1586 - poly_1586 
    poly_1593 = poly_1 * poly_635 
    poly_1594 = poly_4 * poly_627 - poly_1538 
    poly_1595 = poly_4 * poly_628 - poly_1539 
    poly_1596 = poly_2 * poly_635 - poly_1594 
    poly_1597 = poly_3 * poly_635 - poly_1595 
    poly_1598 = poly_16 * poly_230 - poly_1542 
    poly_1599 = poly_1 * poly_632 
    poly_1600 = poly_2 * poly_633 - poly_1552 
    poly_1601 = poly_3 * poly_634 - poly_1569 
    poly_1602 = poly_4 * poly_635 - poly_1598 

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
    poly_1601,    poly_1602,    ]) 

    return poly 



