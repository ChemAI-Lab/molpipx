import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3B2.monomials_MOL_3_2_7 import f_monomials as f_monos 

# File created from ./MOL_3_2_7.POLY 

N_POLYS = 2022

# Total number of monomials = 2022 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(2022) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) + jnp.take(mono,3) + jnp.take(mono,4) + jnp.take(mono,5) + jnp.take(mono,6) + jnp.take(mono,7) 
    poly_3 = jnp.take(mono,8) + jnp.take(mono,9) + jnp.take(mono,10) 
    poly_4 = poly_1 * poly_2 
    poly_5 = jnp.take(mono,11) + jnp.take(mono,12) + jnp.take(mono,13) + jnp.take(mono,14) + jnp.take(mono,15) + jnp.take(mono,16) 
    poly_6 = jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) + jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) 
    poly_7 = jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) 
    poly_8 = poly_1 * poly_3 
    poly_9 = jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) + jnp.take(mono,29) + jnp.take(mono,30) + jnp.take(mono,31) 
    poly_10 = poly_2 * poly_3 - poly_9 
    poly_11 = jnp.take(mono,32) + jnp.take(mono,33) + jnp.take(mono,34) 
    poly_12 = poly_1 * poly_1 
    poly_13 = poly_2 * poly_2 - poly_7 - poly_6 - poly_5 - poly_7 - poly_6 - poly_5 
    poly_14 = poly_3 * poly_3 - poly_11 - poly_11 
    poly_15 = poly_1 * poly_5 
    poly_16 = poly_1 * poly_6 
    poly_17 = jnp.take(mono,35) + jnp.take(mono,36) + jnp.take(mono,37) + jnp.take(mono,38) + jnp.take(mono,39) + jnp.take(mono,40) 
    poly_18 = jnp.take(mono,41) + jnp.take(mono,42) 
    poly_19 = poly_1 * poly_7 
    poly_20 = jnp.take(mono,43) + jnp.take(mono,44) + jnp.take(mono,45) + jnp.take(mono,46) + jnp.take(mono,47) + jnp.take(mono,48) + jnp.take(mono,49) + jnp.take(mono,50) + jnp.take(mono,51) + jnp.take(mono,52) + jnp.take(mono,53) + jnp.take(mono,54) 
    poly_21 = poly_1 * poly_9 
    poly_22 = jnp.take(mono,55) + jnp.take(mono,56) + jnp.take(mono,57) 
    poly_23 = poly_1 * poly_10 
    poly_24 = jnp.take(mono,58) + jnp.take(mono,59) + jnp.take(mono,60) + jnp.take(mono,61) + jnp.take(mono,62) + jnp.take(mono,63) + jnp.take(mono,64) + jnp.take(mono,65) + jnp.take(mono,66) + jnp.take(mono,67) + jnp.take(mono,68) + jnp.take(mono,69) 
    poly_25 = jnp.take(mono,70) + jnp.take(mono,71) + jnp.take(mono,72) + jnp.take(mono,73) + jnp.take(mono,74) + jnp.take(mono,75) + jnp.take(mono,76) + jnp.take(mono,77) + jnp.take(mono,78) + jnp.take(mono,79) + jnp.take(mono,80) + jnp.take(mono,81) 
    poly_26 = poly_3 * poly_5 - poly_24 
    poly_27 = poly_3 * poly_6 - poly_25 
    poly_28 = poly_3 * poly_7 - poly_22 
    poly_29 = poly_1 * poly_11 
    poly_30 = jnp.take(mono,82) + jnp.take(mono,83) + jnp.take(mono,84) + jnp.take(mono,85) + jnp.take(mono,86) + jnp.take(mono,87) + jnp.take(mono,88) + jnp.take(mono,89) + jnp.take(mono,90) + jnp.take(mono,91) + jnp.take(mono,92) + jnp.take(mono,93) 
    poly_31 = jnp.take(mono,94) 
    poly_32 = poly_2 * poly_11 - poly_30 
    poly_33 = poly_1 * poly_4 
    poly_34 = poly_1 * poly_13 
    poly_35 = poly_2 * poly_5 - poly_20 - poly_17 - poly_17 
    poly_36 = poly_2 * poly_6 - poly_20 - poly_18 - poly_17 - poly_18 - poly_18 
    poly_37 = poly_2 * poly_7 - poly_20 
    poly_38 = poly_1 * poly_8 
    poly_39 = poly_2 * poly_9 - poly_25 - poly_24 - poly_22 - poly_22 
    poly_40 = poly_3 * poly_13 - poly_39 
    poly_41 = poly_1 * poly_14 
    poly_42 = poly_3 * poly_9 - poly_30 
    poly_43 = poly_2 * poly_14 - poly_42 
    poly_44 = poly_3 * poly_11 - poly_31 - poly_31 - poly_31 
    poly_45 = poly_1 * poly_12 
    poly_46 = poly_2 * poly_13 - poly_37 - poly_36 - poly_35 
    poly_47 = poly_3 * poly_14 - poly_44 
    poly_48 = poly_1 * poly_17 
    poly_49 = poly_1 * poly_18 
    poly_50 = poly_1 * poly_20 
    poly_51 = jnp.take(mono,95) + jnp.take(mono,96) + jnp.take(mono,97) + jnp.take(mono,98) + jnp.take(mono,99) + jnp.take(mono,100) 
    poly_52 = jnp.take(mono,101) + jnp.take(mono,102) + jnp.take(mono,103) + jnp.take(mono,104) + jnp.take(mono,105) + jnp.take(mono,106) 
    poly_53 = jnp.take(mono,107) + jnp.take(mono,108) + jnp.take(mono,109) 
    poly_54 = poly_1 * poly_22 
    poly_55 = poly_1 * poly_24 
    poly_56 = poly_1 * poly_25 
    poly_57 = jnp.take(mono,110) + jnp.take(mono,111) + jnp.take(mono,112) + jnp.take(mono,113) + jnp.take(mono,114) + jnp.take(mono,115) + jnp.take(mono,116) + jnp.take(mono,117) + jnp.take(mono,118) + jnp.take(mono,119) + jnp.take(mono,120) + jnp.take(mono,121) 
    poly_58 = poly_1 * poly_26 
    poly_59 = jnp.take(mono,122) + jnp.take(mono,123) + jnp.take(mono,124) + jnp.take(mono,125) + jnp.take(mono,126) + jnp.take(mono,127) + jnp.take(mono,128) + jnp.take(mono,129) + jnp.take(mono,130) + jnp.take(mono,131) + jnp.take(mono,132) + jnp.take(mono,133) 
    poly_60 = poly_1 * poly_27 
    poly_61 = poly_3 * poly_17 - poly_59 
    poly_62 = poly_3 * poly_18 
    poly_63 = poly_1 * poly_28 
    poly_64 = jnp.take(mono,134) + jnp.take(mono,135) + jnp.take(mono,136) + jnp.take(mono,137) + jnp.take(mono,138) + jnp.take(mono,139) + jnp.take(mono,140) + jnp.take(mono,141) + jnp.take(mono,142) + jnp.take(mono,143) + jnp.take(mono,144) + jnp.take(mono,145) 
    poly_65 = poly_3 * poly_20 - poly_64 - poly_57 
    poly_66 = poly_1 * poly_30 
    poly_67 = jnp.take(mono,146) + jnp.take(mono,147) + jnp.take(mono,148) + jnp.take(mono,149) + jnp.take(mono,150) + jnp.take(mono,151) 
    poly_68 = jnp.take(mono,152) + jnp.take(mono,153) + jnp.take(mono,154) + jnp.take(mono,155) + jnp.take(mono,156) + jnp.take(mono,157) 
    poly_69 = jnp.take(mono,158) + jnp.take(mono,159) + jnp.take(mono,160) + jnp.take(mono,161) + jnp.take(mono,162) + jnp.take(mono,163) 
    poly_70 = poly_1 * poly_31 
    poly_71 = poly_1 * poly_32 
    poly_72 = poly_5 * poly_11 - poly_67 
    poly_73 = poly_6 * poly_11 - poly_68 
    poly_74 = poly_31 * poly_2 
    poly_75 = poly_7 * poly_11 - poly_69 
    poly_76 = poly_1 * poly_15 
    poly_77 = poly_1 * poly_16 
    poly_78 = poly_1 * poly_19 
    poly_79 = poly_1 * poly_35 
    poly_80 = jnp.take(mono,164) + jnp.take(mono,165) + jnp.take(mono,166) + jnp.take(mono,167) + jnp.take(mono,168) + jnp.take(mono,169) 
    poly_81 = poly_1 * poly_36 
    poly_82 = poly_17 * poly_2 - poly_52 - poly_51 - poly_80 - poly_51 
    poly_83 = poly_2 * poly_18 - poly_52 
    poly_84 = poly_5 * poly_6 - poly_52 - poly_82 - poly_52 
    poly_85 = poly_1 * poly_37 
    poly_86 = poly_5 * poly_7 - poly_51 
    poly_87 = poly_6 * poly_7 - poly_52 
    poly_88 = poly_1 * poly_21 
    poly_89 = poly_1 * poly_39 
    poly_90 = poly_2 * poly_22 - poly_57 
    poly_91 = poly_1 * poly_23 
    poly_92 = poly_5 * poly_9 - poly_59 - poly_57 
    poly_93 = poly_6 * poly_9 - poly_62 - poly_61 - poly_57 
    poly_94 = poly_1 * poly_40 
    poly_95 = poly_2 * poly_24 - poly_64 - poly_59 - poly_61 - poly_57 - poly_92 - poly_61 
    poly_96 = poly_2 * poly_25 - poly_64 - poly_62 - poly_59 - poly_57 - poly_93 - poly_62 
    poly_97 = poly_2 * poly_26 - poly_65 - poly_59 
    poly_98 = poly_3 * poly_36 - poly_96 - poly_93 
    poly_99 = poly_3 * poly_37 - poly_90 
    poly_100 = poly_1 * poly_29 
    poly_101 = poly_2 * poly_30 - poly_73 - poly_72 - poly_69 - poly_68 - poly_67 - poly_69 - poly_68 - poly_67 
    poly_102 = poly_11 * poly_13 - poly_101 
    poly_103 = poly_1 * poly_42 
    poly_104 = poly_3 * poly_22 - poly_69 
    poly_105 = poly_1 * poly_43 
    poly_106 = poly_3 * poly_24 - poly_72 - poly_67 - poly_67 
    poly_107 = poly_3 * poly_25 - poly_73 - poly_68 - poly_68 
    poly_108 = poly_3 * poly_26 - poly_72 
    poly_109 = poly_3 * poly_27 - poly_73 
    poly_110 = poly_7 * poly_14 - poly_104 
    poly_111 = poly_1 * poly_44 
    poly_112 = poly_9 * poly_11 - poly_74 
    poly_113 = poly_3 * poly_30 - poly_74 - poly_112 - poly_74 
    poly_114 = poly_31 * poly_3 
    poly_115 = poly_3 * poly_32 - poly_74 
    poly_116 = poly_1 * poly_33 
    poly_117 = poly_1 * poly_34 
    poly_118 = poly_5 * poly_5 - poly_53 - poly_51 - poly_80 - poly_53 - poly_51 - poly_80 
    poly_119 = poly_6 * poly_6 - poly_53 - poly_51 - poly_83 - poly_53 - poly_51 - poly_83 
    poly_120 = poly_7 * poly_7 - poly_53 - poly_53 
    poly_121 = poly_1 * poly_46 
    poly_122 = poly_5 * poly_13 - poly_87 - poly_82 
    poly_123 = poly_6 * poly_13 - poly_86 - poly_83 - poly_80 
    poly_124 = poly_7 * poly_13 - poly_84 
    poly_125 = poly_1 * poly_38 
    poly_126 = poly_2 * poly_39 - poly_93 - poly_92 - poly_90 
    poly_127 = poly_3 * poly_46 - poly_126 
    poly_128 = poly_1 * poly_41 
    poly_129 = poly_3 * poly_39 - poly_101 
    poly_130 = poly_13 * poly_14 - poly_129 
    poly_131 = poly_11 * poly_11 - poly_114 - poly_114 
    poly_132 = poly_1 * poly_47 
    poly_133 = poly_3 * poly_42 - poly_112 
    poly_134 = poly_2 * poly_47 - poly_133 
    poly_135 = poly_11 * poly_14 - poly_114 
    poly_136 = poly_1 * poly_45 
    poly_137 = poly_2 * poly_46 - poly_124 - poly_123 - poly_122 
    poly_138 = poly_3 * poly_47 - poly_135 
    poly_139 = poly_1 * poly_51 
    poly_140 = poly_1 * poly_52 
    poly_141 = poly_1 * poly_53 
    poly_142 = jnp.take(mono,170) + jnp.take(mono,171) + jnp.take(mono,172) + jnp.take(mono,173) + jnp.take(mono,174) + jnp.take(mono,175) 
    poly_143 = poly_1 * poly_57 
    poly_144 = poly_1 * poly_59 
    poly_145 = jnp.take(mono,176) + jnp.take(mono,177) + jnp.take(mono,178) + jnp.take(mono,179) + jnp.take(mono,180) + jnp.take(mono,181) 
    poly_146 = poly_1 * poly_61 
    poly_147 = poly_1 * poly_62 
    poly_148 = jnp.take(mono,182) + jnp.take(mono,183) + jnp.take(mono,184) + jnp.take(mono,185) + jnp.take(mono,186) + jnp.take(mono,187) 
    poly_149 = poly_1 * poly_64 
    poly_150 = jnp.take(mono,188) + jnp.take(mono,189) + jnp.take(mono,190) + jnp.take(mono,191) + jnp.take(mono,192) + jnp.take(mono,193) 
    poly_151 = poly_1 * poly_65 
    poly_152 = poly_3 * poly_51 - poly_145 
    poly_153 = poly_3 * poly_52 - poly_148 
    poly_154 = poly_3 * poly_53 - poly_150 
    poly_155 = poly_1 * poly_67 
    poly_156 = poly_1 * poly_68 
    poly_157 = poly_1 * poly_69 
    poly_158 = jnp.take(mono,194) + jnp.take(mono,195) + jnp.take(mono,196) + jnp.take(mono,197) + jnp.take(mono,198) + jnp.take(mono,199) + jnp.take(mono,200) + jnp.take(mono,201) + jnp.take(mono,202) + jnp.take(mono,203) + jnp.take(mono,204) + jnp.take(mono,205) 
    poly_159 = poly_1 * poly_72 
    poly_160 = jnp.take(mono,206) + jnp.take(mono,207) + jnp.take(mono,208) + jnp.take(mono,209) + jnp.take(mono,210) + jnp.take(mono,211) 
    poly_161 = poly_1 * poly_73 
    poly_162 = poly_11 * poly_17 - poly_160 
    poly_163 = poly_11 * poly_18 
    poly_164 = jnp.take(mono,212) + jnp.take(mono,213) + jnp.take(mono,214) + jnp.take(mono,215) + jnp.take(mono,216) + jnp.take(mono,217) + jnp.take(mono,218) + jnp.take(mono,219) + jnp.take(mono,220) + jnp.take(mono,221) + jnp.take(mono,222) + jnp.take(mono,223) 
    poly_165 = poly_1 * poly_74 
    poly_166 = poly_31 * poly_5 
    poly_167 = poly_31 * poly_6 
    poly_168 = poly_1 * poly_75 
    poly_169 = poly_11 * poly_20 - poly_164 - poly_158 
    poly_170 = poly_31 * poly_7 
    poly_171 = poly_1 * poly_48 
    poly_172 = poly_1 * poly_49 
    poly_173 = poly_1 * poly_50 
    poly_174 = poly_1 * poly_80 
    poly_175 = poly_1 * poly_82 
    poly_176 = poly_1 * poly_83 
    poly_177 = poly_1 * poly_84 
    poly_178 = jnp.take(mono,224) + jnp.take(mono,225) + jnp.take(mono,226) + jnp.take(mono,227) + jnp.take(mono,228) + jnp.take(mono,229) + jnp.take(mono,230) + jnp.take(mono,231) + jnp.take(mono,232) + jnp.take(mono,233) + jnp.take(mono,234) + jnp.take(mono,235) 
    poly_179 = poly_5 * poly_18 
    poly_180 = poly_1 * poly_86 
    poly_181 = jnp.take(mono,236) + jnp.take(mono,237) + jnp.take(mono,238) + jnp.take(mono,239) + jnp.take(mono,240) + jnp.take(mono,241) 
    poly_182 = poly_1 * poly_87 
    poly_183 = poly_7 * poly_17 - poly_181 
    poly_184 = poly_7 * poly_18 
    poly_185 = poly_2 * poly_53 - poly_142 
    poly_186 = poly_1 * poly_54 
    poly_187 = poly_1 * poly_90 
    poly_188 = poly_1 * poly_55 
    poly_189 = poly_1 * poly_92 
    poly_190 = poly_1 * poly_56 
    poly_191 = poly_5 * poly_22 - poly_145 
    poly_192 = poly_1 * poly_93 
    poly_193 = poly_6 * poly_22 - poly_148 
    poly_194 = poly_1 * poly_58 
    poly_195 = jnp.take(mono,242) + jnp.take(mono,243) + jnp.take(mono,244) + jnp.take(mono,245) + jnp.take(mono,246) + jnp.take(mono,247) + jnp.take(mono,248) + jnp.take(mono,249) + jnp.take(mono,250) + jnp.take(mono,251) + jnp.take(mono,252) + jnp.take(mono,253) 
    poly_196 = poly_1 * poly_60 
    poly_197 = poly_9 * poly_17 - poly_148 - poly_145 - poly_195 - poly_145 
    poly_198 = poly_9 * poly_18 - poly_148 
    poly_199 = poly_1 * poly_63 
    poly_200 = poly_9 * poly_20 - poly_153 - poly_152 - poly_150 - poly_193 - poly_191 - poly_150 
    poly_201 = poly_1 * poly_95 
    poly_202 = poly_1 * poly_96 
    poly_203 = poly_2 * poly_57 - poly_150 - poly_148 - poly_145 - poly_193 - poly_191 - poly_150 - poly_148 - poly_145 
    poly_204 = poly_1 * poly_97 
    poly_205 = poly_3 * poly_80 - poly_197 
    poly_206 = poly_5 * poly_25 - poly_153 - poly_148 - poly_200 - poly_195 - poly_203 - poly_148 
    poly_207 = poly_1 * poly_98 
    poly_208 = poly_3 * poly_82 - poly_206 - poly_195 
    poly_209 = poly_3 * poly_83 - poly_198 
    poly_210 = poly_3 * poly_84 - poly_200 - poly_203 
    poly_211 = poly_1 * poly_99 
    poly_212 = poly_7 * poly_24 - poly_152 - poly_191 
    poly_213 = poly_7 * poly_25 - poly_153 - poly_193 
    poly_214 = poly_7 * poly_26 - poly_145 
    poly_215 = poly_7 * poly_27 - poly_148 
    poly_216 = poly_1 * poly_66 
    poly_217 = poly_1 * poly_101 
    poly_218 = poly_2 * poly_67 - poly_162 - poly_158 
    poly_219 = poly_2 * poly_68 - poly_163 - poly_160 - poly_158 
    poly_220 = poly_2 * poly_69 - poly_164 - poly_158 
    poly_221 = poly_1 * poly_70 
    poly_222 = poly_1 * poly_71 
    poly_223 = poly_5 * poly_30 - poly_164 - poly_160 - poly_162 - poly_158 - poly_218 - poly_160 
    poly_224 = poly_6 * poly_30 - poly_164 - poly_163 - poly_162 - poly_158 - poly_219 - poly_163 
    poly_225 = poly_1 * poly_102 
    poly_226 = poly_5 * poly_32 - poly_169 - poly_162 
    poly_227 = poly_11 * poly_36 - poly_224 - poly_219 
    poly_228 = poly_31 * poly_13 
    poly_229 = poly_2 * poly_75 - poly_169 
    poly_230 = poly_1 * poly_104 
    poly_231 = poly_1 * poly_106 
    poly_232 = poly_1 * poly_107 
    poly_233 = poly_3 * poly_57 - poly_164 - poly_158 
    poly_234 = poly_1 * poly_108 
    poly_235 = poly_9 * poly_26 - poly_164 - poly_223 
    poly_236 = poly_1 * poly_109 
    poly_237 = poly_3 * poly_61 - poly_162 
    poly_238 = poly_14 * poly_18 
    poly_239 = poly_1 * poly_110 
    poly_240 = poly_3 * poly_64 - poly_169 - poly_158 
    poly_241 = poly_3 * poly_65 - poly_169 - poly_164 
    poly_242 = poly_1 * poly_112 
    poly_243 = poly_11 * poly_22 - poly_170 
    poly_244 = poly_1 * poly_113 
    poly_245 = poly_3 * poly_67 - poly_166 
    poly_246 = poly_3 * poly_68 - poly_167 
    poly_247 = poly_3 * poly_69 - poly_170 - poly_243 - poly_170 
    poly_248 = poly_1 * poly_114 
    poly_249 = poly_31 * poly_9 
    poly_250 = poly_1 * poly_115 
    poly_251 = poly_11 * poly_24 - poly_166 - poly_245 - poly_166 
    poly_252 = poly_11 * poly_25 - poly_167 - poly_246 - poly_167 
    poly_253 = poly_11 * poly_26 - poly_166 
    poly_254 = poly_11 * poly_27 - poly_167 
    poly_255 = poly_31 * poly_10 
    poly_256 = poly_3 * poly_75 - poly_170 
    poly_257 = poly_1 * poly_76 
    poly_258 = poly_1 * poly_77 
    poly_259 = poly_1 * poly_78 
    poly_260 = poly_1 * poly_79 
    poly_261 = poly_1 * poly_118 
    poly_262 = poly_1 * poly_81 
    poly_263 = poly_5 * poly_17 - poly_142 - poly_178 - poly_142 
    poly_264 = poly_1 * poly_119 
    poly_265 = poly_6 * poly_17 - poly_142 - poly_179 - poly_178 
    poly_266 = poly_6 * poly_18 - poly_142 
    poly_267 = poly_1 * poly_85 
    poly_268 = poly_5 * poly_20 - poly_142 - poly_185 - poly_181 - poly_183 - poly_178 - poly_142 - poly_181 
    poly_269 = poly_6 * poly_20 - poly_142 - poly_185 - poly_184 - poly_179 - poly_183 - poly_142 - poly_184 
    poly_270 = poly_1 * poly_120 
    poly_271 = poly_7 * poly_20 - poly_142 - poly_185 - poly_142 
    poly_272 = poly_1 * poly_122 
    poly_273 = poly_2 * poly_80 - poly_181 - poly_178 - poly_263 
    poly_274 = poly_1 * poly_123 
    poly_275 = poly_13 * poly_17 - poly_184 - poly_183 - poly_273 
    poly_276 = poly_13 * poly_18 - poly_181 
    poly_277 = poly_6 * poly_35 - poly_181 - poly_179 - poly_268 - poly_263 - poly_275 - poly_181 
    poly_278 = poly_1 * poly_124 
    poly_279 = poly_5 * poly_37 - poly_183 - poly_271 
    poly_280 = poly_7 * poly_36 - poly_179 - poly_269 
    poly_281 = poly_1 * poly_88 
    poly_282 = poly_1 * poly_89 
    poly_283 = poly_7 * poly_22 - poly_150 
    poly_284 = poly_1 * poly_126 
    poly_285 = poly_7 * poly_39 - poly_200 
    poly_286 = poly_1 * poly_91 
    poly_287 = poly_5 * poly_39 - poly_195 - poly_193 
    poly_288 = poly_6 * poly_39 - poly_198 - poly_197 - poly_191 
    poly_289 = poly_1 * poly_94 
    poly_290 = poly_9 * poly_35 - poly_205 - poly_206 - poly_203 - poly_191 - poly_287 
    poly_291 = poly_9 * poly_36 - poly_209 - poly_208 - poly_203 - poly_193 - poly_288 
    poly_292 = poly_3 * poly_118 - poly_290 
    poly_293 = poly_3 * poly_119 - poly_291 
    poly_294 = poly_3 * poly_120 - poly_283 
    poly_295 = poly_1 * poly_127 
    poly_296 = poly_2 * poly_95 - poly_212 - poly_205 - poly_208 - poly_203 - poly_290 
    poly_297 = poly_2 * poly_96 - poly_213 - poly_209 - poly_206 - poly_203 - poly_291 
    poly_298 = poly_3 * poly_122 - poly_296 - poly_287 
    poly_299 = poly_3 * poly_123 - poly_297 - poly_288 
    poly_300 = poly_3 * poly_124 - poly_285 
    poly_301 = poly_1 * poly_100 
    poly_302 = poly_2 * poly_101 - poly_224 - poly_223 - poly_220 - poly_219 - poly_218 
    poly_303 = poly_11 * poly_46 - poly_302 
    poly_304 = poly_1 * poly_103 
    poly_305 = poly_1 * poly_129 
    poly_306 = poly_2 * poly_104 - poly_233 
    poly_307 = poly_1 * poly_105 
    poly_308 = poly_3 * poly_92 - poly_223 - poly_218 
    poly_309 = poly_3 * poly_93 - poly_224 - poly_219 
    poly_310 = poly_1 * poly_130 
    poly_311 = poly_3 * poly_95 - poly_226 - poly_218 
    poly_312 = poly_3 * poly_96 - poly_227 - poly_219 
    poly_313 = poly_2 * poly_108 - poly_241 - poly_235 
    poly_314 = poly_3 * poly_98 - poly_227 - poly_224 
    poly_315 = poly_14 * poly_37 - poly_306 
    poly_316 = poly_1 * poly_111 
    poly_317 = poly_11 * poly_39 - poly_228 
    poly_318 = poly_3 * poly_101 - poly_228 - poly_317 - poly_228 
    poly_319 = poly_3 * poly_102 - poly_228 
    poly_320 = poly_1 * poly_131 
    poly_321 = poly_11 * poly_30 - poly_255 - poly_249 - poly_249 
    poly_322 = poly_31 * poly_11 
    poly_323 = poly_2 * poly_131 - poly_321 
    poly_324 = poly_1 * poly_133 
    poly_325 = poly_3 * poly_104 - poly_243 
    poly_326 = poly_1 * poly_134 
    poly_327 = poly_3 * poly_106 - poly_251 - poly_245 
    poly_328 = poly_3 * poly_107 - poly_252 - poly_246 
    poly_329 = poly_3 * poly_108 - poly_253 
    poly_330 = poly_3 * poly_109 - poly_254 
    poly_331 = poly_7 * poly_47 - poly_325 
    poly_332 = poly_1 * poly_135 
    poly_333 = poly_11 * poly_42 - poly_249 
    poly_334 = poly_3 * poly_113 - poly_255 - poly_321 
    poly_335 = poly_31 * poly_14 
    poly_336 = poly_14 * poly_32 - poly_249 
    poly_337 = poly_1 * poly_116 
    poly_338 = poly_1 * poly_117 
    poly_339 = poly_1 * poly_121 
    poly_340 = poly_2 * poly_118 - poly_268 - poly_263 
    poly_341 = poly_2 * poly_119 - poly_269 - poly_266 - poly_265 
    poly_342 = poly_2 * poly_120 - poly_271 
    poly_343 = poly_1 * poly_137 
    poly_344 = poly_5 * poly_46 - poly_280 - poly_275 
    poly_345 = poly_6 * poly_46 - poly_279 - poly_276 - poly_273 
    poly_346 = poly_7 * poly_46 - poly_277 
    poly_347 = poly_1 * poly_125 
    poly_348 = poly_2 * poly_126 - poly_288 - poly_287 - poly_285 
    poly_349 = poly_3 * poly_137 - poly_348 
    poly_350 = poly_1 * poly_128 
    poly_351 = poly_3 * poly_126 - poly_302 
    poly_352 = poly_14 * poly_46 - poly_351 
    poly_353 = poly_1 * poly_132 
    poly_354 = poly_3 * poly_129 - poly_317 
    poly_355 = poly_13 * poly_47 - poly_354 
    poly_356 = poly_3 * poly_131 - poly_322 
    poly_357 = poly_1 * poly_138 
    poly_358 = poly_3 * poly_133 - poly_333 
    poly_359 = poly_2 * poly_138 - poly_358 
    poly_360 = poly_11 * poly_47 - poly_335 
    poly_361 = poly_1 * poly_136 
    poly_362 = poly_2 * poly_137 - poly_346 - poly_345 - poly_344 
    poly_363 = poly_3 * poly_138 - poly_360 
    poly_364 = poly_1 * poly_142 
    poly_365 = jnp.take(mono,254) 
    poly_366 = poly_1 * poly_145 
    poly_367 = poly_1 * poly_148 
    poly_368 = poly_1 * poly_150 
    poly_369 = poly_1 * poly_152 
    poly_370 = poly_1 * poly_153 
    poly_371 = jnp.take(mono,255) + jnp.take(mono,256) + jnp.take(mono,257) + jnp.take(mono,258) + jnp.take(mono,259) + jnp.take(mono,260) + jnp.take(mono,261) + jnp.take(mono,262) + jnp.take(mono,263) + jnp.take(mono,264) + jnp.take(mono,265) + jnp.take(mono,266) 
    poly_372 = poly_1 * poly_154 
    poly_373 = poly_3 * poly_142 - poly_371 
    poly_374 = poly_1 * poly_158 
    poly_375 = jnp.take(mono,267) + jnp.take(mono,268) + jnp.take(mono,269) 
    poly_376 = poly_1 * poly_160 
    poly_377 = poly_1 * poly_162 
    poly_378 = poly_1 * poly_163 
    poly_379 = poly_1 * poly_164 
    poly_380 = jnp.take(mono,270) + jnp.take(mono,271) + jnp.take(mono,272) + jnp.take(mono,273) + jnp.take(mono,274) + jnp.take(mono,275) + jnp.take(mono,276) + jnp.take(mono,277) + jnp.take(mono,278) + jnp.take(mono,279) + jnp.take(mono,280) + jnp.take(mono,281) 
    poly_381 = jnp.take(mono,282) + jnp.take(mono,283) + jnp.take(mono,284) + jnp.take(mono,285) + jnp.take(mono,286) + jnp.take(mono,287) + jnp.take(mono,288) + jnp.take(mono,289) + jnp.take(mono,290) + jnp.take(mono,291) + jnp.take(mono,292) + jnp.take(mono,293) 
    poly_382 = poly_1 * poly_166 
    poly_383 = poly_1 * poly_167 
    poly_384 = poly_31 * poly_17 
    poly_385 = poly_31 * poly_18 
    poly_386 = poly_1 * poly_169 
    poly_387 = poly_11 * poly_51 - poly_380 
    poly_388 = poly_11 * poly_52 - poly_381 
    poly_389 = poly_11 * poly_53 - poly_375 
    poly_390 = poly_1 * poly_170 
    poly_391 = poly_31 * poly_20 
    poly_392 = poly_1 * poly_139 
    poly_393 = poly_1 * poly_140 
    poly_394 = poly_1 * poly_141 
    poly_395 = poly_1 * poly_178 
    poly_396 = poly_1 * poly_179 
    poly_397 = jnp.take(mono,294) + jnp.take(mono,295) + jnp.take(mono,296) + jnp.take(mono,297) + jnp.take(mono,298) + jnp.take(mono,299) 
    poly_398 = poly_1 * poly_181 
    poly_399 = poly_1 * poly_183 
    poly_400 = poly_1 * poly_184 
    poly_401 = poly_1 * poly_185 
    poly_402 = jnp.take(mono,300) + jnp.take(mono,301) + jnp.take(mono,302) + jnp.take(mono,303) + jnp.take(mono,304) + jnp.take(mono,305) + jnp.take(mono,306) + jnp.take(mono,307) + jnp.take(mono,308) + jnp.take(mono,309) + jnp.take(mono,310) + jnp.take(mono,311) 
    poly_403 = poly_2 * poly_142 - poly_365 - poly_402 - poly_397 - poly_365 - poly_365 - poly_365 - poly_365 - poly_365 
    poly_404 = poly_1 * poly_143 
    poly_405 = poly_1 * poly_191 
    poly_406 = poly_1 * poly_193 
    poly_407 = poly_1 * poly_144 
    poly_408 = poly_1 * poly_195 
    poly_409 = jnp.take(mono,312) + jnp.take(mono,313) + jnp.take(mono,314) + jnp.take(mono,315) + jnp.take(mono,316) + jnp.take(mono,317) + jnp.take(mono,318) + jnp.take(mono,319) + jnp.take(mono,320) + jnp.take(mono,321) + jnp.take(mono,322) + jnp.take(mono,323) 
    poly_410 = poly_1 * poly_146 
    poly_411 = poly_1 * poly_197 
    poly_412 = poly_1 * poly_147 
    poly_413 = poly_17 * poly_22 - poly_409 
    poly_414 = poly_1 * poly_198 
    poly_415 = poly_18 * poly_22 
    poly_416 = poly_1 * poly_149 
    poly_417 = poly_1 * poly_200 
    poly_418 = poly_9 * poly_53 - poly_373 
    poly_419 = poly_1 * poly_151 
    poly_420 = poly_9 * poly_51 - poly_371 - poly_409 
    poly_421 = poly_9 * poly_52 - poly_371 - poly_415 - poly_413 
    poly_422 = poly_1 * poly_203 
    poly_423 = poly_1 * poly_205 
    poly_424 = poly_1 * poly_206 
    poly_425 = poly_2 * poly_145 - poly_371 - poly_409 
    poly_426 = poly_1 * poly_208 
    poly_427 = poly_1 * poly_209 
    poly_428 = poly_18 * poly_24 - poly_421 
    poly_429 = poly_1 * poly_210 
    poly_430 = poly_3 * poly_178 - poly_420 - poly_425 
    poly_431 = poly_18 * poly_26 
    poly_432 = poly_1 * poly_212 
    poly_433 = poly_1 * poly_213 
    poly_434 = poly_2 * poly_150 - poly_371 - poly_418 
    poly_435 = poly_1 * poly_214 
    poly_436 = poly_3 * poly_181 - poly_413 
    poly_437 = poly_7 * poly_59 - poly_436 - poly_409 
    poly_438 = poly_1 * poly_215 
    poly_439 = poly_7 * poly_61 - poly_413 
    poly_440 = poly_18 * poly_28 
    poly_441 = poly_2 * poly_154 - poly_373 
    poly_442 = poly_1 * poly_155 
    poly_443 = poly_1 * poly_156 
    poly_444 = poly_1 * poly_157 
    poly_445 = poly_1 * poly_218 
    poly_446 = poly_1 * poly_219 
    poly_447 = jnp.take(mono,324) + jnp.take(mono,325) + jnp.take(mono,326) + jnp.take(mono,327) + jnp.take(mono,328) + jnp.take(mono,329) + jnp.take(mono,330) + jnp.take(mono,331) + jnp.take(mono,332) + jnp.take(mono,333) + jnp.take(mono,334) + jnp.take(mono,335) 
    poly_448 = poly_1 * poly_220 
    poly_449 = poly_7 * poly_67 - poly_387 
    poly_450 = poly_7 * poly_68 - poly_388 
    poly_451 = poly_1 * poly_159 
    poly_452 = poly_1 * poly_223 
    poly_453 = poly_5 * poly_68 - poly_381 - poly_447 
    poly_454 = poly_1 * poly_161 
    poly_455 = jnp.take(mono,336) + jnp.take(mono,337) + jnp.take(mono,338) + jnp.take(mono,339) + jnp.take(mono,340) + jnp.take(mono,341) + jnp.take(mono,342) + jnp.take(mono,343) + jnp.take(mono,344) + jnp.take(mono,345) + jnp.take(mono,346) + jnp.take(mono,347) 
    poly_456 = poly_5 * poly_69 - poly_380 - poly_449 
    poly_457 = poly_1 * poly_224 
    poly_458 = poly_6 * poly_67 - poly_381 - poly_447 
    poly_459 = poly_18 * poly_30 - poly_381 
    poly_460 = poly_6 * poly_69 - poly_381 - poly_450 
    poly_461 = poly_1 * poly_165 
    poly_462 = poly_1 * poly_168 
    poly_463 = poly_20 * poly_30 - poly_389 - poly_388 - poly_387 - poly_381 - poly_380 - poly_375 - poly_460 - poly_456 - poly_450 - poly_449 - poly_447 - poly_389 - poly_388 - poly_387 - poly_375 - poly_375 - poly_375 
    poly_464 = poly_1 * poly_226 
    poly_465 = poly_11 * poly_80 - poly_455 
    poly_466 = poly_1 * poly_227 
    poly_467 = poly_11 * poly_82 - poly_453 - poly_458 
    poly_468 = poly_11 * poly_83 - poly_459 
    poly_469 = poly_11 * poly_84 - poly_463 - poly_447 
    poly_470 = poly_1 * poly_228 
    poly_471 = poly_31 * poly_35 
    poly_472 = poly_31 * poly_36 
    poly_473 = poly_1 * poly_229 
    poly_474 = poly_5 * poly_75 - poly_387 
    poly_475 = poly_6 * poly_75 - poly_388 
    poly_476 = poly_31 * poly_37 
    poly_477 = poly_1 * poly_233 
    poly_478 = poly_1 * poly_235 
    poly_479 = poly_3 * poly_145 - poly_380 
    poly_480 = poly_1 * poly_237 
    poly_481 = poly_1 * poly_238 
    poly_482 = poly_3 * poly_148 - poly_381 
    poly_483 = poly_1 * poly_240 
    poly_484 = poly_3 * poly_150 - poly_389 - poly_375 - poly_375 
    poly_485 = poly_1 * poly_241 
    poly_486 = poly_14 * poly_51 - poly_479 
    poly_487 = poly_14 * poly_52 - poly_482 
    poly_488 = poly_3 * poly_154 - poly_389 
    poly_489 = poly_1 * poly_243 
    poly_490 = poly_1 * poly_245 
    poly_491 = poly_1 * poly_246 
    poly_492 = jnp.take(mono,348) + jnp.take(mono,349) + jnp.take(mono,350) + jnp.take(mono,351) + jnp.take(mono,352) + jnp.take(mono,353) + jnp.take(mono,354) + jnp.take(mono,355) + jnp.take(mono,356) + jnp.take(mono,357) + jnp.take(mono,358) + jnp.take(mono,359) 
    poly_493 = poly_1 * poly_247 
    poly_494 = poly_3 * poly_158 - poly_391 - poly_492 
    poly_495 = poly_1 * poly_249 
    poly_496 = poly_31 * poly_22 
    poly_497 = poly_1 * poly_251 
    poly_498 = poly_1 * poly_252 
    poly_499 = poly_22 * poly_32 - poly_476 
    poly_500 = poly_1 * poly_253 
    poly_501 = poly_3 * poly_160 - poly_384 
    poly_502 = poly_11 * poly_59 - poly_384 - poly_501 - poly_384 
    poly_503 = poly_1 * poly_254 
    poly_504 = poly_11 * poly_61 - poly_384 
    poly_505 = poly_18 * poly_44 
    poly_506 = poly_3 * poly_164 - poly_391 - poly_499 
    poly_507 = poly_1 * poly_255 
    poly_508 = poly_31 * poly_24 
    poly_509 = poly_31 * poly_25 
    poly_510 = poly_31 * poly_26 
    poly_511 = poly_31 * poly_27 
    poly_512 = poly_1 * poly_256 
    poly_513 = poly_9 * poly_75 - poly_476 
    poly_514 = poly_3 * poly_169 - poly_391 - poly_513 
    poly_515 = poly_31 * poly_28 
    poly_516 = poly_1 * poly_171 
    poly_517 = poly_1 * poly_172 
    poly_518 = poly_1 * poly_173 
    poly_519 = poly_1 * poly_174 
    poly_520 = poly_1 * poly_175 
    poly_521 = poly_1 * poly_263 
    poly_522 = poly_1 * poly_176 
    poly_523 = poly_1 * poly_177 
    poly_524 = jnp.take(mono,360) + jnp.take(mono,361) + jnp.take(mono,362) + jnp.take(mono,363) + jnp.take(mono,364) + jnp.take(mono,365) 
    poly_525 = poly_1 * poly_265 
    poly_526 = poly_1 * poly_266 
    poly_527 = poly_17 * poly_18 - poly_397 
    poly_528 = poly_1 * poly_180 
    poly_529 = poly_1 * poly_268 
    poly_530 = poly_5 * poly_52 - poly_403 - poly_397 - poly_397 
    poly_531 = poly_1 * poly_182 
    poly_532 = poly_5 * poly_51 - poly_365 - poly_402 - poly_524 - poly_365 - poly_365 - poly_365 - poly_365 - poly_365 
    poly_533 = poly_5 * poly_53 - poly_402 
    poly_534 = poly_1 * poly_269 
    poly_535 = poly_6 * poly_51 - poly_403 - poly_397 - poly_397 
    poly_536 = poly_18 * poly_20 - poly_403 
    poly_537 = poly_6 * poly_53 - poly_403 
    poly_538 = poly_1 * poly_271 
    poly_539 = poly_7 * poly_51 - poly_402 
    poly_540 = poly_7 * poly_52 - poly_403 
    poly_541 = poly_7 * poly_53 - poly_365 - poly_365 - poly_365 
    poly_542 = poly_1 * poly_273 
    poly_543 = poly_1 * poly_275 
    poly_544 = poly_1 * poly_276 
    poly_545 = poly_1 * poly_277 
    poly_546 = poly_2 * poly_178 - poly_402 - poly_397 - poly_532 - poly_535 - poly_524 - poly_397 - poly_524 
    poly_547 = poly_18 * poly_35 - poly_530 
    poly_548 = poly_1 * poly_279 
    poly_549 = poly_7 * poly_80 - poly_532 
    poly_550 = poly_1 * poly_280 
    poly_551 = poly_7 * poly_82 - poly_530 - poly_535 
    poly_552 = poly_7 * poly_83 - poly_536 
    poly_553 = poly_13 * poly_53 - poly_397 
    poly_554 = poly_1 * poly_186 
    poly_555 = poly_1 * poly_187 
    poly_556 = poly_1 * poly_283 
    poly_557 = poly_1 * poly_285 
    poly_558 = poly_1 * poly_188 
    poly_559 = poly_1 * poly_189 
    poly_560 = poly_1 * poly_287 
    poly_561 = poly_1 * poly_190 
    poly_562 = jnp.take(mono,366) + jnp.take(mono,367) + jnp.take(mono,368) + jnp.take(mono,369) + jnp.take(mono,370) + jnp.take(mono,371) + jnp.take(mono,372) + jnp.take(mono,373) + jnp.take(mono,374) + jnp.take(mono,375) + jnp.take(mono,376) + jnp.take(mono,377) 
    poly_563 = poly_1 * poly_192 
    poly_564 = poly_5 * poly_90 - poly_409 - poly_562 
    poly_565 = poly_1 * poly_288 
    poly_566 = poly_6 * poly_90 - poly_415 - poly_413 - poly_564 
    poly_567 = poly_1 * poly_194 
    poly_568 = jnp.take(mono,378) + jnp.take(mono,379) + jnp.take(mono,380) + jnp.take(mono,381) + jnp.take(mono,382) + jnp.take(mono,383) + jnp.take(mono,384) + jnp.take(mono,385) + jnp.take(mono,386) + jnp.take(mono,387) + jnp.take(mono,388) + jnp.take(mono,389) 
    poly_569 = poly_1 * poly_196 
    poly_570 = poly_17 * poly_39 - poly_415 - poly_409 - poly_568 
    poly_571 = poly_18 * poly_39 - poly_413 
    poly_572 = poly_1 * poly_199 
    poly_573 = poly_20 * poly_39 - poly_421 - poly_420 - poly_418 - poly_566 - poly_562 
    poly_574 = poly_1 * poly_201 
    poly_575 = poly_1 * poly_290 
    poly_576 = poly_1 * poly_202 
    poly_577 = poly_22 * poly_35 - poly_425 - poly_562 
    poly_578 = poly_1 * poly_291 
    poly_579 = poly_22 * poly_36 - poly_428 - poly_566 
    poly_580 = poly_1 * poly_204 
    poly_581 = poly_9 * poly_80 - poly_425 - poly_413 - poly_570 
    poly_582 = poly_6 * poly_59 - poly_371 - poly_431 - poly_430 - poly_421 - poly_425 
    poly_583 = poly_1 * poly_292 
    poly_584 = poly_17 * poly_26 - poly_373 - poly_430 - poly_373 
    poly_585 = poly_1 * poly_207 
    poly_586 = poly_3 * poly_263 - poly_584 - poly_581 
    poly_587 = poly_18 * poly_25 - poly_371 
    poly_588 = poly_1 * poly_293 
    poly_589 = poly_3 * poly_265 - poly_582 
    poly_590 = poly_3 * poly_266 - poly_587 
    poly_591 = poly_1 * poly_211 
    poly_592 = poly_7 * poly_92 - poly_420 - poly_562 
    poly_593 = poly_7 * poly_93 - poly_421 - poly_566 
    poly_594 = poly_3 * poly_268 - poly_592 - poly_577 
    poly_595 = poly_3 * poly_269 - poly_593 - poly_579 
    poly_596 = poly_1 * poly_294 
    poly_597 = poly_7 * poly_64 - poly_373 - poly_418 - poly_373 
    poly_598 = poly_3 * poly_271 - poly_597 - poly_564 
    poly_599 = poly_1 * poly_296 
    poly_600 = poly_1 * poly_297 
    poly_601 = poly_2 * poly_203 - poly_434 - poly_428 - poly_425 - poly_579 - poly_577 
    poly_602 = poly_1 * poly_298 
    poly_603 = poly_3 * poly_273 - poly_570 
    poly_604 = poly_5 * poly_96 - poly_440 - poly_428 - poly_593 - poly_582 - poly_601 
    poly_605 = poly_1 * poly_299 
    poly_606 = poly_3 * poly_275 - poly_604 - poly_568 
    poly_607 = poly_3 * poly_276 - poly_571 
    poly_608 = poly_3 * poly_277 - poly_573 - poly_601 
    poly_609 = poly_1 * poly_300 
    poly_610 = poly_7 * poly_95 - poly_430 - poly_577 
    poly_611 = poly_7 * poly_96 - poly_431 - poly_579 
    poly_612 = poly_3 * poly_279 - poly_610 - poly_562 
    poly_613 = poly_3 * poly_280 - poly_611 - poly_566 
    poly_614 = poly_1 * poly_216 
    poly_615 = poly_1 * poly_217 
    poly_616 = poly_5 * poly_67 - poly_380 - poly_375 - poly_455 - poly_375 
    poly_617 = poly_6 * poly_68 - poly_380 - poly_375 - poly_459 - poly_375 
    poly_618 = poly_22 * poly_28 - poly_389 - poly_484 
    poly_619 = poly_1 * poly_302 
    poly_620 = poly_13 * poly_67 - poly_467 - poly_450 
    poly_621 = poly_13 * poly_68 - poly_468 - poly_465 - poly_449 
    poly_622 = poly_7 * poly_101 - poly_463 - poly_447 
    poly_623 = poly_1 * poly_221 
    poly_624 = poly_1 * poly_222 
    poly_625 = poly_5 * poly_101 - poly_460 - poly_453 - poly_450 - poly_458 - poly_620 
    poly_626 = poly_6 * poly_101 - poly_456 - poly_459 - poly_455 - poly_449 - poly_621 
    poly_627 = poly_1 * poly_225 
    poly_628 = poly_11 * poly_118 - poly_616 
    poly_629 = poly_11 * poly_119 - poly_617 
    poly_630 = poly_7 * poly_75 - poly_389 
    poly_631 = poly_1 * poly_303 
    poly_632 = poly_5 * poly_102 - poly_475 - poly_467 
    poly_633 = poly_11 * poly_123 - poly_626 - poly_621 
    poly_634 = poly_31 * poly_46 
    poly_635 = poly_7 * poly_102 - poly_469 
    poly_636 = poly_1 * poly_230 
    poly_637 = poly_1 * poly_306 
    poly_638 = poly_1 * poly_231 
    poly_639 = poly_1 * poly_308 
    poly_640 = poly_1 * poly_232 
    poly_641 = poly_5 * poly_104 - poly_479 
    poly_642 = poly_1 * poly_309 
    poly_643 = poly_6 * poly_104 - poly_482 
    poly_644 = poly_1 * poly_234 
    poly_645 = poly_3 * poly_195 - poly_453 - poly_458 
    poly_646 = poly_1 * poly_236 
    poly_647 = poly_3 * poly_197 - poly_455 
    poly_648 = poly_3 * poly_198 - poly_459 
    poly_649 = poly_1 * poly_239 
    poly_650 = poly_3 * poly_200 - poly_463 - poly_447 
    poly_651 = poly_1 * poly_311 
    poly_652 = poly_1 * poly_312 
    poly_653 = poly_3 * poly_203 - poly_469 - poly_447 
    poly_654 = poly_1 * poly_313 
    poly_655 = poly_14 * poly_80 - poly_647 
    poly_656 = poly_3 * poly_206 - poly_453 - poly_467 
    poly_657 = poly_1 * poly_314 
    poly_658 = poly_3 * poly_208 - poly_467 - poly_458 
    poly_659 = poly_14 * poly_83 - poly_648 
    poly_660 = poly_3 * poly_210 - poly_463 - poly_469 
    poly_661 = poly_1 * poly_315 
    poly_662 = poly_3 * poly_212 - poly_474 - poly_449 
    poly_663 = poly_3 * poly_213 - poly_475 - poly_450 
    poly_664 = poly_7 * poly_108 - poly_479 
    poly_665 = poly_7 * poly_109 - poly_482 
    poly_666 = poly_1 * poly_242 
    poly_667 = poly_1 * poly_317 
    poly_668 = poly_11 * poly_90 - poly_476 
    poly_669 = poly_1 * poly_244 
    poly_670 = poly_9 * poly_67 - poly_384 - poly_492 - poly_384 
    poly_671 = poly_9 * poly_68 - poly_385 - poly_384 - poly_492 - poly_385 - poly_385 
    poly_672 = poly_1 * poly_318 
    poly_673 = poly_3 * poly_218 - poly_471 - poly_670 
    poly_674 = poly_3 * poly_219 - poly_472 - poly_671 
    poly_675 = poly_2 * poly_247 - poly_506 - poly_494 
    poly_676 = poly_1 * poly_248 
    poly_677 = poly_31 * poly_39 
    poly_678 = poly_1 * poly_250 
    poly_679 = poly_11 * poly_92 - poly_471 - poly_670 
    poly_680 = poly_11 * poly_93 - poly_472 - poly_671 
    poly_681 = poly_3 * poly_223 - poly_471 - poly_679 
    poly_682 = poly_3 * poly_224 - poly_472 - poly_680 
    poly_683 = poly_1 * poly_319 
    poly_684 = poly_11 * poly_95 - poly_471 - poly_673 
    poly_685 = poly_11 * poly_96 - poly_472 - poly_674 
    poly_686 = poly_3 * poly_226 - poly_471 - poly_684 
    poly_687 = poly_3 * poly_227 - poly_472 - poly_685 
    poly_688 = poly_31 * poly_40 
    poly_689 = poly_3 * poly_229 - poly_476 
    poly_690 = poly_1 * poly_321 
    poly_691 = poly_11 * poly_67 - poly_508 
    poly_692 = poly_11 * poly_68 - poly_509 
    poly_693 = poly_11 * poly_69 - poly_515 - poly_496 - poly_496 
    poly_694 = poly_1 * poly_322 
    poly_695 = poly_31 * poly_30 
    poly_696 = poly_1 * poly_323 
    poly_697 = poly_5 * poly_131 - poly_691 
    poly_698 = poly_6 * poly_131 - poly_692 
    poly_699 = poly_31 * poly_32 
    poly_700 = poly_7 * poly_131 - poly_693 
    poly_701 = poly_1 * poly_325 
    poly_702 = poly_1 * poly_327 
    poly_703 = poly_1 * poly_328 
    poly_704 = poly_3 * poly_233 - poly_499 - poly_492 
    poly_705 = poly_1 * poly_329 
    poly_706 = poly_3 * poly_235 - poly_501 - poly_502 
    poly_707 = poly_1 * poly_330 
    poly_708 = poly_3 * poly_237 - poly_504 
    poly_709 = poly_18 * poly_47 
    poly_710 = poly_1 * poly_331 
    poly_711 = poly_3 * poly_240 - poly_513 - poly_494 
    poly_712 = poly_3 * poly_241 - poly_514 - poly_506 
    poly_713 = poly_1 * poly_333 
    poly_714 = poly_11 * poly_104 - poly_496 
    poly_715 = poly_1 * poly_334 
    poly_716 = poly_14 * poly_67 - poly_510 
    poly_717 = poly_14 * poly_68 - poly_511 
    poly_718 = poly_3 * poly_247 - poly_515 - poly_693 
    poly_719 = poly_1 * poly_335 
    poly_720 = poly_31 * poly_42 
    poly_721 = poly_1 * poly_336 
    poly_722 = poly_3 * poly_251 - poly_508 - poly_697 
    poly_723 = poly_3 * poly_252 - poly_509 - poly_698 
    poly_724 = poly_11 * poly_108 - poly_510 
    poly_725 = poly_11 * poly_109 - poly_511 
    poly_726 = poly_31 * poly_43 
    poly_727 = poly_14 * poly_75 - poly_496 
    poly_728 = poly_1 * poly_257 
    poly_729 = poly_1 * poly_258 
    poly_730 = poly_1 * poly_259 
    poly_731 = poly_1 * poly_260 
    poly_732 = poly_1 * poly_261 
    poly_733 = poly_1 * poly_262 
    poly_734 = poly_1 * poly_264 
    poly_735 = poly_6 * poly_80 - poly_397 - poly_530 - poly_546 
    poly_736 = poly_18 * poly_18 - poly_365 - poly_365 
    poly_737 = poly_1 * poly_267 
    poly_738 = poly_1 * poly_270 
    poly_739 = poly_7 * poly_84 - poly_397 - poly_553 - poly_397 
    poly_740 = poly_1 * poly_272 
    poly_741 = poly_1 * poly_340 
    poly_742 = poly_5 * poly_80 - poly_402 - poly_524 - poly_524 
    poly_743 = poly_1 * poly_274 
    poly_744 = poly_17 * poly_35 - poly_403 - poly_402 - poly_546 - poly_532 - poly_742 
    poly_745 = poly_6 * poly_118 - poly_530 - poly_744 
    poly_746 = poly_1 * poly_341 
    poly_747 = poly_2 * poly_265 - poly_535 - poly_527 - poly_735 
    poly_748 = poly_18 * poly_36 - poly_402 
    poly_749 = poly_5 * poly_119 - poly_536 - poly_747 
    poly_750 = poly_1 * poly_278 
    poly_751 = poly_7 * poly_118 - poly_524 
    poly_752 = poly_7 * poly_119 - poly_527 
    poly_753 = poly_1 * poly_342 
    poly_754 = poly_5 * poly_120 - poly_539 
    poly_755 = poly_6 * poly_120 - poly_540 
    poly_756 = poly_1 * poly_344 
    poly_757 = poly_2 * poly_273 - poly_549 - poly_546 - poly_742 
    poly_758 = poly_1 * poly_345 
    poly_759 = poly_17 * poly_46 - poly_552 - poly_551 - poly_757 
    poly_760 = poly_18 * poly_46 - poly_549 
    poly_761 = poly_5 * poly_123 - poly_552 - poly_547 - poly_752 - poly_759 - poly_747 - poly_552 
    poly_762 = poly_1 * poly_346 
    poly_763 = poly_5 * poly_124 - poly_551 - poly_755 
    poly_764 = poly_7 * poly_123 - poly_547 - poly_749 
    poly_765 = poly_1 * poly_281 
    poly_766 = poly_1 * poly_282 
    poly_767 = poly_1 * poly_284 
    poly_768 = poly_2 * poly_283 - poly_564 
    poly_769 = poly_1 * poly_348 
    poly_770 = poly_7 * poly_126 - poly_573 
    poly_771 = poly_1 * poly_286 
    poly_772 = poly_5 * poly_126 - poly_568 - poly_566 
    poly_773 = poly_6 * poly_126 - poly_571 - poly_570 - poly_562 
    poly_774 = poly_1 * poly_289 
    poly_775 = poly_9 * poly_118 - poly_584 - poly_577 
    poly_776 = poly_9 * poly_119 - poly_590 - poly_589 - poly_579 
    poly_777 = poly_1 * poly_295 
    poly_778 = poly_2 * poly_290 - poly_592 - poly_581 - poly_586 - poly_577 - poly_775 
    poly_779 = poly_2 * poly_291 - poly_593 - poly_587 - poly_582 - poly_579 - poly_776 
    poly_780 = poly_2 * poly_292 - poly_594 - poly_584 
    poly_781 = poly_3 * poly_341 - poly_779 - poly_776 
    poly_782 = poly_3 * poly_342 - poly_768 
    poly_783 = poly_1 * poly_349 
    poly_784 = poly_2 * poly_296 - poly_610 - poly_603 - poly_606 - poly_601 - poly_778 
    poly_785 = poly_2 * poly_297 - poly_611 - poly_607 - poly_604 - poly_601 - poly_779 
    poly_786 = poly_3 * poly_344 - poly_784 - poly_772 
    poly_787 = poly_3 * poly_345 - poly_785 - poly_773 
    poly_788 = poly_3 * poly_346 - poly_770 
    poly_789 = poly_1 * poly_301 
    poly_790 = poly_2 * poly_302 - poly_626 - poly_625 - poly_622 - poly_621 - poly_620 
    poly_791 = poly_11 * poly_137 - poly_790 
    poly_792 = poly_1 * poly_304 
    poly_793 = poly_1 * poly_305 
    poly_794 = poly_3 * poly_283 - poly_618 
    poly_795 = poly_1 * poly_351 
    poly_796 = poly_3 * poly_285 - poly_622 
    poly_797 = poly_1 * poly_307 
    poly_798 = poly_3 * poly_287 - poly_625 - poly_620 
    poly_799 = poly_3 * poly_288 - poly_626 - poly_621 
    poly_800 = poly_1 * poly_310 
    poly_801 = poly_3 * poly_290 - poly_628 - poly_616 - poly_616 
    poly_802 = poly_3 * poly_291 - poly_629 - poly_617 - poly_617 
    poly_803 = poly_3 * poly_292 - poly_628 
    poly_804 = poly_3 * poly_293 - poly_629 
    poly_805 = poly_14 * poly_120 - poly_794 
    poly_806 = poly_1 * poly_352 
    poly_807 = poly_3 * poly_296 - poly_632 - poly_620 
    poly_808 = poly_3 * poly_297 - poly_633 - poly_621 
    poly_809 = poly_3 * poly_298 - poly_632 - poly_625 
    poly_810 = poly_3 * poly_299 - poly_633 - poly_626 
    poly_811 = poly_14 * poly_124 - poly_796 
    poly_812 = poly_1 * poly_316 
    poly_813 = poly_11 * poly_126 - poly_634 
    poly_814 = poly_3 * poly_302 - poly_634 - poly_813 - poly_634 
    poly_815 = poly_3 * poly_303 - poly_634 
    poly_816 = poly_1 * poly_320 
    poly_817 = poly_11 * poly_101 - poly_688 - poly_677 - poly_677 
    poly_818 = poly_31 * poly_31 
    poly_819 = poly_11 * poly_102 - poly_688 
    poly_820 = poly_1 * poly_324 
    poly_821 = poly_1 * poly_354 
    poly_822 = poly_2 * poly_325 - poly_704 
    poly_823 = poly_1 * poly_326 
    poly_824 = poly_3 * poly_308 - poly_679 - poly_670 
    poly_825 = poly_3 * poly_309 - poly_680 - poly_671 
    poly_826 = poly_1 * poly_355 
    poly_827 = poly_3 * poly_311 - poly_684 - poly_673 
    poly_828 = poly_3 * poly_312 - poly_685 - poly_674 
    poly_829 = poly_2 * poly_329 - poly_712 - poly_706 
    poly_830 = poly_3 * poly_314 - poly_687 - poly_682 
    poly_831 = poly_37 * poly_47 - poly_822 
    poly_832 = poly_1 * poly_332 
    poly_833 = poly_11 * poly_129 - poly_677 
    poly_834 = poly_3 * poly_318 - poly_688 - poly_817 
    poly_835 = poly_14 * poly_102 - poly_677 
    poly_836 = poly_1 * poly_356 
    poly_837 = poly_9 * poly_131 - poly_699 
    poly_838 = poly_3 * poly_321 - poly_695 - poly_837 
    poly_839 = poly_31 * poly_44 
    poly_840 = poly_3 * poly_323 - poly_699 
    poly_841 = poly_1 * poly_358 
    poly_842 = poly_3 * poly_325 - poly_714 
    poly_843 = poly_1 * poly_359 
    poly_844 = poly_3 * poly_327 - poly_722 - poly_716 
    poly_845 = poly_3 * poly_328 - poly_723 - poly_717 
    poly_846 = poly_3 * poly_329 - poly_724 
    poly_847 = poly_3 * poly_330 - poly_725 
    poly_848 = poly_7 * poly_138 - poly_842 
    poly_849 = poly_1 * poly_360 
    poly_850 = poly_11 * poly_133 - poly_720 
    poly_851 = poly_3 * poly_334 - poly_726 - poly_838 
    poly_852 = poly_31 * poly_47 
    poly_853 = poly_32 * poly_47 - poly_720 
    poly_854 = poly_1 * poly_337 
    poly_855 = poly_1 * poly_338 
    poly_856 = poly_1 * poly_339 
    poly_857 = poly_5 * poly_118 - poly_533 - poly_532 - poly_742 
    poly_858 = poly_6 * poly_119 - poly_537 - poly_535 - poly_748 
    poly_859 = poly_7 * poly_120 - poly_541 
    poly_860 = poly_1 * poly_343 
    poly_861 = poly_13 * poly_118 - poly_739 - poly_735 - poly_735 
    poly_862 = poly_13 * poly_119 - poly_739 - poly_736 - poly_735 - poly_736 - poly_736 
    poly_863 = poly_7 * poly_124 - poly_553 
    poly_864 = poly_1 * poly_362 
    poly_865 = poly_5 * poly_137 - poly_764 - poly_759 
    poly_866 = poly_6 * poly_137 - poly_763 - poly_760 - poly_757 
    poly_867 = poly_7 * poly_137 - poly_761 
    poly_868 = poly_1 * poly_347 
    poly_869 = poly_2 * poly_348 - poly_773 - poly_772 - poly_770 
    poly_870 = poly_3 * poly_362 - poly_869 
    poly_871 = poly_1 * poly_350 
    poly_872 = poly_3 * poly_348 - poly_790 
    poly_873 = poly_14 * poly_137 - poly_872 
    poly_874 = poly_1 * poly_353 
    poly_875 = poly_3 * poly_351 - poly_813 
    poly_876 = poly_46 * poly_47 - poly_875 
    poly_877 = poly_11 * poly_131 - poly_839 
    poly_878 = poly_1 * poly_357 
    poly_879 = poly_3 * poly_354 - poly_833 
    poly_880 = poly_13 * poly_138 - poly_879 
    poly_881 = poly_3 * poly_356 - poly_839 - poly_877 - poly_877 
    poly_882 = poly_1 * poly_363 
    poly_883 = poly_3 * poly_358 - poly_850 
    poly_884 = poly_2 * poly_363 - poly_883 
    poly_885 = poly_11 * poly_138 - poly_852 
    poly_886 = poly_1 * poly_361 
    poly_887 = poly_2 * poly_362 - poly_867 - poly_866 - poly_865 
    poly_888 = poly_3 * poly_363 - poly_885 
    poly_889 = poly_1 * poly_365 
    poly_890 = poly_1 * poly_371 
    poly_891 = poly_1 * poly_373 
    poly_892 = poly_365 * poly_3 
    poly_893 = poly_1 * poly_375 
    poly_894 = poly_1 * poly_380 
    poly_895 = poly_1 * poly_381 
    poly_896 = jnp.take(mono,390) + jnp.take(mono,391) + jnp.take(mono,392) + jnp.take(mono,393) + jnp.take(mono,394) + jnp.take(mono,395) 
    poly_897 = poly_1 * poly_384 
    poly_898 = poly_1 * poly_385 
    poly_899 = poly_1 * poly_387 
    poly_900 = poly_1 * poly_388 
    poly_901 = poly_1 * poly_389 
    poly_902 = poly_11 * poly_142 - poly_896 
    poly_903 = poly_1 * poly_391 
    poly_904 = poly_31 * poly_51 
    poly_905 = poly_31 * poly_52 
    poly_906 = poly_31 * poly_53 
    poly_907 = poly_1 * poly_364 
    poly_908 = poly_1 * poly_397 
    poly_909 = poly_1 * poly_402 
    poly_910 = poly_1 * poly_403 
    poly_911 = poly_365 * poly_2 
    poly_912 = poly_1 * poly_366 
    poly_913 = poly_1 * poly_409 
    poly_914 = poly_1 * poly_367 
    poly_915 = poly_1 * poly_413 
    poly_916 = poly_1 * poly_415 
    poly_917 = poly_1 * poly_368 
    poly_918 = poly_1 * poly_418 
    poly_919 = poly_1 * poly_369 
    poly_920 = poly_1 * poly_420 
    poly_921 = poly_1 * poly_370 
    poly_922 = jnp.take(mono,396) + jnp.take(mono,397) + jnp.take(mono,398) + jnp.take(mono,399) + jnp.take(mono,400) + jnp.take(mono,401) + jnp.take(mono,402) + jnp.take(mono,403) + jnp.take(mono,404) + jnp.take(mono,405) + jnp.take(mono,406) + jnp.take(mono,407) 
    poly_923 = poly_1 * poly_421 
    poly_924 = jnp.take(mono,408) + jnp.take(mono,409) + jnp.take(mono,410) + jnp.take(mono,411) + jnp.take(mono,412) + jnp.take(mono,413) + jnp.take(mono,414) + jnp.take(mono,415) + jnp.take(mono,416) + jnp.take(mono,417) + jnp.take(mono,418) + jnp.take(mono,419) 
    poly_925 = poly_1 * poly_372 
    poly_926 = poly_9 * poly_142 - poly_892 - poly_924 - poly_922 - poly_892 
    poly_927 = poly_1 * poly_425 
    poly_928 = poly_1 * poly_428 
    poly_929 = poly_1 * poly_430 
    poly_930 = poly_1 * poly_431 
    poly_931 = poly_3 * poly_397 - poly_926 
    poly_932 = poly_1 * poly_434 
    poly_933 = poly_1 * poly_436 
    poly_934 = poly_1 * poly_437 
    poly_935 = jnp.take(mono,420) + jnp.take(mono,421) + jnp.take(mono,422) + jnp.take(mono,423) + jnp.take(mono,424) + jnp.take(mono,425) + jnp.take(mono,426) + jnp.take(mono,427) + jnp.take(mono,428) + jnp.take(mono,429) + jnp.take(mono,430) + jnp.take(mono,431) 
    poly_936 = poly_1 * poly_439 
    poly_937 = poly_1 * poly_440 
    poly_938 = poly_2 * poly_371 - poly_892 - poly_935 - poly_931 - poly_924 - poly_922 - poly_892 - poly_892 - poly_892 
    poly_939 = poly_1 * poly_441 
    poly_940 = poly_3 * poly_402 - poly_935 - poly_922 
    poly_941 = poly_3 * poly_403 - poly_938 - poly_924 
    poly_942 = poly_1 * poly_374 
    poly_943 = poly_1 * poly_447 
    poly_944 = poly_1 * poly_449 
    poly_945 = poly_1 * poly_450 
    poly_946 = poly_2 * poly_375 - poly_896 
    poly_947 = poly_1 * poly_376 
    poly_948 = poly_1 * poly_453 
    poly_949 = poly_1 * poly_377 
    poly_950 = poly_1 * poly_455 
    poly_951 = poly_1 * poly_378 
    poly_952 = poly_1 * poly_379 
    poly_953 = jnp.take(mono,432) + jnp.take(mono,433) + jnp.take(mono,434) + jnp.take(mono,435) + jnp.take(mono,436) + jnp.take(mono,437) + jnp.take(mono,438) + jnp.take(mono,439) + jnp.take(mono,440) + jnp.take(mono,441) + jnp.take(mono,442) + jnp.take(mono,443) 
    poly_954 = poly_1 * poly_456 
    poly_955 = jnp.take(mono,444) + jnp.take(mono,445) + jnp.take(mono,446) + jnp.take(mono,447) + jnp.take(mono,448) + jnp.take(mono,449) + jnp.take(mono,450) + jnp.take(mono,451) + jnp.take(mono,452) + jnp.take(mono,453) + jnp.take(mono,454) + jnp.take(mono,455) 
    poly_956 = jnp.take(mono,456) + jnp.take(mono,457) + jnp.take(mono,458) + jnp.take(mono,459) + jnp.take(mono,460) + jnp.take(mono,461) + jnp.take(mono,462) + jnp.take(mono,463) + jnp.take(mono,464) + jnp.take(mono,465) + jnp.take(mono,466) + jnp.take(mono,467) 
    poly_957 = poly_1 * poly_458 
    poly_958 = poly_1 * poly_459 
    poly_959 = poly_18 * poly_67 
    poly_960 = poly_1 * poly_460 
    poly_961 = poly_69 * poly_17 - poly_955 - poly_956 
    poly_962 = poly_18 * poly_69 
    poly_963 = poly_1 * poly_382 
    poly_964 = poly_1 * poly_383 
    poly_965 = poly_1 * poly_386 
    poly_966 = poly_1 * poly_463 
    poly_967 = poly_30 * poly_51 - poly_902 - poly_896 - poly_955 - poly_961 - poly_953 - poly_896 
    poly_968 = poly_30 * poly_52 - poly_902 - poly_896 - poly_962 - poly_956 - poly_959 - poly_896 
    poly_969 = poly_30 * poly_53 - poly_902 - poly_946 
    poly_970 = poly_1 * poly_390 
    poly_971 = poly_1 * poly_465 
    poly_972 = poly_1 * poly_467 
    poly_973 = poly_1 * poly_468 
    poly_974 = poly_1 * poly_469 
    poly_975 = poly_11 * poly_178 - poly_967 - poly_953 
    poly_976 = poly_18 * poly_72 - poly_968 
    poly_977 = poly_1 * poly_471 
    poly_978 = poly_31 * poly_80 
    poly_979 = poly_1 * poly_472 
    poly_980 = poly_31 * poly_82 
    poly_981 = poly_31 * poly_83 
    poly_982 = poly_31 * poly_84 
    poly_983 = poly_1 * poly_474 
    poly_984 = poly_7 * poly_160 - poly_955 
    poly_985 = poly_1 * poly_475 
    poly_986 = poly_17 * poly_75 - poly_984 
    poly_987 = poly_18 * poly_75 
    poly_988 = poly_32 * poly_53 - poly_896 
    poly_989 = poly_1 * poly_476 
    poly_990 = poly_31 * poly_86 
    poly_991 = poly_31 * poly_87 
    poly_992 = poly_1 * poly_479 
    poly_993 = poly_1 * poly_482 
    poly_994 = poly_1 * poly_484 
    poly_995 = poly_1 * poly_486 
    poly_996 = poly_1 * poly_487 
    poly_997 = poly_3 * poly_371 - poly_902 - poly_896 - poly_896 
    poly_998 = poly_1 * poly_488 
    poly_999 = poly_3 * poly_373 - poly_902 
    poly_1000 = poly_1 * poly_492 
    poly_1001 = poly_1 * poly_494 
    poly_1002 = poly_3 * poly_375 - poly_906 
    poly_1003 = poly_1 * poly_496 
    poly_1004 = poly_1 * poly_499 
    poly_1005 = poly_1 * poly_501 
    poly_1006 = poly_1 * poly_502 
    poly_1007 = poly_11 * poly_145 - poly_904 
    poly_1008 = poly_1 * poly_504 
    poly_1009 = poly_1 * poly_505 
    poly_1010 = poly_11 * poly_148 - poly_905 
    poly_1011 = poly_1 * poly_506 
    poly_1012 = poly_3 * poly_380 - poly_904 - poly_1007 - poly_904 
    poly_1013 = poly_3 * poly_381 - poly_905 - poly_1010 - poly_905 
    poly_1014 = poly_1 * poly_508 
    poly_1015 = poly_1 * poly_509 
    poly_1016 = poly_31 * poly_57 
    poly_1017 = poly_1 * poly_510 
    poly_1018 = poly_31 * poly_59 
    poly_1019 = poly_1 * poly_511 
    poly_1020 = poly_31 * poly_61 
    poly_1021 = poly_31 * poly_62 
    poly_1022 = poly_1 * poly_513 
    poly_1023 = poly_11 * poly_150 - poly_906 - poly_1002 - poly_906 
    poly_1024 = poly_1 * poly_514 
    poly_1025 = poly_3 * poly_387 - poly_904 
    poly_1026 = poly_3 * poly_388 - poly_905 
    poly_1027 = poly_11 * poly_154 - poly_906 
    poly_1028 = poly_1 * poly_515 
    poly_1029 = poly_31 * poly_64 
    poly_1030 = poly_31 * poly_65 
    poly_1031 = poly_1 * poly_392 
    poly_1032 = poly_1 * poly_393 
    poly_1033 = poly_1 * poly_394 
    poly_1034 = poly_1 * poly_395 
    poly_1035 = poly_1 * poly_524 
    poly_1036 = poly_1 * poly_396 
    poly_1037 = poly_1 * poly_527 
    poly_1038 = poly_1 * poly_398 
    poly_1039 = poly_1 * poly_530 
    poly_1040 = poly_1 * poly_399 
    poly_1041 = poly_1 * poly_532 
    poly_1042 = poly_1 * poly_400 
    poly_1043 = poly_1 * poly_401 
    poly_1044 = jnp.take(mono,468) + jnp.take(mono,469) + jnp.take(mono,470) + jnp.take(mono,471) + jnp.take(mono,472) + jnp.take(mono,473) + jnp.take(mono,474) + jnp.take(mono,475) + jnp.take(mono,476) + jnp.take(mono,477) + jnp.take(mono,478) + jnp.take(mono,479) 
    poly_1045 = poly_1 * poly_533 
    poly_1046 = poly_5 * poly_142 - poly_911 - poly_1044 - poly_911 
    poly_1047 = poly_1 * poly_535 
    poly_1048 = poly_1 * poly_536 
    poly_1049 = poly_18 * poly_51 
    poly_1050 = poly_1 * poly_537 
    poly_1051 = poly_53 * poly_17 - poly_1046 
    poly_1052 = poly_18 * poly_53 
    poly_1053 = poly_1 * poly_539 
    poly_1054 = poly_1 * poly_540 
    poly_1055 = poly_1 * poly_541 
    poly_1056 = poly_7 * poly_142 - poly_911 
    poly_1057 = poly_1 * poly_546 
    poly_1058 = poly_1 * poly_547 
    poly_1059 = jnp.take(mono,480) + jnp.take(mono,481) + jnp.take(mono,482) + jnp.take(mono,483) + jnp.take(mono,484) + jnp.take(mono,485) 
    poly_1060 = poly_1 * poly_549 
    poly_1061 = poly_1 * poly_551 
    poly_1062 = poly_1 * poly_552 
    poly_1063 = poly_1 * poly_553 
    poly_1064 = jnp.take(mono,486) + jnp.take(mono,487) + jnp.take(mono,488) + jnp.take(mono,489) + jnp.take(mono,490) + jnp.take(mono,491) + jnp.take(mono,492) + jnp.take(mono,493) + jnp.take(mono,494) + jnp.take(mono,495) + jnp.take(mono,496) + jnp.take(mono,497) 
    poly_1065 = poly_13 * poly_142 - poly_911 - poly_1064 - poly_1059 
    poly_1066 = poly_1 * poly_404 
    poly_1067 = poly_1 * poly_405 
    poly_1068 = poly_1 * poly_562 
    poly_1069 = poly_1 * poly_406 
    poly_1070 = poly_1 * poly_564 
    poly_1071 = poly_1 * poly_566 
    poly_1072 = poly_1 * poly_407 
    poly_1073 = poly_1 * poly_408 
    poly_1074 = poly_7 * poly_145 - poly_935 
    poly_1075 = poly_1 * poly_568 
    poly_1076 = jnp.take(mono,498) + jnp.take(mono,499) + jnp.take(mono,500) + jnp.take(mono,501) + jnp.take(mono,502) + jnp.take(mono,503) + jnp.take(mono,504) + jnp.take(mono,505) + jnp.take(mono,506) + jnp.take(mono,507) + jnp.take(mono,508) + jnp.take(mono,509) 
    poly_1077 = poly_1 * poly_410 
    poly_1078 = poly_1 * poly_411 
    poly_1079 = poly_1 * poly_570 
    poly_1080 = poly_1 * poly_412 
    poly_1081 = jnp.take(mono,510) + jnp.take(mono,511) + jnp.take(mono,512) + jnp.take(mono,513) + jnp.take(mono,514) + jnp.take(mono,515) 
    poly_1082 = poly_1 * poly_414 
    poly_1083 = poly_7 * poly_148 - poly_938 
    poly_1084 = poly_1 * poly_571 
    poly_1085 = poly_18 * poly_90 - poly_1083 
    poly_1086 = poly_1 * poly_416 
    poly_1087 = poly_1 * poly_417 
    poly_1088 = poly_22 * poly_53 - poly_892 
    poly_1089 = poly_1 * poly_573 
    poly_1090 = poly_39 * poly_53 - poly_926 
    poly_1091 = poly_1 * poly_419 
    poly_1092 = poly_39 * poly_51 - poly_924 - poly_1076 
    poly_1093 = poly_39 * poly_52 - poly_922 - poly_1085 - poly_1081 
    poly_1094 = poly_1 * poly_422 
    poly_1095 = poly_1 * poly_577 
    poly_1096 = poly_1 * poly_579 
    poly_1097 = poly_1 * poly_423 
    poly_1098 = poly_1 * poly_581 
    poly_1099 = poly_1 * poly_424 
    poly_1100 = poly_22 * poly_80 - poly_1081 
    poly_1101 = poly_1 * poly_582 
    poly_1102 = poly_6 * poly_145 - poly_931 - poly_924 
    poly_1103 = poly_1 * poly_584 
    poly_1104 = poly_5 * poly_145 - poly_892 - poly_922 - poly_1100 - poly_892 
    poly_1105 = poly_1 * poly_426 
    poly_1106 = poly_1 * poly_586 
    poly_1107 = poly_1 * poly_427 
    poly_1108 = poly_18 * poly_92 - poly_1093 
    poly_1109 = poly_1 * poly_587 
    poly_1110 = poly_18 * poly_57 - poly_924 
    poly_1111 = poly_1 * poly_429 
    poly_1112 = poly_3 * poly_524 - poly_1104 
    poly_1113 = poly_18 * poly_59 - poly_931 
    poly_1114 = poly_1 * poly_589 
    poly_1115 = poly_1 * poly_590 
    poly_1116 = poly_3 * poly_527 - poly_1113 
    poly_1117 = poly_1 * poly_432 
    poly_1118 = poly_1 * poly_592 
    poly_1119 = poly_1 * poly_433 
    poly_1120 = poly_5 * poly_150 - poly_935 - poly_922 
    poly_1121 = poly_1 * poly_593 
    poly_1122 = poly_6 * poly_150 - poly_938 - poly_924 
    poly_1123 = poly_1 * poly_435 
    poly_1124 = poly_9 * poly_181 - poly_935 - poly_1081 - poly_1083 
    poly_1125 = poly_7 * poly_195 - poly_1124 - poly_1076 
    poly_1126 = poly_1 * poly_594 
    poly_1127 = poly_3 * poly_530 - poly_1124 - poly_1108 
    poly_1128 = poly_26 * poly_51 - poly_892 - poly_940 - poly_1104 - poly_892 
    poly_1129 = poly_1 * poly_438 
    poly_1130 = poly_7 * poly_197 - poly_1081 
    poly_1131 = poly_7 * poly_198 - poly_1085 
    poly_1132 = poly_3 * poly_533 - poly_1120 
    poly_1133 = poly_1 * poly_595 
    poly_1134 = poly_3 * poly_535 - poly_1125 - poly_1102 
    poly_1135 = poly_18 * poly_65 - poly_941 
    poly_1136 = poly_3 * poly_537 - poly_1122 
    poly_1137 = poly_1 * poly_597 
    poly_1138 = poly_7 * poly_150 - poly_892 - poly_1088 - poly_892 
    poly_1139 = poly_1 * poly_598 
    poly_1140 = poly_3 * poly_539 - poly_1074 
    poly_1141 = poly_3 * poly_540 - poly_1083 
    poly_1142 = poly_7 * poly_154 - poly_892 
    poly_1143 = poly_1 * poly_601 
    poly_1144 = poly_1 * poly_603 
    poly_1145 = poly_1 * poly_604 
    poly_1146 = poly_13 * poly_145 - poly_938 - poly_1076 
    poly_1147 = poly_1 * poly_606 
    poly_1148 = poly_1 * poly_607 
    poly_1149 = poly_18 * poly_95 - poly_1124 
    poly_1150 = poly_1 * poly_608 
    poly_1151 = poly_3 * poly_546 - poly_1092 - poly_1146 
    poly_1152 = poly_18 * poly_97 - poly_1127 
    poly_1153 = poly_1 * poly_610 
    poly_1154 = poly_1 * poly_611 
    poly_1155 = poly_13 * poly_150 - poly_931 - poly_1090 
    poly_1156 = poly_1 * poly_612 
    poly_1157 = poly_3 * poly_549 - poly_1081 
    poly_1158 = poly_7 * poly_206 - poly_1127 - poly_1102 
    poly_1159 = poly_1 * poly_613 
    poly_1160 = poly_3 * poly_551 - poly_1158 - poly_1076 
    poly_1161 = poly_3 * poly_552 - poly_1085 
    poly_1162 = poly_13 * poly_154 - poly_926 
    poly_1163 = poly_1 * poly_442 
    poly_1164 = poly_1 * poly_443 
    poly_1165 = poly_1 * poly_444 
    poly_1166 = poly_1 * poly_445 
    poly_1167 = poly_1 * poly_616 
    poly_1168 = poly_1 * poly_446 
    poly_1169 = poly_1 * poly_617 
    poly_1170 = poly_1 * poly_448 
    poly_1171 = poly_20 * poly_67 - poly_902 - poly_967 - poly_956 - poly_961 - poly_946 
    poly_1172 = poly_20 * poly_68 - poly_902 - poly_968 - poly_962 - poly_955 - poly_946 
    poly_1173 = poly_1 * poly_618 
    poly_1174 = poly_7 * poly_158 - poly_902 - poly_946 
    poly_1175 = poly_1 * poly_620 
    poly_1176 = poly_1 * poly_621 
    poly_1177 = jnp.take(mono,516) + jnp.take(mono,517) + jnp.take(mono,518) + jnp.take(mono,519) + jnp.take(mono,520) + jnp.take(mono,521) + jnp.take(mono,522) + jnp.take(mono,523) + jnp.take(mono,524) + jnp.take(mono,525) + jnp.take(mono,526) + jnp.take(mono,527) 
    poly_1178 = poly_1 * poly_622 
    poly_1179 = poly_7 * poly_218 - poly_967 - poly_1171 
    poly_1180 = poly_7 * poly_219 - poly_968 - poly_1172 
    poly_1181 = poly_1 * poly_451 
    poly_1182 = poly_1 * poly_452 
    poly_1183 = poly_6 * poly_160 - poly_896 - poly_968 - poly_975 
    poly_1184 = poly_1 * poly_625 
    poly_1185 = poly_5 * poly_219 - poly_962 - poly_959 - poly_1183 - poly_1172 - poly_1177 - poly_1183 
    poly_1186 = poly_1 * poly_454 
    poly_1187 = jnp.take(mono,528) + jnp.take(mono,529) + jnp.take(mono,530) + jnp.take(mono,531) + jnp.take(mono,532) + jnp.take(mono,533) + jnp.take(mono,534) + jnp.take(mono,535) + jnp.take(mono,536) + jnp.take(mono,537) + jnp.take(mono,538) + jnp.take(mono,539) 
    poly_1188 = jnp.take(mono,540) + jnp.take(mono,541) + jnp.take(mono,542) + jnp.take(mono,543) + jnp.take(mono,544) + jnp.take(mono,545) + jnp.take(mono,546) + jnp.take(mono,547) + jnp.take(mono,548) + jnp.take(mono,549) + jnp.take(mono,550) + jnp.take(mono,551) 
    poly_1189 = poly_1 * poly_457 
    poly_1190 = poly_17 * poly_67 - poly_896 - poly_953 - poly_896 
    poly_1191 = poly_18 * poly_68 - poly_896 
    poly_1192 = poly_22 * poly_65 - poly_997 - poly_969 
    poly_1193 = poly_1 * poly_626 
    poly_1194 = poly_6 * poly_218 - poly_956 - poly_959 - poly_1190 - poly_1171 - poly_1177 
    poly_1195 = poly_18 * poly_101 - poly_956 
    poly_1196 = poly_6 * poly_220 - poly_962 - poly_956 - poly_1192 - poly_1180 - poly_1174 
    poly_1197 = poly_1 * poly_461 
    poly_1198 = poly_1 * poly_462 
    poly_1199 = poly_20 * poly_101 - poly_969 - poly_968 - poly_967 - poly_959 - poly_953 - poly_946 - poly_1196 - poly_1188 - poly_1180 - poly_1179 - poly_1177 
    poly_1200 = poly_1 * poly_464 
    poly_1201 = poly_1 * poly_628 
    poly_1202 = poly_5 * poly_160 - poly_902 - poly_953 
    poly_1203 = poly_1 * poly_466 
    poly_1204 = poly_11 * poly_263 - poly_1202 - poly_1190 
    poly_1205 = poly_2 * poly_456 - poly_969 - poly_955 - poly_956 - poly_1188 - poly_1192 
    poly_1206 = poly_1 * poly_629 
    poly_1207 = poly_11 * poly_265 - poly_1183 
    poly_1208 = poly_11 * poly_266 - poly_1191 
    poly_1209 = poly_2 * poly_460 - poly_969 - poly_962 - poly_961 - poly_1196 - poly_1192 
    poly_1210 = poly_1 * poly_470 
    poly_1211 = poly_31 * poly_118 
    poly_1212 = poly_31 * poly_119 
    poly_1213 = poly_1 * poly_473 
    poly_1214 = poly_7 * poly_223 - poly_953 - poly_1188 
    poly_1215 = poly_7 * poly_224 - poly_959 - poly_1196 
    poly_1216 = poly_1 * poly_630 
    poly_1217 = poly_7 * poly_169 - poly_902 - poly_969 
    poly_1218 = poly_31 * poly_120 
    poly_1219 = poly_1 * poly_632 
    poly_1220 = poly_11 * poly_273 - poly_1187 
    poly_1221 = poly_1 * poly_633 
    poly_1222 = poly_11 * poly_275 - poly_1185 - poly_1194 
    poly_1223 = poly_11 * poly_276 - poly_1195 
    poly_1224 = poly_11 * poly_277 - poly_1199 - poly_1177 
    poly_1225 = poly_1 * poly_634 
    poly_1226 = poly_31 * poly_122 
    poly_1227 = poly_31 * poly_123 
    poly_1228 = poly_1 * poly_635 
    poly_1229 = poly_5 * poly_229 - poly_986 - poly_1217 
    poly_1230 = poly_7 * poly_227 - poly_976 - poly_1209 
    poly_1231 = poly_31 * poly_124 
    poly_1232 = poly_1 * poly_477 
    poly_1233 = poly_1 * poly_641 
    poly_1234 = poly_1 * poly_643 
    poly_1235 = poly_1 * poly_478 
    poly_1236 = poly_1 * poly_645 
    poly_1237 = poly_3 * poly_409 - poly_955 - poly_961 
    poly_1238 = poly_1 * poly_480 
    poly_1239 = poly_1 * poly_647 
    poly_1240 = poly_1 * poly_481 
    poly_1241 = poly_3 * poly_413 - poly_956 
    poly_1242 = poly_1 * poly_648 
    poly_1243 = poly_18 * poly_104 
    poly_1244 = poly_1 * poly_483 
    poly_1245 = poly_1 * poly_650 
    poly_1246 = poly_42 * poly_53 - poly_999 
    poly_1247 = poly_1 * poly_485 
    poly_1248 = poly_3 * poly_420 - poly_967 - poly_953 
    poly_1249 = poly_3 * poly_421 - poly_968 - poly_959 
    poly_1250 = poly_1 * poly_653 
    poly_1251 = poly_1 * poly_655 
    poly_1252 = poly_1 * poly_656 
    poly_1253 = poly_2 * poly_479 - poly_997 - poly_1237 
    poly_1254 = poly_1 * poly_658 
    poly_1255 = poly_1 * poly_659 
    poly_1256 = poly_18 * poly_106 - poly_1249 
    poly_1257 = poly_1 * poly_660 
    poly_1258 = poly_3 * poly_430 - poly_975 - poly_967 
    poly_1259 = poly_18 * poly_108 
    poly_1260 = poly_1 * poly_662 
    poly_1261 = poly_1 * poly_663 
    poly_1262 = poly_2 * poly_484 - poly_997 - poly_1246 
    poly_1263 = poly_1 * poly_664 
    poly_1264 = poly_14 * poly_181 - poly_1241 
    poly_1265 = poly_3 * poly_437 - poly_986 - poly_955 
    poly_1266 = poly_1 * poly_665 
    poly_1267 = poly_7 * poly_237 - poly_1241 
    poly_1268 = poly_18 * poly_110 
    poly_1269 = poly_2 * poly_488 - poly_999 
    poly_1270 = poly_1 * poly_489 
    poly_1271 = poly_1 * poly_668 
    poly_1272 = poly_1 * poly_490 
    poly_1273 = poly_1 * poly_670 
    poly_1274 = poly_1 * poly_491 
    poly_1275 = poly_22 * poly_67 - poly_904 
    poly_1276 = poly_1 * poly_671 
    poly_1277 = poly_22 * poly_68 - poly_905 
    poly_1278 = poly_1 * poly_493 
    poly_1279 = poly_9 * poly_158 - poly_905 - poly_904 - poly_1002 - poly_1277 - poly_1275 - poly_905 - poly_904 - poly_1002 
    poly_1280 = poly_1 * poly_673 
    poly_1281 = poly_1 * poly_674 
    poly_1282 = poly_3 * poly_447 - poly_982 - poly_1279 
    poly_1283 = poly_1 * poly_675 
    poly_1284 = poly_3 * poly_449 - poly_990 - poly_1275 
    poly_1285 = poly_3 * poly_450 - poly_991 - poly_1277 
    poly_1286 = poly_1 * poly_495 
    poly_1287 = poly_1 * poly_677 
    poly_1288 = poly_31 * poly_90 
    poly_1289 = poly_1 * poly_497 
    poly_1290 = poly_1 * poly_679 
    poly_1291 = poly_1 * poly_498 
    poly_1292 = poly_5 * poly_243 - poly_1007 - poly_1275 
    poly_1293 = poly_1 * poly_680 
    poly_1294 = poly_6 * poly_243 - poly_1010 - poly_1277 
    poly_1295 = poly_1 * poly_500 
    poly_1296 = poly_9 * poly_160 - poly_905 - poly_1007 - poly_978 
    poly_1297 = poly_11 * poly_195 - poly_980 - poly_1296 
    poly_1298 = poly_1 * poly_681 
    poly_1299 = poly_3 * poly_453 - poly_980 - poly_1296 
    poly_1300 = poly_26 * poly_67 - poly_906 - poly_1012 - poly_1211 - poly_906 
    poly_1301 = poly_1 * poly_503 
    poly_1302 = poly_11 * poly_197 - poly_978 
    poly_1303 = poly_11 * poly_198 - poly_981 
    poly_1304 = poly_3 * poly_456 - poly_990 - poly_1292 
    poly_1305 = poly_1 * poly_682 
    poly_1306 = poly_3 * poly_458 - poly_980 - poly_1297 
    poly_1307 = poly_18 * poly_113 - poly_1013 
    poly_1308 = poly_3 * poly_460 - poly_991 - poly_1294 
    poly_1309 = poly_1 * poly_507 
    poly_1310 = poly_31 * poly_92 
    poly_1311 = poly_31 * poly_93 
    poly_1312 = poly_1 * poly_512 
    poly_1313 = poly_39 * poly_75 - poly_1231 
    poly_1314 = poly_3 * poly_463 - poly_982 - poly_1313 
    poly_1315 = poly_1 * poly_684 
    poly_1316 = poly_1 * poly_685 
    poly_1317 = poly_22 * poly_102 - poly_1231 
    poly_1318 = poly_1 * poly_686 
    poly_1319 = poly_3 * poly_465 - poly_978 
    poly_1320 = poly_11 * poly_206 - poly_980 - poly_1299 
    poly_1321 = poly_1 * poly_687 
    poly_1322 = poly_3 * poly_467 - poly_980 - poly_1320 
    poly_1323 = poly_3 * poly_468 - poly_981 
    poly_1324 = poly_3 * poly_469 - poly_982 - poly_1317 
    poly_1325 = poly_1 * poly_688 
    poly_1326 = poly_31 * poly_95 
    poly_1327 = poly_31 * poly_96 
    poly_1328 = poly_31 * poly_97 
    poly_1329 = poly_31 * poly_98 
    poly_1330 = poly_1 * poly_689 
    poly_1331 = poly_7 * poly_251 - poly_1012 - poly_1292 
    poly_1332 = poly_7 * poly_252 - poly_1013 - poly_1294 
    poly_1333 = poly_26 * poly_75 - poly_904 
    poly_1334 = poly_27 * poly_75 - poly_905 
    poly_1335 = poly_31 * poly_99 
    poly_1336 = poly_1 * poly_691 
    poly_1337 = poly_1 * poly_692 
    poly_1338 = poly_1 * poly_693 
    poly_1339 = poly_11 * poly_158 - poly_1029 - poly_1016 
    poly_1340 = poly_1 * poly_695 
    poly_1341 = poly_31 * poly_67 
    poly_1342 = poly_31 * poly_68 
    poly_1343 = poly_31 * poly_69 
    poly_1344 = poly_1 * poly_697 
    poly_1345 = poly_11 * poly_160 - poly_1018 
    poly_1346 = poly_1 * poly_698 
    poly_1347 = poly_17 * poly_131 - poly_1345 
    poly_1348 = poly_18 * poly_131 
    poly_1349 = poly_11 * poly_164 - poly_1030 - poly_1016 
    poly_1350 = poly_1 * poly_699 
    poly_1351 = poly_31 * poly_72 
    poly_1352 = poly_31 * poly_73 
    poly_1353 = poly_1 * poly_700 
    poly_1354 = poly_11 * poly_169 - poly_1030 - poly_1029 
    poly_1355 = poly_31 * poly_75 
    poly_1356 = poly_1 * poly_704 
    poly_1357 = poly_1 * poly_706 
    poly_1358 = poly_3 * poly_479 - poly_1007 
    poly_1359 = poly_1 * poly_708 
    poly_1360 = poly_1 * poly_709 
    poly_1361 = poly_3 * poly_482 - poly_1010 
    poly_1362 = poly_1 * poly_711 
    poly_1363 = poly_3 * poly_484 - poly_1023 - poly_1002 
    poly_1364 = poly_1 * poly_712 
    poly_1365 = poly_47 * poly_51 - poly_1358 
    poly_1366 = poly_47 * poly_52 - poly_1361 
    poly_1367 = poly_3 * poly_488 - poly_1027 
    poly_1368 = poly_1 * poly_714 
    poly_1369 = poly_1 * poly_716 
    poly_1370 = poly_1 * poly_717 
    poly_1371 = poly_3 * poly_492 - poly_1016 - poly_1339 
    poly_1372 = poly_1 * poly_718 
    poly_1373 = poly_3 * poly_494 - poly_1029 - poly_1339 
    poly_1374 = poly_1 * poly_720 
    poly_1375 = poly_31 * poly_104 
    poly_1376 = poly_1 * poly_722 
    poly_1377 = poly_1 * poly_723 
    poly_1378 = poly_32 * poly_104 - poly_1288 
    poly_1379 = poly_1 * poly_724 
    poly_1380 = poly_14 * poly_160 - poly_1020 
    poly_1381 = poly_3 * poly_502 - poly_1018 - poly_1347 
    poly_1382 = poly_1 * poly_725 
    poly_1383 = poly_11 * poly_237 - poly_1020 
    poly_1384 = poly_18 * poly_135 
    poly_1385 = poly_3 * poly_506 - poly_1030 - poly_1349 
    poly_1386 = poly_1 * poly_726 
    poly_1387 = poly_31 * poly_106 
    poly_1388 = poly_31 * poly_107 
    poly_1389 = poly_31 * poly_108 
    poly_1390 = poly_31 * poly_109 
    poly_1391 = poly_1 * poly_727 
    poly_1392 = poly_42 * poly_75 - poly_1288 
    poly_1393 = poly_3 * poly_514 - poly_1030 - poly_1354 
    poly_1394 = poly_31 * poly_110 
    poly_1395 = poly_1 * poly_516 
    poly_1396 = poly_1 * poly_517 
    poly_1397 = poly_1 * poly_518 
    poly_1398 = poly_1 * poly_519 
    poly_1399 = poly_1 * poly_520 
    poly_1400 = poly_1 * poly_521 
    poly_1401 = poly_1 * poly_522 
    poly_1402 = poly_1 * poly_523 
    poly_1403 = poly_1 * poly_525 
    poly_1404 = poly_1 * poly_735 
    poly_1405 = poly_1 * poly_526 
    poly_1406 = poly_1 * poly_736 
    poly_1407 = poly_1 * poly_528 
    poly_1408 = poly_1 * poly_529 
    poly_1409 = poly_18 * poly_80 - poly_1059 
    poly_1410 = poly_1 * poly_531 
    poly_1411 = poly_1 * poly_534 
    poly_1412 = poly_17 * poly_51 - poly_911 - poly_1044 - poly_911 
    poly_1413 = poly_18 * poly_52 - poly_911 
    poly_1414 = poly_1 * poly_538 
    poly_1415 = poly_1 * poly_739 
    poly_1416 = poly_7 * poly_178 - poly_1064 - poly_1044 
    poly_1417 = poly_18 * poly_86 - poly_1065 
    poly_1418 = poly_20 * poly_53 - poly_911 - poly_1056 - poly_911 
    poly_1419 = poly_1 * poly_542 
    poly_1420 = poly_1 * poly_742 
    poly_1421 = poly_1 * poly_543 
    poly_1422 = poly_1 * poly_744 
    poly_1423 = poly_1 * poly_544 
    poly_1424 = poly_1 * poly_545 
    poly_1425 = poly_2 * poly_524 - poly_1044 - poly_1412 
    poly_1426 = poly_1 * poly_745 
    poly_1427 = poly_5 * poly_178 - poly_911 - poly_1051 - poly_1044 - poly_1425 - poly_1412 - poly_911 - poly_1051 
    poly_1428 = poly_18 * poly_118 
    poly_1429 = poly_1 * poly_747 
    poly_1430 = poly_1 * poly_748 
    poly_1431 = poly_18 * poly_82 - poly_1044 
    poly_1432 = poly_1 * poly_749 
    poly_1433 = poly_6 * poly_178 - poly_1046 - poly_1049 - poly_1044 - poly_1059 - poly_1412 - poly_1059 
    poly_1434 = poly_18 * poly_84 - poly_1046 
    poly_1435 = poly_1 * poly_548 
    poly_1436 = poly_1 * poly_751 
    poly_1437 = poly_5 * poly_181 - poly_1056 - poly_1044 
    poly_1438 = poly_1 * poly_550 
    poly_1439 = poly_7 * poly_263 - poly_1437 - poly_1412 
    poly_1440 = poly_35 * poly_53 - poly_1064 - poly_1044 
    poly_1441 = poly_1 * poly_752 
    poly_1442 = poly_7 * poly_265 - poly_1409 
    poly_1443 = poly_7 * poly_266 - poly_1413 
    poly_1444 = poly_36 * poly_53 - poly_1065 - poly_1049 
    poly_1445 = poly_1 * poly_754 
    poly_1446 = poly_7 * poly_181 - poly_1046 
    poly_1447 = poly_1 * poly_755 
    poly_1448 = poly_17 * poly_120 - poly_1446 
    poly_1449 = poly_18 * poly_120 
    poly_1450 = poly_37 * poly_53 - poly_911 
    poly_1451 = poly_1 * poly_757 
    poly_1452 = poly_1 * poly_759 
    poly_1453 = poly_1 * poly_760 
    poly_1454 = poly_1 * poly_761 
    poly_1455 = poly_13 * poly_178 - poly_1056 - poly_1049 - poly_1439 - poly_1442 - poly_1425 
    poly_1456 = poly_18 * poly_122 - poly_1437 
    poly_1457 = poly_1 * poly_763 
    poly_1458 = poly_7 * poly_273 - poly_1427 
    poly_1459 = poly_1 * poly_764 
    poly_1460 = poly_7 * poly_275 - poly_1428 - poly_1433 
    poly_1461 = poly_7 * poly_276 - poly_1434 
    poly_1462 = poly_46 * poly_53 - poly_1059 
    poly_1463 = poly_1 * poly_554 
    poly_1464 = poly_1 * poly_555 
    poly_1465 = poly_1 * poly_556 
    poly_1466 = poly_1 * poly_557 
    poly_1467 = poly_1 * poly_768 
    poly_1468 = poly_1 * poly_770 
    poly_1469 = poly_1 * poly_558 
    poly_1470 = poly_1 * poly_559 
    poly_1471 = poly_1 * poly_560 
    poly_1472 = poly_1 * poly_772 
    poly_1473 = poly_1 * poly_561 
    poly_1474 = jnp.take(mono,552) + jnp.take(mono,553) + jnp.take(mono,554) + jnp.take(mono,555) + jnp.take(mono,556) + jnp.take(mono,557) + jnp.take(mono,558) + jnp.take(mono,559) + jnp.take(mono,560) + jnp.take(mono,561) + jnp.take(mono,562) + jnp.take(mono,563) 
    poly_1475 = poly_1 * poly_563 
    poly_1476 = poly_5 * poly_283 - poly_1074 
    poly_1477 = poly_1 * poly_565 
    poly_1478 = poly_6 * poly_283 - poly_1083 
    poly_1479 = poly_1 * poly_773 
    poly_1480 = poly_6 * poly_285 - poly_1085 - poly_1081 - poly_1476 
    poly_1481 = poly_1 * poly_567 
    poly_1482 = jnp.take(mono,564) + jnp.take(mono,565) + jnp.take(mono,566) + jnp.take(mono,567) + jnp.take(mono,568) + jnp.take(mono,569) + jnp.take(mono,570) + jnp.take(mono,571) + jnp.take(mono,572) + jnp.take(mono,573) + jnp.take(mono,574) + jnp.take(mono,575) 
    poly_1483 = poly_1 * poly_569 
    poly_1484 = poly_17 * poly_126 - poly_1085 - poly_1076 - poly_1482 
    poly_1485 = poly_18 * poly_126 - poly_1081 
    poly_1486 = poly_1 * poly_572 
    poly_1487 = poly_20 * poly_126 - poly_1093 - poly_1092 - poly_1090 - poly_1480 - poly_1474 
    poly_1488 = poly_1 * poly_574 
    poly_1489 = poly_1 * poly_575 
    poly_1490 = poly_1 * poly_775 
    poly_1491 = poly_1 * poly_576 
    poly_1492 = poly_22 * poly_118 - poly_1104 
    poly_1493 = poly_1 * poly_578 
    poly_1494 = poly_7 * poly_203 - poly_931 - poly_1155 
    poly_1495 = poly_1 * poly_776 
    poly_1496 = poly_22 * poly_119 - poly_1116 
    poly_1497 = poly_1 * poly_580 
    poly_1498 = poly_39 * poly_80 - poly_1102 - poly_1083 - poly_1484 
    poly_1499 = poly_6 * poly_195 - poly_922 - poly_1113 - poly_1112 - poly_1093 - poly_1100 
    poly_1500 = poly_1 * poly_583 
    poly_1501 = poly_6 * poly_205 - poly_931 - poly_1127 - poly_1151 - poly_1124 - poly_1146 
    poly_1502 = poly_1 * poly_585 
    poly_1503 = poly_5 * poly_197 - poly_922 - poly_1112 
    poly_1504 = poly_18 * poly_93 - poly_922 
    poly_1505 = poly_1 * poly_588 
    poly_1506 = poly_3 * poly_735 - poly_1501 
    poly_1507 = poly_3 * poly_736 
    poly_1508 = poly_1 * poly_591 
    poly_1509 = poly_7 * poly_287 - poly_1092 - poly_1474 
    poly_1510 = poly_7 * poly_288 - poly_1093 - poly_1480 
    poly_1511 = poly_1 * poly_596 
    poly_1512 = poly_7 * poly_200 - poly_926 - poly_1090 - poly_926 
    poly_1513 = poly_3 * poly_739 - poly_1512 - poly_1494 
    poly_1514 = poly_1 * poly_599 
    poly_1515 = poly_1 * poly_778 
    poly_1516 = poly_1 * poly_600 
    poly_1517 = poly_22 * poly_122 - poly_1146 - poly_1474 
    poly_1518 = poly_1 * poly_779 
    poly_1519 = poly_22 * poly_123 - poly_1149 - poly_1480 
    poly_1520 = poly_1 * poly_602 
    poly_1521 = poly_9 * poly_273 - poly_1146 - poly_1081 - poly_1484 
    poly_1522 = poly_5 * poly_291 - poly_1131 - poly_1110 - poly_1510 - poly_1499 - poly_1519 
    poly_1523 = poly_1 * poly_780 
    poly_1524 = poly_3 * poly_742 - poly_1521 - poly_1503 
    poly_1525 = poly_5 * poly_206 - poly_941 - poly_931 - poly_1125 - poly_1146 - poly_1501 
    poly_1526 = poly_1 * poly_605 
    poly_1527 = poly_3 * poly_744 - poly_1525 - poly_1498 
    poly_1528 = poly_18 * poly_96 - poly_935 
    poly_1529 = poly_3 * poly_745 - poly_1509 - poly_1517 
    poly_1530 = poly_1 * poly_781 
    poly_1531 = poly_3 * poly_747 - poly_1522 - poly_1499 
    poly_1532 = poly_18 * poly_98 - poly_940 
    poly_1533 = poly_3 * poly_749 - poly_1510 - poly_1519 
    poly_1534 = poly_1 * poly_609 
    poly_1535 = poly_7 * poly_290 - poly_1112 - poly_1492 
    poly_1536 = poly_7 * poly_291 - poly_1113 - poly_1496 
    poly_1537 = poly_7 * poly_292 - poly_1104 
    poly_1538 = poly_7 * poly_293 - poly_1116 
    poly_1539 = poly_1 * poly_782 
    poly_1540 = poly_7 * poly_212 - poly_940 - poly_1120 
    poly_1541 = poly_7 * poly_213 - poly_941 - poly_1122 
    poly_1542 = poly_26 * poly_120 - poly_1074 
    poly_1543 = poly_27 * poly_120 - poly_1083 
    poly_1544 = poly_1 * poly_784 
    poly_1545 = poly_1 * poly_785 
    poly_1546 = poly_2 * poly_601 - poly_1155 - poly_1149 - poly_1146 - poly_1519 - poly_1517 
    poly_1547 = poly_1 * poly_786 
    poly_1548 = poly_3 * poly_757 - poly_1484 
    poly_1549 = poly_5 * poly_297 - poly_1161 - poly_1149 - poly_1536 - poly_1522 - poly_1546 
    poly_1550 = poly_1 * poly_787 
    poly_1551 = poly_3 * poly_759 - poly_1549 - poly_1482 
    poly_1552 = poly_3 * poly_760 - poly_1485 
    poly_1553 = poly_3 * poly_761 - poly_1487 - poly_1546 
    poly_1554 = poly_1 * poly_788 
    poly_1555 = poly_7 * poly_296 - poly_1151 - poly_1517 
    poly_1556 = poly_7 * poly_297 - poly_1152 - poly_1519 
    poly_1557 = poly_3 * poly_763 - poly_1555 - poly_1474 
    poly_1558 = poly_3 * poly_764 - poly_1556 - poly_1480 
    poly_1559 = poly_1 * poly_614 
    poly_1560 = poly_1 * poly_615 
    poly_1561 = poly_1 * poly_619 
    poly_1562 = poly_2 * poly_616 - poly_1190 - poly_1171 
    poly_1563 = poly_2 * poly_617 - poly_1191 - poly_1183 - poly_1172 
    poly_1564 = poly_2 * poly_618 - poly_1192 - poly_1174 
    poly_1565 = poly_1 * poly_790 
    poly_1566 = poly_46 * poly_67 - poly_1222 - poly_1180 
    poly_1567 = poly_46 * poly_68 - poly_1223 - poly_1220 - poly_1179 
    poly_1568 = poly_7 * poly_302 - poly_1199 - poly_1177 
    poly_1569 = poly_1 * poly_623 
    poly_1570 = poly_1 * poly_624 
    poly_1571 = poly_5 * poly_302 - poly_1196 - poly_1185 - poly_1180 - poly_1194 - poly_1566 
    poly_1572 = poly_6 * poly_302 - poly_1188 - poly_1195 - poly_1187 - poly_1179 - poly_1567 
    poly_1573 = poly_1 * poly_627 
    poly_1574 = poly_2 * poly_625 - poly_1199 - poly_1188 - poly_1185 - poly_1187 - poly_1571 
    poly_1575 = poly_2 * poly_626 - poly_1199 - poly_1196 - poly_1195 - poly_1194 - poly_1572 
    poly_1576 = poly_1 * poly_631 
    poly_1577 = poly_11 * poly_340 - poly_1574 - poly_1562 
    poly_1578 = poly_11 * poly_341 - poly_1575 - poly_1563 
    poly_1579 = poly_2 * poly_630 - poly_1217 
    poly_1580 = poly_1 * poly_791 
    poly_1581 = poly_5 * poly_303 - poly_1230 - poly_1222 
    poly_1582 = poly_11 * poly_345 - poly_1572 - poly_1567 
    poly_1583 = poly_31 * poly_137 
    poly_1584 = poly_7 * poly_303 - poly_1224 
    poly_1585 = poly_1 * poly_636 
    poly_1586 = poly_1 * poly_637 
    poly_1587 = poly_1 * poly_794 
    poly_1588 = poly_1 * poly_796 
    poly_1589 = poly_1 * poly_638 
    poly_1590 = poly_1 * poly_639 
    poly_1591 = poly_1 * poly_798 
    poly_1592 = poly_1 * poly_640 
    poly_1593 = poly_3 * poly_562 - poly_1188 - poly_1179 
    poly_1594 = poly_1 * poly_642 
    poly_1595 = poly_3 * poly_564 - poly_1192 - poly_1174 
    poly_1596 = poly_1 * poly_799 
    poly_1597 = poly_3 * poly_566 - poly_1196 - poly_1180 
    poly_1598 = poly_1 * poly_644 
    poly_1599 = poly_3 * poly_568 - poly_1185 - poly_1194 
    poly_1600 = poly_1 * poly_646 
    poly_1601 = poly_3 * poly_570 - poly_1187 
    poly_1602 = poly_3 * poly_571 - poly_1195 
    poly_1603 = poly_1 * poly_649 
    poly_1604 = poly_3 * poly_573 - poly_1199 - poly_1177 
    poly_1605 = poly_1 * poly_651 
    poly_1606 = poly_1 * poly_801 
    poly_1607 = poly_1 * poly_652 
    poly_1608 = poly_3 * poly_577 - poly_1205 - poly_1171 
    poly_1609 = poly_1 * poly_802 
    poly_1610 = poly_3 * poly_579 - poly_1209 - poly_1172 
    poly_1611 = poly_1 * poly_654 
    poly_1612 = poly_3 * poly_581 - poly_1202 - poly_1190 
    poly_1613 = poly_3 * poly_582 - poly_1207 - poly_1183 - poly_1183 
    poly_1614 = poly_1 * poly_803 
    poly_1615 = poly_3 * poly_584 - poly_1202 - poly_1204 
    poly_1616 = poly_1 * poly_657 
    poly_1617 = poly_3 * poly_586 - poly_1204 - poly_1190 
    poly_1618 = poly_18 * poly_107 - poly_997 
    poly_1619 = poly_1 * poly_804 
    poly_1620 = poly_3 * poly_589 - poly_1207 
    poly_1621 = poly_3 * poly_590 - poly_1208 
    poly_1622 = poly_1 * poly_661 
    poly_1623 = poly_3 * poly_592 - poly_1214 - poly_1171 
    poly_1624 = poly_3 * poly_593 - poly_1215 - poly_1172 
    poly_1625 = poly_3 * poly_594 - poly_1214 - poly_1205 
    poly_1626 = poly_3 * poly_595 - poly_1215 - poly_1209 
    poly_1627 = poly_1 * poly_805 
    poly_1628 = poly_3 * poly_597 - poly_1217 - poly_1174 
    poly_1629 = poly_3 * poly_598 - poly_1217 - poly_1192 
    poly_1630 = poly_1 * poly_807 
    poly_1631 = poly_1 * poly_808 
    poly_1632 = poly_3 * poly_601 - poly_1224 - poly_1177 
    poly_1633 = poly_1 * poly_809 
    poly_1634 = poly_14 * poly_273 - poly_1601 
    poly_1635 = poly_3 * poly_604 - poly_1185 - poly_1222 
    poly_1636 = poly_1 * poly_810 
    poly_1637 = poly_3 * poly_606 - poly_1222 - poly_1194 
    poly_1638 = poly_14 * poly_276 - poly_1602 
    poly_1639 = poly_3 * poly_608 - poly_1199 - poly_1224 
    poly_1640 = poly_1 * poly_811 
    poly_1641 = poly_3 * poly_610 - poly_1229 - poly_1179 
    poly_1642 = poly_3 * poly_611 - poly_1230 - poly_1180 
    poly_1643 = poly_3 * poly_612 - poly_1229 - poly_1188 
    poly_1644 = poly_3 * poly_613 - poly_1230 - poly_1196 
    poly_1645 = poly_1 * poly_666 
    poly_1646 = poly_1 * poly_667 
    poly_1647 = poly_11 * poly_283 - poly_1218 
    poly_1648 = poly_1 * poly_813 
    poly_1649 = poly_11 * poly_285 - poly_1231 
    poly_1650 = poly_1 * poly_669 
    poly_1651 = poly_39 * poly_67 - poly_980 - poly_1277 
    poly_1652 = poly_39 * poly_68 - poly_981 - poly_978 - poly_1275 
    poly_1653 = poly_1 * poly_672 
    poly_1654 = poly_3 * poly_616 - poly_1211 
    poly_1655 = poly_3 * poly_617 - poly_1212 
    poly_1656 = poly_7 * poly_247 - poly_1027 - poly_1002 
    poly_1657 = poly_1 * poly_814 
    poly_1658 = poly_3 * poly_620 - poly_1226 - poly_1651 
    poly_1659 = poly_3 * poly_621 - poly_1227 - poly_1652 
    poly_1660 = poly_7 * poly_318 - poly_1314 - poly_1282 
    poly_1661 = poly_1 * poly_676 
    poly_1662 = poly_31 * poly_126 
    poly_1663 = poly_1 * poly_678 
    poly_1664 = poly_11 * poly_287 - poly_1226 - poly_1651 
    poly_1665 = poly_11 * poly_288 - poly_1227 - poly_1652 
    poly_1666 = poly_3 * poly_625 - poly_1226 - poly_1664 
    poly_1667 = poly_3 * poly_626 - poly_1227 - poly_1665 
    poly_1668 = poly_1 * poly_683 
    poly_1669 = poly_11 * poly_290 - poly_1211 - poly_1654 - poly_1211 
    poly_1670 = poly_11 * poly_291 - poly_1212 - poly_1655 - poly_1212 
    poly_1671 = poly_11 * poly_292 - poly_1211 
    poly_1672 = poly_11 * poly_293 - poly_1212 
    poly_1673 = poly_3 * poly_630 - poly_1218 
    poly_1674 = poly_1 * poly_815 
    poly_1675 = poly_11 * poly_296 - poly_1226 - poly_1658 
    poly_1676 = poly_11 * poly_297 - poly_1227 - poly_1659 
    poly_1677 = poly_3 * poly_632 - poly_1226 - poly_1675 
    poly_1678 = poly_3 * poly_633 - poly_1227 - poly_1676 
    poly_1679 = poly_31 * poly_127 
    poly_1680 = poly_3 * poly_635 - poly_1231 
    poly_1681 = poly_1 * poly_690 
    poly_1682 = poly_1 * poly_817 
    poly_1683 = poly_2 * poly_691 - poly_1347 - poly_1339 
    poly_1684 = poly_11 * poly_219 - poly_1327 - poly_1311 
    poly_1685 = poly_2 * poly_693 - poly_1349 - poly_1339 
    poly_1686 = poly_1 * poly_694 
    poly_1687 = poly_31 * poly_101 
    poly_1688 = poly_1 * poly_818 
    poly_1689 = poly_1 * poly_696 
    poly_1690 = poly_11 * poly_223 - poly_1328 - poly_1310 
    poly_1691 = poly_11 * poly_224 - poly_1329 - poly_1311 
    poly_1692 = poly_31 * poly_74 
    poly_1693 = poly_1 * poly_819 
    poly_1694 = poly_5 * poly_323 - poly_1354 - poly_1347 
    poly_1695 = poly_11 * poly_227 - poly_1329 - poly_1327 
    poly_1696 = poly_31 * poly_102 
    poly_1697 = poly_2 * poly_700 - poly_1354 
    poly_1698 = poly_1 * poly_701 
    poly_1699 = poly_1 * poly_822 
    poly_1700 = poly_1 * poly_702 
    poly_1701 = poly_1 * poly_824 
    poly_1702 = poly_1 * poly_703 
    poly_1703 = poly_5 * poly_325 - poly_1358 
    poly_1704 = poly_1 * poly_825 
    poly_1705 = poly_6 * poly_325 - poly_1361 
    poly_1706 = poly_1 * poly_705 
    poly_1707 = poly_3 * poly_645 - poly_1296 - poly_1297 
    poly_1708 = poly_1 * poly_707 
    poly_1709 = poly_3 * poly_647 - poly_1302 
    poly_1710 = poly_3 * poly_648 - poly_1303 
    poly_1711 = poly_1 * poly_710 
    poly_1712 = poly_3 * poly_650 - poly_1313 - poly_1279 
    poly_1713 = poly_1 * poly_827 
    poly_1714 = poly_1 * poly_828 
    poly_1715 = poly_3 * poly_653 - poly_1317 - poly_1282 
    poly_1716 = poly_1 * poly_829 
    poly_1717 = poly_47 * poly_80 - poly_1709 
    poly_1718 = poly_3 * poly_656 - poly_1299 - poly_1320 
    poly_1719 = poly_1 * poly_830 
    poly_1720 = poly_3 * poly_658 - poly_1322 - poly_1306 
    poly_1721 = poly_18 * poly_134 - poly_1366 
    poly_1722 = poly_3 * poly_660 - poly_1314 - poly_1324 
    poly_1723 = poly_1 * poly_831 
    poly_1724 = poly_3 * poly_662 - poly_1331 - poly_1284 
    poly_1725 = poly_3 * poly_663 - poly_1332 - poly_1285 
    poly_1726 = poly_7 * poly_329 - poly_1358 
    poly_1727 = poly_7 * poly_330 - poly_1361 
    poly_1728 = poly_1 * poly_713 
    poly_1729 = poly_1 * poly_833 
    poly_1730 = poly_11 * poly_306 - poly_1288 
    poly_1731 = poly_1 * poly_715 
    poly_1732 = poly_3 * poly_670 - poly_1310 - poly_1683 
    poly_1733 = poly_3 * poly_671 - poly_1311 - poly_1684 
    poly_1734 = poly_1 * poly_834 
    poly_1735 = poly_3 * poly_673 - poly_1326 - poly_1683 
    poly_1736 = poly_3 * poly_674 - poly_1327 - poly_1684 
    poly_1737 = poly_2 * poly_718 - poly_1385 - poly_1373 
    poly_1738 = poly_1 * poly_719 
    poly_1739 = poly_31 * poly_129 
    poly_1740 = poly_1 * poly_721 
    poly_1741 = poly_3 * poly_679 - poly_1310 - poly_1690 
    poly_1742 = poly_3 * poly_680 - poly_1311 - poly_1691 
    poly_1743 = poly_3 * poly_681 - poly_1328 - poly_1690 
    poly_1744 = poly_3 * poly_682 - poly_1329 - poly_1691 
    poly_1745 = poly_1 * poly_835 
    poly_1746 = poly_3 * poly_684 - poly_1326 - poly_1694 
    poly_1747 = poly_3 * poly_685 - poly_1327 - poly_1695 
    poly_1748 = poly_3 * poly_686 - poly_1328 - poly_1694 
    poly_1749 = poly_3 * poly_687 - poly_1329 - poly_1695 
    poly_1750 = poly_31 * poly_130 
    poly_1751 = poly_14 * poly_229 - poly_1288 
    poly_1752 = poly_1 * poly_837 
    poly_1753 = poly_22 * poly_131 - poly_1355 
    poly_1754 = poly_1 * poly_838 
    poly_1755 = poly_3 * poly_691 - poly_1341 
    poly_1756 = poly_3 * poly_692 - poly_1342 
    poly_1757 = poly_3 * poly_693 - poly_1343 - poly_1753 
    poly_1758 = poly_1 * poly_839 
    poly_1759 = poly_31 * poly_112 
    poly_1760 = poly_31 * poly_113 
    poly_1761 = poly_1 * poly_840 
    poly_1762 = poly_11 * poly_251 - poly_1387 - poly_1351 
    poly_1763 = poly_11 * poly_252 - poly_1388 - poly_1352 
    poly_1764 = poly_26 * poly_131 - poly_1341 
    poly_1765 = poly_27 * poly_131 - poly_1342 
    poly_1766 = poly_31 * poly_115 
    poly_1767 = poly_3 * poly_700 - poly_1355 
    poly_1768 = poly_1 * poly_842 
    poly_1769 = poly_1 * poly_844 
    poly_1770 = poly_1 * poly_845 
    poly_1771 = poly_3 * poly_704 - poly_1378 - poly_1371 
    poly_1772 = poly_1 * poly_846 
    poly_1773 = poly_3 * poly_706 - poly_1380 - poly_1381 
    poly_1774 = poly_1 * poly_847 
    poly_1775 = poly_3 * poly_708 - poly_1383 
    poly_1776 = poly_18 * poly_138 
    poly_1777 = poly_1 * poly_848 
    poly_1778 = poly_3 * poly_711 - poly_1392 - poly_1373 
    poly_1779 = poly_3 * poly_712 - poly_1393 - poly_1385 
    poly_1780 = poly_1 * poly_850 
    poly_1781 = poly_11 * poly_325 - poly_1375 
    poly_1782 = poly_1 * poly_851 
    poly_1783 = poly_47 * poly_67 - poly_1389 
    poly_1784 = poly_47 * poly_68 - poly_1390 
    poly_1785 = poly_3 * poly_718 - poly_1394 - poly_1757 
    poly_1786 = poly_1 * poly_852 
    poly_1787 = poly_31 * poly_133 
    poly_1788 = poly_1 * poly_853 
    poly_1789 = poly_3 * poly_722 - poly_1387 - poly_1762 
    poly_1790 = poly_3 * poly_723 - poly_1388 - poly_1763 
    poly_1791 = poly_11 * poly_329 - poly_1389 
    poly_1792 = poly_11 * poly_330 - poly_1390 
    poly_1793 = poly_31 * poly_134 
    poly_1794 = poly_47 * poly_75 - poly_1375 
    poly_1795 = poly_1 * poly_728 
    poly_1796 = poly_1 * poly_729 
    poly_1797 = poly_1 * poly_730 
    poly_1798 = poly_1 * poly_731 
    poly_1799 = poly_1 * poly_732 
    poly_1800 = poly_1 * poly_733 
    poly_1801 = poly_1 * poly_734 
    poly_1802 = poly_1 * poly_737 
    poly_1803 = poly_1 * poly_738 
    poly_1804 = poly_1 * poly_740 
    poly_1805 = poly_1 * poly_741 
    poly_1806 = poly_6 * poly_273 - poly_1059 - poly_1437 - poly_1455 
    poly_1807 = poly_1 * poly_857 
    poly_1808 = poly_1 * poly_743 
    poly_1809 = poly_17 * poly_118 - poly_1046 - poly_1427 
    poly_1810 = poly_1 * poly_746 
    poly_1811 = poly_5 * poly_265 - poly_1049 - poly_1433 
    poly_1812 = poly_2 * poly_736 - poly_1413 
    poly_1813 = poly_7 * poly_277 - poly_1059 - poly_1462 - poly_1059 
    poly_1814 = poly_1 * poly_858 
    poly_1815 = poly_6 * poly_265 - poly_1051 - poly_1431 - poly_1412 
    poly_1816 = poly_18 * poly_119 - poly_1051 
    poly_1817 = poly_1 * poly_750 
    poly_1818 = poly_5 * poly_268 - poly_1046 - poly_1418 - poly_1437 - poly_1427 - poly_1416 
    poly_1819 = poly_20 * poly_119 - poly_1049 - poly_1444 - poly_1443 - poly_1442 - poly_1431 
    poly_1820 = poly_1 * poly_753 
    poly_1821 = poly_7 * poly_268 - poly_1044 - poly_1440 
    poly_1822 = poly_7 * poly_269 - poly_1049 - poly_1444 
    poly_1823 = poly_1 * poly_859 
    poly_1824 = poly_7 * poly_271 - poly_1056 - poly_1418 
    poly_1825 = poly_1 * poly_756 
    poly_1826 = poly_1 * poly_861 
    poly_1827 = poly_5 * poly_273 - poly_1064 - poly_1425 
    poly_1828 = poly_1 * poly_758 
    poly_1829 = poly_17 * poly_122 - poly_1065 - poly_1064 - poly_1455 - poly_1439 - poly_1827 
    poly_1830 = poly_5 * poly_277 - poly_1065 - poly_1444 - poly_1428 - poly_1455 - poly_1433 
    poly_1831 = poly_1 * poly_862 
    poly_1832 = poly_13 * poly_265 - poly_1416 - poly_1413 - poly_1806 
    poly_1833 = poly_18 * poly_123 - poly_1064 
    poly_1834 = poly_6 * poly_277 - poly_1064 - poly_1440 - poly_1434 - poly_1456 - poly_1427 
    poly_1835 = poly_1 * poly_762 
    poly_1836 = poly_7 * poly_340 - poly_1425 - poly_1818 
    poly_1837 = poly_7 * poly_341 - poly_1431 - poly_1819 
    poly_1838 = poly_1 * poly_863 
    poly_1839 = poly_5 * poly_342 - poly_1448 - poly_1824 
    poly_1840 = poly_7 * poly_280 - poly_1065 - poly_1444 
    poly_1841 = poly_1 * poly_865 
    poly_1842 = poly_2 * poly_757 - poly_1458 - poly_1455 - poly_1827 
    poly_1843 = poly_1 * poly_866 
    poly_1844 = poly_17 * poly_137 - poly_1461 - poly_1460 - poly_1842 
    poly_1845 = poly_18 * poly_137 - poly_1458 
    poly_1846 = poly_13 * poly_277 - poly_1450 - poly_1431 - poly_1425 - poly_1819 - poly_1818 
    poly_1847 = poly_1 * poly_867 
    poly_1848 = poly_5 * poly_346 - poly_1460 - poly_1840 
    poly_1849 = poly_7 * poly_345 - poly_1456 - poly_1834 
    poly_1850 = poly_1 * poly_765 
    poly_1851 = poly_1 * poly_766 
    poly_1852 = poly_1 * poly_767 
    poly_1853 = poly_7 * poly_283 - poly_1088 
    poly_1854 = poly_1 * poly_769 
    poly_1855 = poly_7 * poly_285 - poly_1090 
    poly_1856 = poly_1 * poly_869 
    poly_1857 = poly_7 * poly_348 - poly_1487 
    poly_1858 = poly_1 * poly_771 
    poly_1859 = poly_5 * poly_348 - poly_1482 - poly_1480 
    poly_1860 = poly_6 * poly_348 - poly_1485 - poly_1484 - poly_1474 
    poly_1861 = poly_1 * poly_774 
    poly_1862 = poly_39 * poly_118 - poly_1501 - poly_1494 
    poly_1863 = poly_39 * poly_119 - poly_1507 - poly_1506 - poly_1494 
    poly_1864 = poly_1 * poly_777 
    poly_1865 = poly_2 * poly_775 - poly_1509 - poly_1498 - poly_1503 - poly_1492 - poly_1862 
    poly_1866 = poly_2 * poly_776 - poly_1510 - poly_1504 - poly_1499 - poly_1496 - poly_1863 
    poly_1867 = poly_3 * poly_857 - poly_1865 
    poly_1868 = poly_3 * poly_858 - poly_1866 
    poly_1869 = poly_3 * poly_859 - poly_1853 
    poly_1870 = poly_1 * poly_783 
    poly_1871 = poly_2 * poly_778 - poly_1535 - poly_1521 - poly_1527 - poly_1517 - poly_1865 
    poly_1872 = poly_2 * poly_779 - poly_1536 - poly_1528 - poly_1522 - poly_1519 - poly_1866 
    poly_1873 = poly_3 * poly_861 - poly_1871 - poly_1862 
    poly_1874 = poly_3 * poly_862 - poly_1872 - poly_1863 
    poly_1875 = poly_3 * poly_863 - poly_1855 
    poly_1876 = poly_1 * poly_870 
    poly_1877 = poly_2 * poly_784 - poly_1555 - poly_1548 - poly_1551 - poly_1546 - poly_1871 
    poly_1878 = poly_2 * poly_785 - poly_1556 - poly_1552 - poly_1549 - poly_1546 - poly_1872 
    poly_1879 = poly_3 * poly_865 - poly_1877 - poly_1859 
    poly_1880 = poly_3 * poly_866 - poly_1878 - poly_1860 
    poly_1881 = poly_3 * poly_867 - poly_1857 
    poly_1882 = poly_1 * poly_789 
    poly_1883 = poly_2 * poly_790 - poly_1572 - poly_1571 - poly_1568 - poly_1567 - poly_1566 
    poly_1884 = poly_11 * poly_362 - poly_1883 
    poly_1885 = poly_1 * poly_792 
    poly_1886 = poly_1 * poly_793 
    poly_1887 = poly_1 * poly_795 
    poly_1888 = poly_2 * poly_794 - poly_1595 
    poly_1889 = poly_1 * poly_872 
    poly_1890 = poly_3 * poly_770 - poly_1568 
    poly_1891 = poly_1 * poly_797 
    poly_1892 = poly_3 * poly_772 - poly_1571 - poly_1566 
    poly_1893 = poly_3 * poly_773 - poly_1572 - poly_1567 
    poly_1894 = poly_1 * poly_800 
    poly_1895 = poly_3 * poly_775 - poly_1574 - poly_1562 
    poly_1896 = poly_3 * poly_776 - poly_1575 - poly_1563 
    poly_1897 = poly_1 * poly_806 
    poly_1898 = poly_3 * poly_778 - poly_1577 - poly_1562 
    poly_1899 = poly_3 * poly_779 - poly_1578 - poly_1563 
    poly_1900 = poly_2 * poly_803 - poly_1625 - poly_1615 
    poly_1901 = poly_3 * poly_781 - poly_1578 - poly_1575 
    poly_1902 = poly_14 * poly_342 - poly_1888 
    poly_1903 = poly_1 * poly_873 
    poly_1904 = poly_3 * poly_784 - poly_1581 - poly_1566 
    poly_1905 = poly_3 * poly_785 - poly_1582 - poly_1567 
    poly_1906 = poly_3 * poly_786 - poly_1581 - poly_1571 
    poly_1907 = poly_3 * poly_787 - poly_1582 - poly_1572 
    poly_1908 = poly_14 * poly_346 - poly_1890 
    poly_1909 = poly_1 * poly_812 
    poly_1910 = poly_11 * poly_348 - poly_1583 
    poly_1911 = poly_3 * poly_790 - poly_1583 - poly_1910 - poly_1583 
    poly_1912 = poly_3 * poly_791 - poly_1583 
    poly_1913 = poly_1 * poly_816 
    poly_1914 = poly_11 * poly_302 - poly_1679 - poly_1662 - poly_1662 
    poly_1915 = poly_11 * poly_303 - poly_1679 
    poly_1916 = poly_1 * poly_820 
    poly_1917 = poly_1 * poly_821 
    poly_1918 = poly_3 * poly_794 - poly_1647 
    poly_1919 = poly_1 * poly_875 
    poly_1920 = poly_3 * poly_796 - poly_1649 
    poly_1921 = poly_1 * poly_823 
    poly_1922 = poly_3 * poly_798 - poly_1664 - poly_1651 
    poly_1923 = poly_3 * poly_799 - poly_1665 - poly_1652 
    poly_1924 = poly_1 * poly_826 
    poly_1925 = poly_3 * poly_801 - poly_1669 - poly_1654 
    poly_1926 = poly_3 * poly_802 - poly_1670 - poly_1655 
    poly_1927 = poly_3 * poly_803 - poly_1671 
    poly_1928 = poly_3 * poly_804 - poly_1672 
    poly_1929 = poly_47 * poly_120 - poly_1918 
    poly_1930 = poly_1 * poly_876 
    poly_1931 = poly_3 * poly_807 - poly_1675 - poly_1658 
    poly_1932 = poly_3 * poly_808 - poly_1676 - poly_1659 
    poly_1933 = poly_3 * poly_809 - poly_1677 - poly_1666 
    poly_1934 = poly_3 * poly_810 - poly_1678 - poly_1667 
    poly_1935 = poly_47 * poly_124 - poly_1920 
    poly_1936 = poly_1 * poly_832 
    poly_1937 = poly_11 * poly_351 - poly_1662 
    poly_1938 = poly_3 * poly_814 - poly_1679 - poly_1914 
    poly_1939 = poly_14 * poly_303 - poly_1662 
    poly_1940 = poly_1 * poly_836 
    poly_1941 = poly_39 * poly_131 - poly_1696 
    poly_1942 = poly_3 * poly_817 - poly_1687 - poly_1941 
    poly_1943 = poly_31 * poly_114 
    poly_1944 = poly_3 * poly_819 - poly_1696 
    poly_1945 = poly_1 * poly_877 
    poly_1946 = poly_11 * poly_321 - poly_1760 - poly_1759 
    poly_1947 = poly_31 * poly_131 
    poly_1948 = poly_2 * poly_877 - poly_1946 
    poly_1949 = poly_1 * poly_841 
    poly_1950 = poly_1 * poly_879 
    poly_1951 = poly_2 * poly_842 - poly_1771 
    poly_1952 = poly_1 * poly_843 
    poly_1953 = poly_3 * poly_824 - poly_1741 - poly_1732 
    poly_1954 = poly_3 * poly_825 - poly_1742 - poly_1733 
    poly_1955 = poly_1 * poly_880 
    poly_1956 = poly_3 * poly_827 - poly_1746 - poly_1735 
    poly_1957 = poly_3 * poly_828 - poly_1747 - poly_1736 
    poly_1958 = poly_2 * poly_846 - poly_1779 - poly_1773 
    poly_1959 = poly_3 * poly_830 - poly_1749 - poly_1744 
    poly_1960 = poly_37 * poly_138 - poly_1951 
    poly_1961 = poly_1 * poly_849 
    poly_1962 = poly_11 * poly_354 - poly_1739 
    poly_1963 = poly_3 * poly_834 - poly_1750 - poly_1942 
    poly_1964 = poly_47 * poly_102 - poly_1739 
    poly_1965 = poly_1 * poly_881 
    poly_1966 = poly_42 * poly_131 - poly_1692 
    poly_1967 = poly_3 * poly_838 - poly_1760 - poly_1946 
    poly_1968 = poly_31 * poly_135 
    poly_1969 = poly_14 * poly_323 - poly_1692 
    poly_1970 = poly_1 * poly_883 
    poly_1971 = poly_3 * poly_842 - poly_1781 
    poly_1972 = poly_1 * poly_884 
    poly_1973 = poly_3 * poly_844 - poly_1789 - poly_1783 
    poly_1974 = poly_3 * poly_845 - poly_1790 - poly_1784 
    poly_1975 = poly_3 * poly_846 - poly_1791 
    poly_1976 = poly_3 * poly_847 - poly_1792 
    poly_1977 = poly_7 * poly_363 - poly_1971 
    poly_1978 = poly_1 * poly_885 
    poly_1979 = poly_11 * poly_358 - poly_1787 
    poly_1980 = poly_3 * poly_851 - poly_1793 - poly_1967 
    poly_1981 = poly_31 * poly_138 
    poly_1982 = poly_32 * poly_138 - poly_1787 
    poly_1983 = poly_1 * poly_854 
    poly_1984 = poly_1 * poly_855 
    poly_1985 = poly_1 * poly_856 
    poly_1986 = poly_1 * poly_860 
    poly_1987 = poly_2 * poly_857 - poly_1818 - poly_1809 
    poly_1988 = poly_2 * poly_858 - poly_1819 - poly_1816 - poly_1815 
    poly_1989 = poly_2 * poly_859 - poly_1824 
    poly_1990 = poly_1 * poly_864 
    poly_1991 = poly_46 * poly_118 - poly_1822 - poly_1811 
    poly_1992 = poly_46 * poly_119 - poly_1821 - poly_1812 - poly_1806 
    poly_1993 = poly_7 * poly_346 - poly_1462 
    poly_1994 = poly_1 * poly_887 
    poly_1995 = poly_5 * poly_362 - poly_1849 - poly_1844 
    poly_1996 = poly_6 * poly_362 - poly_1848 - poly_1845 - poly_1842 
    poly_1997 = poly_7 * poly_362 - poly_1846 
    poly_1998 = poly_1 * poly_868 
    poly_1999 = poly_2 * poly_869 - poly_1860 - poly_1859 - poly_1857 
    poly_2000 = poly_3 * poly_887 - poly_1999 
    poly_2001 = poly_1 * poly_871 
    poly_2002 = poly_3 * poly_869 - poly_1883 
    poly_2003 = poly_14 * poly_362 - poly_2002 
    poly_2004 = poly_1 * poly_874 
    poly_2005 = poly_3 * poly_872 - poly_1910 
    poly_2006 = poly_47 * poly_137 - poly_2005 
    poly_2007 = poly_1 * poly_878 
    poly_2008 = poly_3 * poly_875 - poly_1937 
    poly_2009 = poly_46 * poly_138 - poly_2008 
    poly_2010 = poly_3 * poly_877 - poly_1947 
    poly_2011 = poly_1 * poly_882 
    poly_2012 = poly_3 * poly_879 - poly_1962 
    poly_2013 = poly_13 * poly_363 - poly_2012 
    poly_2014 = poly_47 * poly_131 - poly_1943 
    poly_2015 = poly_1 * poly_888 
    poly_2016 = poly_3 * poly_883 - poly_1979 
    poly_2017 = poly_2 * poly_888 - poly_2016 
    poly_2018 = poly_11 * poly_363 - poly_1981 
    poly_2019 = poly_1 * poly_886 
    poly_2020 = poly_2 * poly_887 - poly_1997 - poly_1996 - poly_1995 
    poly_2021 = poly_3 * poly_888 - poly_2018 

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
    poly_2021,    ]) 

    return poly 



