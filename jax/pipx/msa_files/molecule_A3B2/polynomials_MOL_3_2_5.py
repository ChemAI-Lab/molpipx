import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3B2.monomials_MOL_3_2_5 import f_monomials as f_monos 

# File created from ./MOL_3_2_5.POLY 

N_POLYS = 364

# Total number of monomials = 364 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(364) 

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
    poly_361,    poly_362,    poly_363,    ]) 

    return poly 



