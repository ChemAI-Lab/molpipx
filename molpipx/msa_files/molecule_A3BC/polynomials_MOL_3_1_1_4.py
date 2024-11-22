import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3BC.monomials_MOL_3_1_1_4 import f_monomials as f_monos 

# File created from ./MOL_3_1_1_4.POLY 

N_POLYS = 231

# Total number of monomials = 231 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(231) 

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
    ]) 

    return poly 



