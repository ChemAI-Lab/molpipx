import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A3B2.monomials_MOL_3_2_6 import f_monomials as f_monos

# File created from ./MOL_3_2_6.POLY

N_POLYS = 889

# Total number of monomials = 889


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(889)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = jnp.take(mono, 2) + jnp.take(mono, 3) + jnp.take(mono, 4) + \
        jnp.take(mono, 5) + jnp.take(mono, 6) + jnp.take(mono, 7)
    poly_3 = jnp.take(mono, 8) + jnp.take(mono, 9) + jnp.take(mono, 10)
    poly_4 = poly_1 * poly_2
    poly_5 = jnp.take(mono, 11) + jnp.take(mono, 12) + jnp.take(mono, 13) + \
        jnp.take(mono, 14) + jnp.take(mono, 15) + jnp.take(mono, 16)
    poly_6 = jnp.take(mono, 17) + jnp.take(mono, 18) + jnp.take(mono, 19) + \
        jnp.take(mono, 20) + jnp.take(mono, 21) + jnp.take(mono, 22)
    poly_7 = jnp.take(mono, 23) + jnp.take(mono, 24) + jnp.take(mono, 25)
    poly_8 = poly_1 * poly_3
    poly_9 = jnp.take(mono, 26) + jnp.take(mono, 27) + jnp.take(mono, 28) + \
        jnp.take(mono, 29) + jnp.take(mono, 30) + jnp.take(mono, 31)
    poly_10 = poly_2 * poly_3 - poly_9
    poly_11 = jnp.take(mono, 32) + jnp.take(mono, 33) + jnp.take(mono, 34)
    poly_12 = poly_1 * poly_1
    poly_13 = poly_2 * poly_2 - poly_7 - poly_6 - poly_5 - poly_7 - poly_6 - poly_5
    poly_14 = poly_3 * poly_3 - poly_11 - poly_11
    poly_15 = poly_1 * poly_5
    poly_16 = poly_1 * poly_6
    poly_17 = jnp.take(mono, 35) + jnp.take(mono, 36) + jnp.take(mono, 37) + \
        jnp.take(mono, 38) + jnp.take(mono, 39) + jnp.take(mono, 40)
    poly_18 = jnp.take(mono, 41) + jnp.take(mono, 42)
    poly_19 = poly_1 * poly_7
    poly_20 = jnp.take(mono, 43) + jnp.take(mono, 44) + jnp.take(mono, 45) + jnp.take(mono, 46) + jnp.take(mono, 47) + jnp.take(mono, 48) + \
        jnp.take(mono, 49) + jnp.take(mono, 50) + jnp.take(mono, 51) + \
        jnp.take(mono, 52) + jnp.take(mono, 53) + jnp.take(mono, 54)
    poly_21 = poly_1 * poly_9
    poly_22 = jnp.take(mono, 55) + jnp.take(mono, 56) + jnp.take(mono, 57)
    poly_23 = poly_1 * poly_10
    poly_24 = jnp.take(mono, 58) + jnp.take(mono, 59) + jnp.take(mono, 60) + jnp.take(mono, 61) + jnp.take(mono, 62) + jnp.take(mono, 63) + \
        jnp.take(mono, 64) + jnp.take(mono, 65) + jnp.take(mono, 66) + \
        jnp.take(mono, 67) + jnp.take(mono, 68) + jnp.take(mono, 69)
    poly_25 = jnp.take(mono, 70) + jnp.take(mono, 71) + jnp.take(mono, 72) + jnp.take(mono, 73) + jnp.take(mono, 74) + jnp.take(mono, 75) + \
        jnp.take(mono, 76) + jnp.take(mono, 77) + jnp.take(mono, 78) + \
        jnp.take(mono, 79) + jnp.take(mono, 80) + jnp.take(mono, 81)
    poly_26 = poly_3 * poly_5 - poly_24
    poly_27 = poly_3 * poly_6 - poly_25
    poly_28 = poly_3 * poly_7 - poly_22
    poly_29 = poly_1 * poly_11
    poly_30 = jnp.take(mono, 82) + jnp.take(mono, 83) + jnp.take(mono, 84) + jnp.take(mono, 85) + jnp.take(mono, 86) + jnp.take(mono, 87) + \
        jnp.take(mono, 88) + jnp.take(mono, 89) + jnp.take(mono, 90) + \
        jnp.take(mono, 91) + jnp.take(mono, 92) + jnp.take(mono, 93)
    poly_31 = jnp.take(mono, 94)
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
    poly_51 = jnp.take(mono, 95) + jnp.take(mono, 96) + jnp.take(mono, 97) + \
        jnp.take(mono, 98) + jnp.take(mono, 99) + jnp.take(mono, 100)
    poly_52 = jnp.take(mono, 101) + jnp.take(mono, 102) + jnp.take(mono, 103) + \
        jnp.take(mono, 104) + jnp.take(mono, 105) + jnp.take(mono, 106)
    poly_53 = jnp.take(mono, 107) + jnp.take(mono, 108) + jnp.take(mono, 109)
    poly_54 = poly_1 * poly_22
    poly_55 = poly_1 * poly_24
    poly_56 = poly_1 * poly_25
    poly_57 = jnp.take(mono, 110) + jnp.take(mono, 111) + jnp.take(mono, 112) + jnp.take(mono, 113) + jnp.take(mono, 114) + jnp.take(mono, 115) + \
        jnp.take(mono, 116) + jnp.take(mono, 117) + jnp.take(mono, 118) + \
        jnp.take(mono, 119) + jnp.take(mono, 120) + jnp.take(mono, 121)
    poly_58 = poly_1 * poly_26
    poly_59 = jnp.take(mono, 122) + jnp.take(mono, 123) + jnp.take(mono, 124) + jnp.take(mono, 125) + jnp.take(mono, 126) + jnp.take(mono, 127) + \
        jnp.take(mono, 128) + jnp.take(mono, 129) + jnp.take(mono, 130) + \
        jnp.take(mono, 131) + jnp.take(mono, 132) + jnp.take(mono, 133)
    poly_60 = poly_1 * poly_27
    poly_61 = poly_3 * poly_17 - poly_59
    poly_62 = poly_3 * poly_18
    poly_63 = poly_1 * poly_28
    poly_64 = jnp.take(mono, 134) + jnp.take(mono, 135) + jnp.take(mono, 136) + jnp.take(mono, 137) + jnp.take(mono, 138) + jnp.take(mono, 139) + \
        jnp.take(mono, 140) + jnp.take(mono, 141) + jnp.take(mono, 142) + \
        jnp.take(mono, 143) + jnp.take(mono, 144) + jnp.take(mono, 145)
    poly_65 = poly_3 * poly_20 - poly_64 - poly_57
    poly_66 = poly_1 * poly_30
    poly_67 = jnp.take(mono, 146) + jnp.take(mono, 147) + jnp.take(mono, 148) + \
        jnp.take(mono, 149) + jnp.take(mono, 150) + jnp.take(mono, 151)
    poly_68 = jnp.take(mono, 152) + jnp.take(mono, 153) + jnp.take(mono, 154) + \
        jnp.take(mono, 155) + jnp.take(mono, 156) + jnp.take(mono, 157)
    poly_69 = jnp.take(mono, 158) + jnp.take(mono, 159) + jnp.take(mono, 160) + \
        jnp.take(mono, 161) + jnp.take(mono, 162) + jnp.take(mono, 163)
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
    poly_80 = jnp.take(mono, 164) + jnp.take(mono, 165) + jnp.take(mono, 166) + \
        jnp.take(mono, 167) + jnp.take(mono, 168) + jnp.take(mono, 169)
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
    poly_95 = poly_2 * poly_24 - poly_64 - poly_59 - \
        poly_61 - poly_57 - poly_92 - poly_61
    poly_96 = poly_2 * poly_25 - poly_64 - poly_62 - \
        poly_59 - poly_57 - poly_93 - poly_62
    poly_97 = poly_2 * poly_26 - poly_65 - poly_59
    poly_98 = poly_3 * poly_36 - poly_96 - poly_93
    poly_99 = poly_3 * poly_37 - poly_90
    poly_100 = poly_1 * poly_29
    poly_101 = poly_2 * poly_30 - poly_73 - poly_72 - \
        poly_69 - poly_68 - poly_67 - poly_69 - poly_68 - poly_67
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
    poly_118 = poly_5 * poly_5 - poly_53 - poly_51 - \
        poly_80 - poly_53 - poly_51 - poly_80
    poly_119 = poly_6 * poly_6 - poly_53 - poly_51 - \
        poly_83 - poly_53 - poly_51 - poly_83
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
    poly_142 = jnp.take(mono, 170) + jnp.take(mono, 171) + jnp.take(mono, 172) + \
        jnp.take(mono, 173) + jnp.take(mono, 174) + jnp.take(mono, 175)
    poly_143 = poly_1 * poly_57
    poly_144 = poly_1 * poly_59
    poly_145 = jnp.take(mono, 176) + jnp.take(mono, 177) + jnp.take(mono, 178) + \
        jnp.take(mono, 179) + jnp.take(mono, 180) + jnp.take(mono, 181)
    poly_146 = poly_1 * poly_61
    poly_147 = poly_1 * poly_62
    poly_148 = jnp.take(mono, 182) + jnp.take(mono, 183) + jnp.take(mono, 184) + \
        jnp.take(mono, 185) + jnp.take(mono, 186) + jnp.take(mono, 187)
    poly_149 = poly_1 * poly_64
    poly_150 = jnp.take(mono, 188) + jnp.take(mono, 189) + jnp.take(mono, 190) + \
        jnp.take(mono, 191) + jnp.take(mono, 192) + jnp.take(mono, 193)
    poly_151 = poly_1 * poly_65
    poly_152 = poly_3 * poly_51 - poly_145
    poly_153 = poly_3 * poly_52 - poly_148
    poly_154 = poly_3 * poly_53 - poly_150
    poly_155 = poly_1 * poly_67
    poly_156 = poly_1 * poly_68
    poly_157 = poly_1 * poly_69
    poly_158 = jnp.take(mono, 194) + jnp.take(mono, 195) + jnp.take(mono, 196) + jnp.take(mono, 197) + jnp.take(mono, 198) + jnp.take(mono, 199) + \
        jnp.take(mono, 200) + jnp.take(mono, 201) + jnp.take(mono, 202) + \
        jnp.take(mono, 203) + jnp.take(mono, 204) + jnp.take(mono, 205)
    poly_159 = poly_1 * poly_72
    poly_160 = jnp.take(mono, 206) + jnp.take(mono, 207) + jnp.take(mono, 208) + \
        jnp.take(mono, 209) + jnp.take(mono, 210) + jnp.take(mono, 211)
    poly_161 = poly_1 * poly_73
    poly_162 = poly_11 * poly_17 - poly_160
    poly_163 = poly_11 * poly_18
    poly_164 = jnp.take(mono, 212) + jnp.take(mono, 213) + jnp.take(mono, 214) + jnp.take(mono, 215) + jnp.take(mono, 216) + jnp.take(mono, 217) + \
        jnp.take(mono, 218) + jnp.take(mono, 219) + jnp.take(mono, 220) + \
        jnp.take(mono, 221) + jnp.take(mono, 222) + jnp.take(mono, 223)
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
    poly_178 = jnp.take(mono, 224) + jnp.take(mono, 225) + jnp.take(mono, 226) + jnp.take(mono, 227) + jnp.take(mono, 228) + jnp.take(mono, 229) + \
        jnp.take(mono, 230) + jnp.take(mono, 231) + jnp.take(mono, 232) + \
        jnp.take(mono, 233) + jnp.take(mono, 234) + jnp.take(mono, 235)
    poly_179 = poly_5 * poly_18
    poly_180 = poly_1 * poly_86
    poly_181 = jnp.take(mono, 236) + jnp.take(mono, 237) + jnp.take(mono, 238) + \
        jnp.take(mono, 239) + jnp.take(mono, 240) + jnp.take(mono, 241)
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
    poly_195 = jnp.take(mono, 242) + jnp.take(mono, 243) + jnp.take(mono, 244) + jnp.take(mono, 245) + jnp.take(mono, 246) + jnp.take(mono, 247) + \
        jnp.take(mono, 248) + jnp.take(mono, 249) + jnp.take(mono, 250) + \
        jnp.take(mono, 251) + jnp.take(mono, 252) + jnp.take(mono, 253)
    poly_196 = poly_1 * poly_60
    poly_197 = poly_9 * poly_17 - poly_148 - poly_145 - poly_195 - poly_145
    poly_198 = poly_9 * poly_18 - poly_148
    poly_199 = poly_1 * poly_63
    poly_200 = poly_9 * poly_20 - poly_153 - poly_152 - \
        poly_150 - poly_193 - poly_191 - poly_150
    poly_201 = poly_1 * poly_95
    poly_202 = poly_1 * poly_96
    poly_203 = poly_2 * poly_57 - poly_150 - poly_148 - poly_145 - \
        poly_193 - poly_191 - poly_150 - poly_148 - poly_145
    poly_204 = poly_1 * poly_97
    poly_205 = poly_3 * poly_80 - poly_197
    poly_206 = poly_5 * poly_25 - poly_153 - poly_148 - \
        poly_200 - poly_195 - poly_203 - poly_148
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
    poly_223 = poly_5 * poly_30 - poly_164 - poly_160 - \
        poly_162 - poly_158 - poly_218 - poly_160
    poly_224 = poly_6 * poly_30 - poly_164 - poly_163 - \
        poly_162 - poly_158 - poly_219 - poly_163
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
    poly_268 = poly_5 * poly_20 - poly_142 - poly_185 - \
        poly_181 - poly_183 - poly_178 - poly_142 - poly_181
    poly_269 = poly_6 * poly_20 - poly_142 - poly_185 - \
        poly_184 - poly_179 - poly_183 - poly_142 - poly_184
    poly_270 = poly_1 * poly_120
    poly_271 = poly_7 * poly_20 - poly_142 - poly_185 - poly_142
    poly_272 = poly_1 * poly_122
    poly_273 = poly_2 * poly_80 - poly_181 - poly_178 - poly_263
    poly_274 = poly_1 * poly_123
    poly_275 = poly_13 * poly_17 - poly_184 - poly_183 - poly_273
    poly_276 = poly_13 * poly_18 - poly_181
    poly_277 = poly_6 * poly_35 - poly_181 - poly_179 - \
        poly_268 - poly_263 - poly_275 - poly_181
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
    poly_290 = poly_9 * poly_35 - poly_205 - \
        poly_206 - poly_203 - poly_191 - poly_287
    poly_291 = poly_9 * poly_36 - poly_209 - \
        poly_208 - poly_203 - poly_193 - poly_288
    poly_292 = poly_3 * poly_118 - poly_290
    poly_293 = poly_3 * poly_119 - poly_291
    poly_294 = poly_3 * poly_120 - poly_283
    poly_295 = poly_1 * poly_127
    poly_296 = poly_2 * poly_95 - poly_212 - \
        poly_205 - poly_208 - poly_203 - poly_290
    poly_297 = poly_2 * poly_96 - poly_213 - \
        poly_209 - poly_206 - poly_203 - poly_291
    poly_298 = poly_3 * poly_122 - poly_296 - poly_287
    poly_299 = poly_3 * poly_123 - poly_297 - poly_288
    poly_300 = poly_3 * poly_124 - poly_285
    poly_301 = poly_1 * poly_100
    poly_302 = poly_2 * poly_101 - poly_224 - \
        poly_223 - poly_220 - poly_219 - poly_218
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
    poly_365 = jnp.take(mono, 254)
    poly_366 = poly_1 * poly_145
    poly_367 = poly_1 * poly_148
    poly_368 = poly_1 * poly_150
    poly_369 = poly_1 * poly_152
    poly_370 = poly_1 * poly_153
    poly_371 = jnp.take(mono, 255) + jnp.take(mono, 256) + jnp.take(mono, 257) + jnp.take(mono, 258) + jnp.take(mono, 259) + jnp.take(mono, 260) + \
        jnp.take(mono, 261) + jnp.take(mono, 262) + jnp.take(mono, 263) + \
        jnp.take(mono, 264) + jnp.take(mono, 265) + jnp.take(mono, 266)
    poly_372 = poly_1 * poly_154
    poly_373 = poly_3 * poly_142 - poly_371
    poly_374 = poly_1 * poly_158
    poly_375 = jnp.take(mono, 267) + jnp.take(mono, 268) + jnp.take(mono, 269)
    poly_376 = poly_1 * poly_160
    poly_377 = poly_1 * poly_162
    poly_378 = poly_1 * poly_163
    poly_379 = poly_1 * poly_164
    poly_380 = jnp.take(mono, 270) + jnp.take(mono, 271) + jnp.take(mono, 272) + jnp.take(mono, 273) + jnp.take(mono, 274) + jnp.take(mono, 275) + \
        jnp.take(mono, 276) + jnp.take(mono, 277) + jnp.take(mono, 278) + \
        jnp.take(mono, 279) + jnp.take(mono, 280) + jnp.take(mono, 281)
    poly_381 = jnp.take(mono, 282) + jnp.take(mono, 283) + jnp.take(mono, 284) + jnp.take(mono, 285) + jnp.take(mono, 286) + jnp.take(mono, 287) + \
        jnp.take(mono, 288) + jnp.take(mono, 289) + jnp.take(mono, 290) + \
        jnp.take(mono, 291) + jnp.take(mono, 292) + jnp.take(mono, 293)
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
    poly_397 = jnp.take(mono, 294) + jnp.take(mono, 295) + jnp.take(mono, 296) + \
        jnp.take(mono, 297) + jnp.take(mono, 298) + jnp.take(mono, 299)
    poly_398 = poly_1 * poly_181
    poly_399 = poly_1 * poly_183
    poly_400 = poly_1 * poly_184
    poly_401 = poly_1 * poly_185
    poly_402 = jnp.take(mono, 300) + jnp.take(mono, 301) + jnp.take(mono, 302) + jnp.take(mono, 303) + jnp.take(mono, 304) + jnp.take(mono, 305) + \
        jnp.take(mono, 306) + jnp.take(mono, 307) + jnp.take(mono, 308) + \
        jnp.take(mono, 309) + jnp.take(mono, 310) + jnp.take(mono, 311)
    poly_403 = poly_2 * poly_142 - poly_365 - poly_402 - poly_397 - \
        poly_365 - poly_365 - poly_365 - poly_365 - poly_365
    poly_404 = poly_1 * poly_143
    poly_405 = poly_1 * poly_191
    poly_406 = poly_1 * poly_193
    poly_407 = poly_1 * poly_144
    poly_408 = poly_1 * poly_195
    poly_409 = jnp.take(mono, 312) + jnp.take(mono, 313) + jnp.take(mono, 314) + jnp.take(mono, 315) + jnp.take(mono, 316) + jnp.take(mono, 317) + \
        jnp.take(mono, 318) + jnp.take(mono, 319) + jnp.take(mono, 320) + \
        jnp.take(mono, 321) + jnp.take(mono, 322) + jnp.take(mono, 323)
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
    poly_447 = jnp.take(mono, 324) + jnp.take(mono, 325) + jnp.take(mono, 326) + jnp.take(mono, 327) + jnp.take(mono, 328) + jnp.take(mono, 329) + \
        jnp.take(mono, 330) + jnp.take(mono, 331) + jnp.take(mono, 332) + \
        jnp.take(mono, 333) + jnp.take(mono, 334) + jnp.take(mono, 335)
    poly_448 = poly_1 * poly_220
    poly_449 = poly_7 * poly_67 - poly_387
    poly_450 = poly_7 * poly_68 - poly_388
    poly_451 = poly_1 * poly_159
    poly_452 = poly_1 * poly_223
    poly_453 = poly_5 * poly_68 - poly_381 - poly_447
    poly_454 = poly_1 * poly_161
    poly_455 = jnp.take(mono, 336) + jnp.take(mono, 337) + jnp.take(mono, 338) + jnp.take(mono, 339) + jnp.take(mono, 340) + jnp.take(mono, 341) + \
        jnp.take(mono, 342) + jnp.take(mono, 343) + jnp.take(mono, 344) + \
        jnp.take(mono, 345) + jnp.take(mono, 346) + jnp.take(mono, 347)
    poly_456 = poly_5 * poly_69 - poly_380 - poly_449
    poly_457 = poly_1 * poly_224
    poly_458 = poly_6 * poly_67 - poly_381 - poly_447
    poly_459 = poly_18 * poly_30 - poly_381
    poly_460 = poly_6 * poly_69 - poly_381 - poly_450
    poly_461 = poly_1 * poly_165
    poly_462 = poly_1 * poly_168
    poly_463 = poly_20 * poly_30 - poly_389 - poly_388 - poly_387 - poly_381 - poly_380 - poly_375 - poly_460 - \
        poly_456 - poly_450 - poly_449 - poly_447 - poly_389 - \
        poly_388 - poly_387 - poly_375 - poly_375 - poly_375
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
    poly_492 = jnp.take(mono, 348) + jnp.take(mono, 349) + jnp.take(mono, 350) + jnp.take(mono, 351) + jnp.take(mono, 352) + jnp.take(mono, 353) + \
        jnp.take(mono, 354) + jnp.take(mono, 355) + jnp.take(mono, 356) + \
        jnp.take(mono, 357) + jnp.take(mono, 358) + jnp.take(mono, 359)
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
    poly_524 = jnp.take(mono, 360) + jnp.take(mono, 361) + jnp.take(mono, 362) + \
        jnp.take(mono, 363) + jnp.take(mono, 364) + jnp.take(mono, 365)
    poly_525 = poly_1 * poly_265
    poly_526 = poly_1 * poly_266
    poly_527 = poly_17 * poly_18 - poly_397
    poly_528 = poly_1 * poly_180
    poly_529 = poly_1 * poly_268
    poly_530 = poly_5 * poly_52 - poly_403 - poly_397 - poly_397
    poly_531 = poly_1 * poly_182
    poly_532 = poly_5 * poly_51 - poly_365 - poly_402 - poly_524 - \
        poly_365 - poly_365 - poly_365 - poly_365 - poly_365
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
    poly_546 = poly_2 * poly_178 - poly_402 - poly_397 - \
        poly_532 - poly_535 - poly_524 - poly_397 - poly_524
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
    poly_562 = jnp.take(mono, 366) + jnp.take(mono, 367) + jnp.take(mono, 368) + jnp.take(mono, 369) + jnp.take(mono, 370) + jnp.take(mono, 371) + \
        jnp.take(mono, 372) + jnp.take(mono, 373) + jnp.take(mono, 374) + \
        jnp.take(mono, 375) + jnp.take(mono, 376) + jnp.take(mono, 377)
    poly_563 = poly_1 * poly_192
    poly_564 = poly_5 * poly_90 - poly_409 - poly_562
    poly_565 = poly_1 * poly_288
    poly_566 = poly_6 * poly_90 - poly_415 - poly_413 - poly_564
    poly_567 = poly_1 * poly_194
    poly_568 = jnp.take(mono, 378) + jnp.take(mono, 379) + jnp.take(mono, 380) + jnp.take(mono, 381) + jnp.take(mono, 382) + jnp.take(mono, 383) + \
        jnp.take(mono, 384) + jnp.take(mono, 385) + jnp.take(mono, 386) + \
        jnp.take(mono, 387) + jnp.take(mono, 388) + jnp.take(mono, 389)
    poly_569 = poly_1 * poly_196
    poly_570 = poly_17 * poly_39 - poly_415 - poly_409 - poly_568
    poly_571 = poly_18 * poly_39 - poly_413
    poly_572 = poly_1 * poly_199
    poly_573 = poly_20 * poly_39 - poly_421 - \
        poly_420 - poly_418 - poly_566 - poly_562
    poly_574 = poly_1 * poly_201
    poly_575 = poly_1 * poly_290
    poly_576 = poly_1 * poly_202
    poly_577 = poly_22 * poly_35 - poly_425 - poly_562
    poly_578 = poly_1 * poly_291
    poly_579 = poly_22 * poly_36 - poly_428 - poly_566
    poly_580 = poly_1 * poly_204
    poly_581 = poly_9 * poly_80 - poly_425 - poly_413 - poly_570
    poly_582 = poly_6 * poly_59 - poly_371 - \
        poly_431 - poly_430 - poly_421 - poly_425
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
    poly_601 = poly_2 * poly_203 - poly_434 - \
        poly_428 - poly_425 - poly_579 - poly_577
    poly_602 = poly_1 * poly_298
    poly_603 = poly_3 * poly_273 - poly_570
    poly_604 = poly_5 * poly_96 - poly_440 - \
        poly_428 - poly_593 - poly_582 - poly_601
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
    poly_625 = poly_5 * poly_101 - poly_460 - \
        poly_453 - poly_450 - poly_458 - poly_620
    poly_626 = poly_6 * poly_101 - poly_456 - \
        poly_459 - poly_455 - poly_449 - poly_621
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
    poly_671 = poly_9 * poly_68 - poly_385 - \
        poly_384 - poly_492 - poly_385 - poly_385
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
    poly_744 = poly_17 * poly_35 - poly_403 - \
        poly_402 - poly_546 - poly_532 - poly_742
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
    poly_761 = poly_5 * poly_123 - poly_552 - poly_547 - \
        poly_752 - poly_759 - poly_747 - poly_552
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
    poly_778 = poly_2 * poly_290 - poly_592 - \
        poly_581 - poly_586 - poly_577 - poly_775
    poly_779 = poly_2 * poly_291 - poly_593 - \
        poly_587 - poly_582 - poly_579 - poly_776
    poly_780 = poly_2 * poly_292 - poly_594 - poly_584
    poly_781 = poly_3 * poly_341 - poly_779 - poly_776
    poly_782 = poly_3 * poly_342 - poly_768
    poly_783 = poly_1 * poly_349
    poly_784 = poly_2 * poly_296 - poly_610 - \
        poly_603 - poly_606 - poly_601 - poly_778
    poly_785 = poly_2 * poly_297 - poly_611 - \
        poly_607 - poly_604 - poly_601 - poly_779
    poly_786 = poly_3 * poly_344 - poly_784 - poly_772
    poly_787 = poly_3 * poly_345 - poly_785 - poly_773
    poly_788 = poly_3 * poly_346 - poly_770
    poly_789 = poly_1 * poly_301
    poly_790 = poly_2 * poly_302 - poly_626 - \
        poly_625 - poly_622 - poly_621 - poly_620
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
    poly_862 = poly_13 * poly_119 - poly_739 - \
        poly_736 - poly_735 - poly_736 - poly_736
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
                      poly_881,    poly_882,    poly_883,    poly_884,    poly_885,
                      poly_886,    poly_887,    poly_888,])

    return poly
