import jax.numpy as jnp
from jax import jit

from Data.DDEthanol.PIP_deg_2.monomials_ddethanol_p2 import f_monomials as f_monos

# File created from ./MOL_1_1_1_2_3_1_2.POLY

N_POLYS = 208

# Total number of monomials = 208

@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(208)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2) + jnp.take(mono, 3)
    poly_2 = jnp.take(mono, 4) + jnp.take(mono, 5) + jnp.take(mono, 6)
    poly_3 = jnp.take(mono, 7) + jnp.take(mono, 8)
    poly_4 = jnp.take(mono, 9) + jnp.take(mono, 10) + jnp.take(mono, 11) + \
        jnp.take(mono, 12) + jnp.take(mono, 13) + jnp.take(mono, 14)
    poly_5 = jnp.take(mono, 15)
    poly_6 = jnp.take(mono, 16)
    poly_7 = jnp.take(mono, 17) + jnp.take(mono, 18) + jnp.take(mono, 19)
    poly_8 = jnp.take(mono, 20) + jnp.take(mono, 21)
    poly_9 = jnp.take(mono, 22)
    poly_10 = jnp.take(mono, 23) + jnp.take(mono, 24) + jnp.take(mono, 25)
    poly_11 = jnp.take(mono, 26) + jnp.take(mono, 27)
    poly_12 = jnp.take(mono, 28)
    poly_13 = jnp.take(mono, 29)
    poly_14 = jnp.take(mono, 30) + jnp.take(mono, 31) + jnp.take(mono, 32)
    poly_15 = jnp.take(mono, 33) + jnp.take(mono, 34)
    poly_16 = jnp.take(mono, 35)
    poly_17 = jnp.take(mono, 36)
    poly_18 = jnp.take(mono, 37) + jnp.take(mono, 38) + jnp.take(mono, 39)
    poly_19 = jnp.take(mono, 40) + jnp.take(mono, 41) + jnp.take(mono, 42)
    poly_20 = poly_1 * poly_2 - poly_19
    poly_21 = jnp.take(mono, 43) + jnp.take(mono, 44) + jnp.take(mono, 45)
    poly_22 = poly_1 * poly_3
    poly_23 = poly_2 * poly_3
    poly_24 = jnp.take(mono, 46)
    poly_25 = jnp.take(mono, 47) + jnp.take(mono, 48) + jnp.take(mono, 49) + jnp.take(mono, 50) + jnp.take(mono, 51) + jnp.take(mono, 52) + \
        jnp.take(mono, 53) + jnp.take(mono, 54) + jnp.take(mono, 55) + \
        jnp.take(mono, 56) + jnp.take(mono, 57) + jnp.take(mono, 58)
    poly_26 = jnp.take(mono, 59) + jnp.take(mono, 60) + jnp.take(mono, 61) + \
        jnp.take(mono, 62) + jnp.take(mono, 63) + jnp.take(mono, 64)
    poly_27 = poly_1 * poly_4 - poly_25
    poly_28 = poly_2 * poly_4 - poly_26
    poly_29 = jnp.take(mono, 65) + jnp.take(mono, 66) + jnp.take(mono, 67) + \
        jnp.take(mono, 68) + jnp.take(mono, 69) + jnp.take(mono, 70)
    poly_30 = jnp.take(mono, 71) + jnp.take(mono, 72) + jnp.take(mono, 73) + \
        jnp.take(mono, 74) + jnp.take(mono, 75) + jnp.take(mono, 76)
    poly_31 = jnp.take(mono, 77) + jnp.take(mono, 78) + jnp.take(mono, 79)
    poly_32 = poly_3 * poly_4 - poly_29
    poly_33 = jnp.take(mono, 80) + jnp.take(mono, 81) + jnp.take(mono, 82) + \
        jnp.take(mono, 83) + jnp.take(mono, 84) + jnp.take(mono, 85)
    poly_34 = poly_5 * poly_1
    poly_35 = poly_5 * poly_2
    poly_36 = poly_5 * poly_3
    poly_37 = poly_5 * poly_4
    poly_38 = poly_6 * poly_1
    poly_39 = poly_6 * poly_2
    poly_40 = poly_6 * poly_3
    poly_41 = poly_6 * poly_4
    poly_42 = poly_5 * poly_6
    poly_43 = jnp.take(mono, 86) + jnp.take(mono, 87) + jnp.take(mono, 88) + \
        jnp.take(mono, 89) + jnp.take(mono, 90) + jnp.take(mono, 91)
    poly_44 = jnp.take(mono, 92) + jnp.take(mono, 93) + jnp.take(mono, 94)
    poly_45 = poly_1 * poly_7 - poly_43
    poly_46 = poly_2 * poly_7 - poly_44
    poly_47 = poly_3 * poly_7
    poly_48 = jnp.take(mono, 95) + jnp.take(mono, 96) + jnp.take(mono, 97) + jnp.take(mono, 98) + jnp.take(mono, 99) + jnp.take(mono, 100) + \
        jnp.take(mono, 101) + jnp.take(mono, 102) + jnp.take(mono, 103) + \
        jnp.take(mono, 104) + jnp.take(mono, 105) + jnp.take(mono, 106)
    poly_49 = poly_4 * poly_7 - poly_48
    poly_50 = poly_5 * poly_7
    poly_51 = poly_6 * poly_7
    poly_52 = jnp.take(mono, 107) + jnp.take(mono, 108) + jnp.take(mono, 109)
    poly_53 = poly_1 * poly_8
    poly_54 = poly_2 * poly_8
    poly_55 = jnp.take(mono, 110) + jnp.take(mono, 111)
    poly_56 = jnp.take(mono, 112) + jnp.take(mono, 113) + jnp.take(mono, 114) + \
        jnp.take(mono, 115) + jnp.take(mono, 116) + jnp.take(mono, 117)
    poly_57 = poly_3 * poly_8 - poly_55
    poly_58 = poly_4 * poly_8 - poly_56
    poly_59 = poly_5 * poly_8
    poly_60 = poly_6 * poly_8
    poly_61 = poly_7 * poly_8
    poly_62 = jnp.take(mono, 118)
    poly_63 = poly_9 * poly_1
    poly_64 = poly_9 * poly_2
    poly_65 = poly_9 * poly_3
    poly_66 = poly_9 * poly_4
    poly_67 = poly_5 * poly_9
    poly_68 = poly_6 * poly_9
    poly_69 = poly_9 * poly_7
    poly_70 = poly_9 * poly_8
    poly_71 = jnp.take(mono, 119) + jnp.take(mono, 120) + jnp.take(mono, 121) + \
        jnp.take(mono, 122) + jnp.take(mono, 123) + jnp.take(mono, 124)
    poly_72 = jnp.take(mono, 125) + jnp.take(mono, 126) + jnp.take(mono, 127)
    poly_73 = poly_1 * poly_10 - poly_71
    poly_74 = poly_2 * poly_10 - poly_72
    poly_75 = poly_3 * poly_10
    poly_76 = jnp.take(mono, 128) + jnp.take(mono, 129) + jnp.take(mono, 130) + jnp.take(mono, 131) + jnp.take(mono, 132) + jnp.take(mono, 133) + \
        jnp.take(mono, 134) + jnp.take(mono, 135) + jnp.take(mono, 136) + \
        jnp.take(mono, 137) + jnp.take(mono, 138) + jnp.take(mono, 139)
    poly_77 = poly_4 * poly_10 - poly_76
    poly_78 = poly_5 * poly_10
    poly_79 = poly_6 * poly_10
    poly_80 = jnp.take(mono, 140) + jnp.take(mono, 141) + jnp.take(mono, 142) + \
        jnp.take(mono, 143) + jnp.take(mono, 144) + jnp.take(mono, 145)
    poly_81 = poly_7 * poly_10 - poly_80
    poly_82 = poly_8 * poly_10
    poly_83 = poly_9 * poly_10
    poly_84 = jnp.take(mono, 146) + jnp.take(mono, 147) + jnp.take(mono, 148)
    poly_85 = poly_1 * poly_11
    poly_86 = poly_2 * poly_11
    poly_87 = jnp.take(mono, 149) + jnp.take(mono, 150)
    poly_88 = jnp.take(mono, 151) + jnp.take(mono, 152) + jnp.take(mono, 153) + \
        jnp.take(mono, 154) + jnp.take(mono, 155) + jnp.take(mono, 156)
    poly_89 = poly_3 * poly_11 - poly_87
    poly_90 = poly_4 * poly_11 - poly_88
    poly_91 = poly_5 * poly_11
    poly_92 = poly_6 * poly_11
    poly_93 = poly_7 * poly_11
    poly_94 = jnp.take(mono, 157) + jnp.take(mono, 158)
    poly_95 = poly_8 * poly_11 - poly_94
    poly_96 = poly_9 * poly_11
    poly_97 = poly_10 * poly_11
    poly_98 = jnp.take(mono, 159)
    poly_99 = poly_12 * poly_1
    poly_100 = poly_12 * poly_2
    poly_101 = poly_12 * poly_3
    poly_102 = poly_12 * poly_4
    poly_103 = poly_5 * poly_12
    poly_104 = poly_6 * poly_12
    poly_105 = poly_12 * poly_7
    poly_106 = poly_12 * poly_8
    poly_107 = poly_9 * poly_12
    poly_108 = poly_12 * poly_10
    poly_109 = poly_12 * poly_11
    poly_110 = poly_13 * poly_1
    poly_111 = poly_13 * poly_2
    poly_112 = poly_13 * poly_3
    poly_113 = poly_13 * poly_4
    poly_114 = poly_5 * poly_13
    poly_115 = poly_6 * poly_13
    poly_116 = poly_13 * poly_7
    poly_117 = poly_13 * poly_8
    poly_118 = poly_9 * poly_13
    poly_119 = poly_13 * poly_10
    poly_120 = poly_13 * poly_11
    poly_121 = poly_12 * poly_13
    poly_122 = jnp.take(mono, 160) + jnp.take(mono, 161) + jnp.take(mono, 162) + \
        jnp.take(mono, 163) + jnp.take(mono, 164) + jnp.take(mono, 165)
    poly_123 = jnp.take(mono, 166) + jnp.take(mono, 167) + jnp.take(mono, 168)
    poly_124 = poly_1 * poly_14 - poly_122
    poly_125 = poly_2 * poly_14 - poly_123
    poly_126 = poly_3 * poly_14
    poly_127 = jnp.take(mono, 169) + jnp.take(mono, 170) + jnp.take(mono, 171) + jnp.take(mono, 172) + jnp.take(mono, 173) + jnp.take(mono, 174) + \
        jnp.take(mono, 175) + jnp.take(mono, 176) + jnp.take(mono, 177) + \
        jnp.take(mono, 178) + jnp.take(mono, 179) + jnp.take(mono, 180)
    poly_128 = poly_4 * poly_14 - poly_127
    poly_129 = poly_5 * poly_14
    poly_130 = poly_6 * poly_14
    poly_131 = jnp.take(mono, 181) + jnp.take(mono, 182) + jnp.take(mono, 183) + \
        jnp.take(mono, 184) + jnp.take(mono, 185) + jnp.take(mono, 186)
    poly_132 = poly_7 * poly_14 - poly_131
    poly_133 = poly_8 * poly_14
    poly_134 = poly_9 * poly_14
    poly_135 = jnp.take(mono, 187) + jnp.take(mono, 188) + jnp.take(mono, 189) + \
        jnp.take(mono, 190) + jnp.take(mono, 191) + jnp.take(mono, 192)
    poly_136 = poly_10 * poly_14 - poly_135
    poly_137 = poly_11 * poly_14
    poly_138 = poly_12 * poly_14
    poly_139 = poly_13 * poly_14
    poly_140 = jnp.take(mono, 193) + jnp.take(mono, 194) + jnp.take(mono, 195)
    poly_141 = poly_1 * poly_15
    poly_142 = poly_2 * poly_15
    poly_143 = jnp.take(mono, 196) + jnp.take(mono, 197)
    poly_144 = jnp.take(mono, 198) + jnp.take(mono, 199) + jnp.take(mono, 200) + \
        jnp.take(mono, 201) + jnp.take(mono, 202) + jnp.take(mono, 203)
    poly_145 = poly_3 * poly_15 - poly_143
    poly_146 = poly_4 * poly_15 - poly_144
    poly_147 = poly_5 * poly_15
    poly_148 = poly_6 * poly_15
    poly_149 = poly_7 * poly_15
    poly_150 = jnp.take(mono, 204) + jnp.take(mono, 205)
    poly_151 = poly_8 * poly_15 - poly_150
    poly_152 = poly_9 * poly_15
    poly_153 = poly_10 * poly_15
    poly_154 = jnp.take(mono, 206) + jnp.take(mono, 207)
    poly_155 = poly_11 * poly_15 - poly_154
    poly_156 = poly_12 * poly_15
    poly_157 = poly_13 * poly_15
    poly_158 = poly_14 * poly_15
    poly_159 = jnp.take(mono, 208)
    poly_160 = poly_16 * poly_1
    poly_161 = poly_16 * poly_2
    poly_162 = poly_16 * poly_3
    poly_163 = poly_16 * poly_4
    poly_164 = poly_5 * poly_16
    poly_165 = poly_6 * poly_16
    poly_166 = poly_16 * poly_7
    poly_167 = poly_16 * poly_8
    poly_168 = poly_9 * poly_16
    poly_169 = poly_16 * poly_10
    poly_170 = poly_16 * poly_11
    poly_171 = poly_12 * poly_16
    poly_172 = poly_13 * poly_16
    poly_173 = poly_16 * poly_14
    poly_174 = poly_16 * poly_15
    poly_175 = poly_17 * poly_1
    poly_176 = poly_17 * poly_2
    poly_177 = poly_17 * poly_3
    poly_178 = poly_17 * poly_4
    poly_179 = poly_5 * poly_17
    poly_180 = poly_6 * poly_17
    poly_181 = poly_17 * poly_7
    poly_182 = poly_17 * poly_8
    poly_183 = poly_9 * poly_17
    poly_184 = poly_17 * poly_10
    poly_185 = poly_17 * poly_11
    poly_186 = poly_12 * poly_17
    poly_187 = poly_13 * poly_17
    poly_188 = poly_17 * poly_14
    poly_189 = poly_17 * poly_15
    poly_190 = poly_16 * poly_17
    poly_191 = poly_1 * poly_1 - poly_18 - poly_18
    poly_192 = poly_2 * poly_2 - poly_21 - poly_21
    poly_193 = poly_3 * poly_3 - poly_24 - poly_24
    poly_194 = poly_4 * poly_4 - poly_33 - poly_31 - \
        poly_30 - poly_33 - poly_31 - poly_30
    poly_195 = poly_5 * poly_5
    poly_196 = poly_6 * poly_6
    poly_197 = poly_7 * poly_7 - poly_52 - poly_52
    poly_198 = poly_8 * poly_8 - poly_62 - poly_62
    poly_199 = poly_9 * poly_9
    poly_200 = poly_10 * poly_10 - poly_84 - poly_84
    poly_201 = poly_11 * poly_11 - poly_98 - poly_98
    poly_202 = poly_12 * poly_12
    poly_203 = poly_13 * poly_13
    poly_204 = poly_14 * poly_14 - poly_140 - poly_140
    poly_205 = poly_15 * poly_15 - poly_159 - poly_159
    poly_206 = poly_16 * poly_16
    poly_207 = poly_17 * poly_17

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
                      poly_206,    poly_207,])

    return poly
