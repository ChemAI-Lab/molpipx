import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A4.monomials_MOL_4_8 import f_monomials as f_monos

# File created from ./MOL_4_8.POLY

N_POLYS = 195

# Total number of monomials = 195


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(195)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2) + jnp.take(mono, 3) + \
        jnp.take(mono, 4) + jnp.take(mono, 5) + jnp.take(mono, 6)
    poly_2 = jnp.take(mono, 7) + jnp.take(mono, 8) + jnp.take(mono, 9)
    poly_3 = jnp.take(mono, 10) + jnp.take(mono, 11) + jnp.take(mono, 12) + jnp.take(mono, 13) + jnp.take(mono, 14) + jnp.take(mono, 15) + \
        jnp.take(mono, 16) + jnp.take(mono, 17) + jnp.take(mono, 18) + \
        jnp.take(mono, 19) + jnp.take(mono, 20) + jnp.take(mono, 21)
    poly_4 = poly_1 * poly_1 - poly_3 - poly_2 - poly_3 - poly_2
    poly_5 = jnp.take(mono, 22) + jnp.take(mono, 23) + jnp.take(mono, 24) + jnp.take(mono, 25) + jnp.take(mono, 26) + jnp.take(mono, 27) + \
        jnp.take(mono, 28) + jnp.take(mono, 29) + jnp.take(mono, 30) + \
        jnp.take(mono, 31) + jnp.take(mono, 32) + jnp.take(mono, 33)
    poly_6 = jnp.take(mono, 34) + jnp.take(mono, 35) + \
        jnp.take(mono, 36) + jnp.take(mono, 37)
    poly_7 = jnp.take(mono, 38) + jnp.take(mono, 39) + \
        jnp.take(mono, 40) + jnp.take(mono, 41)
    poly_8 = poly_1 * poly_2 - poly_5
    poly_9 = poly_1 * poly_3 - poly_6 - poly_7 - \
        poly_5 - poly_6 - poly_7 - poly_5 - poly_6 - poly_7
    poly_10 = poly_1 * poly_4 - poly_9 - poly_8
    poly_11 = jnp.take(mono, 42) + jnp.take(mono, 43) + jnp.take(mono, 44)
    poly_12 = jnp.take(mono, 45) + jnp.take(mono, 46) + jnp.take(mono, 47) + jnp.take(mono, 48) + jnp.take(mono, 49) + jnp.take(mono, 50) + \
        jnp.take(mono, 51) + jnp.take(mono, 52) + jnp.take(mono, 53) + \
        jnp.take(mono, 54) + jnp.take(mono, 55) + jnp.take(mono, 56)
    poly_13 = poly_2 * poly_3 - poly_12
    poly_14 = poly_1 * poly_5 - poly_12 - poly_11 - \
        poly_13 - poly_12 - poly_11 - poly_11 - poly_11
    poly_15 = poly_1 * poly_6 - poly_12
    poly_16 = poly_1 * poly_7 - poly_12
    poly_17 = poly_2 * poly_2 - poly_11 - poly_11
    poly_18 = poly_3 * poly_3 - poly_12 - poly_11 - poly_15 - poly_16 - poly_14 - \
        poly_12 - poly_11 - poly_15 - poly_16 - \
        poly_14 - poly_12 - poly_11 - poly_12 - poly_11
    poly_19 = poly_2 * poly_4 - poly_14
    poly_20 = poly_3 * poly_4 - poly_15 - poly_16 - poly_13
    poly_21 = poly_1 * poly_10 - poly_20 - poly_19
    poly_22 = jnp.take(mono, 57) + jnp.take(mono, 58) + jnp.take(mono, 59) + \
        jnp.take(mono, 60) + jnp.take(mono, 61) + jnp.take(mono, 62)
    poly_23 = poly_1 * poly_11 - poly_22
    poly_24 = poly_2 * poly_6
    poly_25 = poly_2 * poly_7
    poly_26 = poly_1 * poly_12 - poly_22 - poly_24 - \
        poly_25 - poly_22 - poly_22 - poly_22
    poly_27 = poly_2 * poly_5 - poly_22 - poly_23 - poly_22
    poly_28 = poly_3 * poly_5 - poly_22 - poly_26 - poly_24 - poly_25 - \
        poly_23 - poly_22 - poly_24 - poly_25 - poly_23 - poly_22 - poly_22
    poly_29 = poly_3 * poly_6 - poly_22 - poly_26 - poly_22
    poly_30 = poly_3 * poly_7 - poly_22 - poly_26 - poly_22
    poly_31 = poly_2 * poly_9 - poly_26 - poly_28
    poly_32 = poly_1 * poly_14 - poly_26 - poly_23 - poly_28
    poly_33 = poly_4 * poly_6 - poly_25
    poly_34 = poly_4 * poly_7 - poly_24
    poly_35 = poly_1 * poly_17 - poly_27
    poly_36 = poly_1 * poly_18 - poly_29 - poly_30 - poly_28
    poly_37 = poly_2 * poly_10 - poly_32
    poly_38 = poly_3 * poly_10 - poly_33 - poly_34 - poly_31
    poly_39 = poly_1 * poly_21 - poly_38 - poly_37
    poly_40 = jnp.take(mono, 63)
    poly_41 = jnp.take(mono, 64) + jnp.take(mono, 65) + jnp.take(mono, 66) + jnp.take(mono, 67) + jnp.take(mono, 68) + jnp.take(mono, 69) + jnp.take(mono, 70) + jnp.take(mono, 71) + jnp.take(mono, 72) + jnp.take(mono, 73) + jnp.take(mono, 74) + jnp.take(mono, 75) + \
        jnp.take(mono, 76) + jnp.take(mono, 77) + jnp.take(mono, 78) + jnp.take(mono, 79) + jnp.take(mono, 80) + jnp.take(mono, 81) + \
        jnp.take(mono, 82) + jnp.take(mono, 83) + jnp.take(mono, 84) + \
        jnp.take(mono, 85) + jnp.take(mono, 86) + jnp.take(mono, 87)
    poly_42 = poly_1 * poly_22 - poly_40 - poly_41 - \
        poly_40 - poly_40 - poly_40 - poly_40 - poly_40
    poly_43 = poly_2 * poly_11 - poly_40 - poly_40 - poly_40
    poly_44 = poly_2 * poly_12 - poly_41
    poly_45 = poly_3 * poly_11 - poly_41
    poly_46 = poly_5 * poly_6 - poly_41
    poly_47 = poly_5 * poly_7 - poly_41
    poly_48 = poly_6 * poly_7 - poly_40 - poly_40 - poly_40 - poly_40
    poly_49 = poly_4 * poly_11 - poly_42
    poly_50 = poly_2 * poly_15 - poly_46
    poly_51 = poly_2 * poly_16 - poly_47
    poly_52 = poly_4 * poly_12 - poly_41 - poly_50 - poly_51
    poly_53 = poly_2 * poly_14 - poly_42 - poly_49 - poly_42
    poly_54 = poly_6 * poly_6 - poly_42 - poly_42
    poly_55 = poly_7 * poly_7 - poly_42 - poly_42
    poly_56 = poly_3 * poly_17 - poly_44
    poly_57 = poly_2 * poly_18 - poly_48
    poly_58 = poly_3 * poly_14 - poly_41 - poly_52 - \
        poly_46 - poly_47 - poly_45 - poly_45
    poly_59 = poly_6 * poly_9 - poly_41 - poly_52 - poly_47
    poly_60 = poly_7 * poly_9 - poly_41 - poly_52 - poly_46
    poly_61 = poly_2 * poly_20 - poly_52 - poly_58
    poly_62 = poly_1 * poly_32 - poly_52 - poly_49 - poly_58
    poly_63 = poly_6 * poly_10 - poly_51
    poly_64 = poly_7 * poly_10 - poly_50
    poly_65 = poly_2 * poly_17 - poly_43
    poly_66 = poly_3 * poly_18 - poly_46 - poly_47 - \
        poly_45 - poly_59 - poly_60 - poly_58
    poly_67 = poly_2 * poly_19 - poly_49
    poly_68 = poly_1 * poly_36 - poly_59 - poly_60 - \
        poly_58 - poly_57 - poly_66 - poly_66
    poly_69 = poly_2 * poly_21 - poly_62
    poly_70 = poly_3 * poly_21 - poly_63 - poly_64 - poly_61
    poly_71 = poly_1 * poly_39 - poly_70 - poly_69
    poly_72 = poly_40 * poly_1
    poly_73 = poly_2 * poly_22 - poly_72
    poly_74 = poly_6 * poly_11
    poly_75 = poly_7 * poly_11
    poly_76 = poly_3 * poly_22 - poly_72 - poly_74 - \
        poly_75 - poly_72 - poly_72 - poly_72
    poly_77 = jnp.take(mono, 88) + jnp.take(mono, 89) + jnp.take(mono, 90) + jnp.take(mono, 91) + jnp.take(mono, 92) + jnp.take(mono, 93) + jnp.take(mono, 94) + jnp.take(mono, 95) + jnp.take(mono, 96) + jnp.take(mono, 97) + jnp.take(mono, 98) + jnp.take(mono, 99) + \
        jnp.take(mono, 100) + jnp.take(mono, 101) + jnp.take(mono, 102) + jnp.take(mono, 103) + jnp.take(mono, 104) + jnp.take(mono, 105) + \
        jnp.take(mono, 106) + jnp.take(mono, 107) + jnp.take(mono, 108) + \
        jnp.take(mono, 109) + jnp.take(mono, 110) + jnp.take(mono, 111)
    poly_78 = poly_1 * poly_42 - poly_72 - poly_76
    poly_79 = poly_5 * poly_11 - poly_72 - poly_73 - poly_72
    poly_80 = poly_2 * poly_26 - poly_76 - poly_77
    poly_81 = poly_6 * poly_12 - poly_72 - poly_76 - poly_72
    poly_82 = poly_7 * poly_12 - poly_72 - poly_76 - poly_72
    poly_83 = poly_8 * poly_11 - poly_72
    poly_84 = poly_6 * poly_17
    poly_85 = poly_7 * poly_17
    poly_86 = poly_9 * poly_11 - poly_76 - poly_77
    poly_87 = poly_2 * poly_29 - poly_81
    poly_88 = poly_6 * poly_14 - poly_75 - poly_75
    poly_89 = poly_2 * poly_30 - poly_82
    poly_90 = poly_7 * poly_14 - poly_74 - poly_74
    poly_91 = poly_1 * poly_48 - poly_76 - poly_81 - poly_82
    poly_92 = poly_10 * poly_11 - poly_78
    poly_93 = poly_2 * poly_33 - poly_88
    poly_94 = poly_2 * poly_34 - poly_90
    poly_95 = poly_10 * poly_12 - poly_77 - poly_93 - poly_94
    poly_96 = poly_2 * poly_27 - poly_73 - poly_79
    poly_97 = poly_2 * poly_28 - poly_76 - poly_86
    poly_98 = poly_1 * poly_53 - poly_80 - poly_79 - poly_97
    poly_99 = poly_1 * poly_54 - poly_81
    poly_100 = poly_1 * poly_55 - poly_82
    poly_101 = poly_5 * poly_18 - poly_76 - poly_91 - poly_87 - poly_89 - poly_86
    poly_102 = poly_6 * poly_18 - poly_74 - poly_90
    poly_103 = poly_7 * poly_18 - poly_75 - poly_88
    poly_104 = poly_2 * poly_31 - poly_77 - poly_86
    poly_105 = poly_2 * poly_36 - poly_91 - poly_101
    poly_106 = poly_3 * poly_32 - poly_77 - poly_95 - poly_88 - poly_90 - poly_86
    poly_107 = poly_4 * poly_29 - poly_82 - poly_80 - poly_99
    poly_108 = poly_4 * poly_30 - poly_81 - poly_80 - poly_100
    poly_109 = poly_2 * poly_38 - poly_95 - poly_106
    poly_110 = poly_1 * poly_62 - poly_95 - poly_92 - poly_106
    poly_111 = poly_6 * poly_21 - poly_94
    poly_112 = poly_7 * poly_21 - poly_93
    poly_113 = poly_1 * poly_65 - poly_96
    poly_114 = poly_1 * poly_66 - poly_102 - poly_103 - poly_101
    poly_115 = poly_2 * poly_37 - poly_92
    poly_116 = poly_10 * poly_18 - poly_99 - poly_100 - poly_97
    poly_117 = poly_2 * poly_39 - poly_110
    poly_118 = poly_3 * poly_39 - poly_111 - poly_112 - poly_109
    poly_119 = poly_1 * poly_71 - poly_118 - poly_117
    poly_120 = poly_40 * poly_2
    poly_121 = poly_40 * poly_3
    poly_122 = poly_40 * poly_4
    poly_123 = poly_11 * poly_12 - poly_121
    poly_124 = poly_2 * poly_42 - poly_122
    poly_125 = poly_6 * poly_22 - poly_121
    poly_126 = poly_7 * poly_22 - poly_121
    poly_127 = poly_2 * poly_41 - poly_121 - poly_123 - poly_121
    poly_128 = poly_6 * poly_23 - poly_123
    poly_129 = poly_7 * poly_23 - poly_123
    poly_130 = poly_3 * poly_41 - poly_122 - poly_121 - poly_120 - poly_125 - poly_128 - poly_126 - poly_129 - poly_124 - poly_123 - poly_122 - poly_121 - poly_120 - \
        poly_125 - poly_126 - poly_124 - poly_123 - poly_122 - poly_121 - poly_120 - \
        poly_122 - poly_121 - poly_120 - poly_120 - poly_120 - poly_120 - poly_120
    poly_131 = poly_3 * poly_42 - poly_121 - poly_125 - poly_126 - poly_121
    poly_132 = poly_4 * poly_41 - poly_121 - poly_131 - \
        poly_128 - poly_129 - poly_127 - poly_121
    poly_133 = poly_1 * poly_78 - poly_122 - poly_131
    poly_134 = poly_11 * poly_11 - poly_120 - poly_120
    poly_135 = poly_2 * poly_48 - poly_130
    poly_136 = poly_11 * poly_17 - poly_120
    poly_137 = poly_2 * poly_44 - poly_123
    poly_138 = poly_2 * poly_45 - poly_121
    poly_139 = poly_11 * poly_14 - poly_122 - poly_124 - poly_122
    poly_140 = poly_6 * poly_27 - poly_127
    poly_141 = poly_2 * poly_54
    poly_142 = poly_7 * poly_27 - poly_127
    poly_143 = poly_2 * poly_52 - poly_131 - poly_132
    poly_144 = poly_1 * poly_81 - poly_125 - poly_141 - poly_135 - poly_125
    poly_145 = poly_2 * poly_55
    poly_146 = poly_1 * poly_82 - poly_126 - poly_135 - poly_145 - poly_126
    poly_147 = poly_11 * poly_18 - poly_130
    poly_148 = poly_6 * poly_28 - poly_129 - poly_123 - poly_143
    poly_149 = poly_7 * poly_28 - poly_128 - poly_123 - poly_143
    poly_150 = poly_6 * poly_30 - poly_121 - poly_146
    poly_151 = poly_11 * poly_19 - poly_122
    poly_152 = poly_2 * poly_50 - poly_128
    poly_153 = poly_2 * poly_51 - poly_129
    poly_154 = poly_11 * poly_20 - poly_131 - poly_132
    poly_155 = poly_2 * poly_59 - poly_144 - poly_148
    poly_156 = poly_6 * poly_32 - poly_129
    poly_157 = poly_2 * poly_60 - poly_146 - poly_149
    poly_158 = poly_7 * poly_32 - poly_128
    poly_159 = poly_6 * poly_34 - poly_122 - poly_145 - poly_122
    poly_160 = poly_11 * poly_21 - poly_133
    poly_161 = poly_2 * poly_63 - poly_156
    poly_162 = poly_2 * poly_64 - poly_158
    poly_163 = poly_12 * poly_21 - poly_132 - poly_161 - poly_162
    poly_164 = poly_2 * poly_53 - poly_124 - poly_139
    poly_165 = poly_2 * poly_58 - poly_131 - poly_154
    poly_166 = poly_3 * poly_54 - poly_125 - poly_144
    poly_167 = poly_3 * poly_55 - poly_126 - poly_146
    poly_168 = poly_3 * poly_65 - poly_137
    poly_169 = poly_17 * poly_18 - poly_135
    poly_170 = poly_1 * poly_98 - poly_143 - poly_139 - poly_165
    poly_171 = poly_4 * poly_54 - poly_135
    poly_172 = poly_4 * poly_55 - poly_135
    poly_173 = poly_2 * poly_66 - poly_150
    poly_174 = poly_1 * poly_101 - poly_148 - poly_149 - \
        poly_147 - poly_173 - poly_165 - poly_147
    poly_175 = poly_1 * poly_102 - poly_150 - poly_148 - poly_166
    poly_176 = poly_1 * poly_103 - poly_150 - poly_149 - poly_167
    poly_177 = poly_2 * poly_61 - poly_132 - poly_154
    poly_178 = poly_2 * poly_68 - poly_159 - poly_174
    poly_179 = poly_3 * poly_62 - poly_132 - \
        poly_163 - poly_156 - poly_158 - poly_154
    poly_180 = poly_6 * poly_38 - poly_132 - poly_163 - poly_157
    poly_181 = poly_7 * poly_38 - poly_132 - poly_163 - poly_155
    poly_182 = poly_2 * poly_70 - poly_163 - poly_179
    poly_183 = poly_1 * poly_110 - poly_163 - poly_160 - poly_179
    poly_184 = poly_6 * poly_39 - poly_162
    poly_185 = poly_7 * poly_39 - poly_161
    poly_186 = poly_2 * poly_65 - poly_136
    poly_187 = poly_3 * poly_66 - poly_148 - poly_149 - \
        poly_147 - poly_175 - poly_176 - poly_174
    poly_188 = poly_2 * poly_67 - poly_151
    poly_189 = poly_4 * poly_66 - poly_166 - poly_167 - poly_165
    poly_190 = poly_2 * poly_69 - poly_160
    poly_191 = poly_18 * poly_21 - poly_171 - poly_172 - poly_169
    poly_192 = poly_2 * poly_71 - poly_183
    poly_193 = poly_3 * poly_71 - poly_184 - poly_185 - poly_182
    poly_194 = poly_1 * poly_119 - poly_193 - poly_192

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
                      poly_191,    poly_192,    poly_193,    poly_194,])

    return poly
