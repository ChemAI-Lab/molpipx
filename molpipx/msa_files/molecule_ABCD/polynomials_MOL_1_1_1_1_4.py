import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABCD.monomials_MOL_1_1_1_1_4 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_1_4.POLY 

N_POLYS = 210

# Total number of monomials = 210 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(210) 

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
    poly_206,    poly_207,    poly_208,    poly_209,    ]) 

    return poly 



