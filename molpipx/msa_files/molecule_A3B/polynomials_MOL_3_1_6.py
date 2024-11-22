import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3B.monomials_MOL_3_1_6 import f_monomials as f_monos 

# File created from ./MOL_3_1_6.POLY 

N_POLYS = 196

# Total number of monomials = 196 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(196) 

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
    ]) 

    return poly 



