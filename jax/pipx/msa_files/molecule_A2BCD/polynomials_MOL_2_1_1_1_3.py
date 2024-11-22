import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A2BCD.monomials_MOL_2_1_1_1_3 import f_monomials as f_monos 

# File created from ./MOL_2_1_1_1_3.POLY 

N_POLYS = 168

# Total number of monomials = 168 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(168) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = jnp.take(mono,4) + jnp.take(mono,5) 
    poly_5 = jnp.take(mono,6) + jnp.take(mono,7) 
    poly_6 = jnp.take(mono,8) + jnp.take(mono,9) 
    poly_7 = jnp.take(mono,10) 
    poly_8 = poly_1 * poly_2 
    poly_9 = poly_1 * poly_3 
    poly_10 = poly_2 * poly_3 
    poly_11 = poly_1 * poly_4 
    poly_12 = poly_2 * poly_4 
    poly_13 = poly_3 * poly_4 
    poly_14 = jnp.take(mono,11) 
    poly_15 = poly_1 * poly_5 
    poly_16 = poly_2 * poly_5 
    poly_17 = poly_3 * poly_5 
    poly_18 = jnp.take(mono,12) + jnp.take(mono,13) 
    poly_19 = jnp.take(mono,14) 
    poly_20 = poly_4 * poly_5 - poly_18 
    poly_21 = poly_1 * poly_6 
    poly_22 = poly_2 * poly_6 
    poly_23 = poly_3 * poly_6 
    poly_24 = jnp.take(mono,15) + jnp.take(mono,16) 
    poly_25 = jnp.take(mono,17) + jnp.take(mono,18) 
    poly_26 = jnp.take(mono,19) 
    poly_27 = poly_4 * poly_6 - poly_24 
    poly_28 = poly_5 * poly_6 - poly_25 
    poly_29 = poly_1 * poly_7 
    poly_30 = poly_2 * poly_7 
    poly_31 = poly_3 * poly_7 
    poly_32 = poly_7 * poly_4 
    poly_33 = poly_7 * poly_5 
    poly_34 = poly_7 * poly_6 
    poly_35 = poly_1 * poly_1 
    poly_36 = poly_2 * poly_2 
    poly_37 = poly_3 * poly_3 
    poly_38 = poly_4 * poly_4 - poly_14 - poly_14 
    poly_39 = poly_5 * poly_5 - poly_19 - poly_19 
    poly_40 = poly_6 * poly_6 - poly_26 - poly_26 
    poly_41 = poly_7 * poly_7 
    poly_42 = poly_1 * poly_10 
    poly_43 = poly_1 * poly_12 
    poly_44 = poly_1 * poly_13 
    poly_45 = poly_2 * poly_13 
    poly_46 = poly_1 * poly_14 
    poly_47 = poly_2 * poly_14 
    poly_48 = poly_3 * poly_14 
    poly_49 = poly_1 * poly_16 
    poly_50 = poly_1 * poly_17 
    poly_51 = poly_2 * poly_17 
    poly_52 = poly_1 * poly_18 
    poly_53 = poly_2 * poly_18 
    poly_54 = poly_3 * poly_18 
    poly_55 = poly_1 * poly_19 
    poly_56 = poly_2 * poly_19 
    poly_57 = poly_3 * poly_19 
    poly_58 = poly_1 * poly_20 
    poly_59 = poly_2 * poly_20 
    poly_60 = poly_3 * poly_20 
    poly_61 = poly_14 * poly_5 
    poly_62 = poly_19 * poly_4 
    poly_63 = poly_1 * poly_22 
    poly_64 = poly_1 * poly_23 
    poly_65 = poly_2 * poly_23 
    poly_66 = poly_1 * poly_24 
    poly_67 = poly_2 * poly_24 
    poly_68 = poly_3 * poly_24 
    poly_69 = poly_1 * poly_25 
    poly_70 = poly_2 * poly_25 
    poly_71 = poly_3 * poly_25 
    poly_72 = jnp.take(mono,20) + jnp.take(mono,21) 
    poly_73 = poly_1 * poly_26 
    poly_74 = poly_2 * poly_26 
    poly_75 = poly_3 * poly_26 
    poly_76 = poly_1 * poly_27 
    poly_77 = poly_2 * poly_27 
    poly_78 = poly_3 * poly_27 
    poly_79 = poly_14 * poly_6 
    poly_80 = poly_4 * poly_25 - poly_72 
    poly_81 = poly_26 * poly_4 
    poly_82 = poly_1 * poly_28 
    poly_83 = poly_2 * poly_28 
    poly_84 = poly_3 * poly_28 
    poly_85 = poly_5 * poly_24 - poly_72 
    poly_86 = poly_19 * poly_6 
    poly_87 = poly_26 * poly_5 
    poly_88 = poly_4 * poly_28 - poly_85 
    poly_89 = poly_1 * poly_30 
    poly_90 = poly_1 * poly_31 
    poly_91 = poly_2 * poly_31 
    poly_92 = poly_1 * poly_32 
    poly_93 = poly_2 * poly_32 
    poly_94 = poly_3 * poly_32 
    poly_95 = poly_7 * poly_14 
    poly_96 = poly_1 * poly_33 
    poly_97 = poly_2 * poly_33 
    poly_98 = poly_3 * poly_33 
    poly_99 = poly_7 * poly_18 
    poly_100 = poly_7 * poly_19 
    poly_101 = poly_7 * poly_20 
    poly_102 = poly_1 * poly_34 
    poly_103 = poly_2 * poly_34 
    poly_104 = poly_3 * poly_34 
    poly_105 = poly_7 * poly_24 
    poly_106 = poly_7 * poly_25 
    poly_107 = poly_7 * poly_26 
    poly_108 = poly_7 * poly_27 
    poly_109 = poly_7 * poly_28 
    poly_110 = poly_1 * poly_8 
    poly_111 = poly_1 * poly_36 
    poly_112 = poly_1 * poly_9 
    poly_113 = poly_2 * poly_10 
    poly_114 = poly_1 * poly_37 
    poly_115 = poly_2 * poly_37 
    poly_116 = poly_1 * poly_11 
    poly_117 = poly_2 * poly_12 
    poly_118 = poly_3 * poly_13 
    poly_119 = poly_1 * poly_38 
    poly_120 = poly_2 * poly_38 
    poly_121 = poly_3 * poly_38 
    poly_122 = poly_14 * poly_4 
    poly_123 = poly_1 * poly_15 
    poly_124 = poly_2 * poly_16 
    poly_125 = poly_3 * poly_17 
    poly_126 = poly_4 * poly_18 - poly_61 
    poly_127 = poly_4 * poly_20 - poly_61 
    poly_128 = poly_1 * poly_39 
    poly_129 = poly_2 * poly_39 
    poly_130 = poly_3 * poly_39 
    poly_131 = poly_5 * poly_18 - poly_62 
    poly_132 = poly_19 * poly_5 
    poly_133 = poly_4 * poly_39 - poly_131 
    poly_134 = poly_1 * poly_21 
    poly_135 = poly_2 * poly_22 
    poly_136 = poly_3 * poly_23 
    poly_137 = poly_4 * poly_24 - poly_79 
    poly_138 = poly_5 * poly_25 - poly_86 
    poly_139 = poly_4 * poly_27 - poly_79 
    poly_140 = poly_5 * poly_28 - poly_86 
    poly_141 = poly_1 * poly_40 
    poly_142 = poly_2 * poly_40 
    poly_143 = poly_3 * poly_40 
    poly_144 = poly_6 * poly_24 - poly_81 
    poly_145 = poly_6 * poly_25 - poly_87 
    poly_146 = poly_26 * poly_6 
    poly_147 = poly_4 * poly_40 - poly_144 
    poly_148 = poly_5 * poly_40 - poly_145 
    poly_149 = poly_1 * poly_29 
    poly_150 = poly_2 * poly_30 
    poly_151 = poly_3 * poly_31 
    poly_152 = poly_7 * poly_38 
    poly_153 = poly_7 * poly_39 
    poly_154 = poly_7 * poly_40 
    poly_155 = poly_1 * poly_41 
    poly_156 = poly_2 * poly_41 
    poly_157 = poly_3 * poly_41 
    poly_158 = poly_7 * poly_32 
    poly_159 = poly_7 * poly_33 
    poly_160 = poly_7 * poly_34 
    poly_161 = poly_1 * poly_35 
    poly_162 = poly_2 * poly_36 
    poly_163 = poly_3 * poly_37 
    poly_164 = poly_4 * poly_38 - poly_122 
    poly_165 = poly_5 * poly_39 - poly_132 
    poly_166 = poly_6 * poly_40 - poly_146 
    poly_167 = poly_7 * poly_41 

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
    poly_166,    poly_167,    ]) 

    return poly 



