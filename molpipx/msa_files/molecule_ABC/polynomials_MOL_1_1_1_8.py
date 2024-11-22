import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_ABC.monomials_MOL_1_1_1_8 import f_monomials as f_monos 

# File created from ./MOL_1_1_1_8.POLY 

N_POLYS = 165

# Total number of monomials = 165 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(165) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) 
    poly_2 = jnp.take(mono,2) 
    poly_3 = jnp.take(mono,3) 
    poly_4 = poly_1 * poly_2 
    poly_5 = poly_1 * poly_3 
    poly_6 = poly_2 * poly_3 
    poly_7 = poly_1 * poly_1 
    poly_8 = poly_2 * poly_2 
    poly_9 = poly_3 * poly_3 
    poly_10 = poly_1 * poly_6 
    poly_11 = poly_1 * poly_4 
    poly_12 = poly_1 * poly_8 
    poly_13 = poly_1 * poly_5 
    poly_14 = poly_2 * poly_6 
    poly_15 = poly_1 * poly_9 
    poly_16 = poly_2 * poly_9 
    poly_17 = poly_1 * poly_7 
    poly_18 = poly_2 * poly_8 
    poly_19 = poly_3 * poly_9 
    poly_20 = poly_1 * poly_10 
    poly_21 = poly_1 * poly_14 
    poly_22 = poly_1 * poly_16 
    poly_23 = poly_1 * poly_11 
    poly_24 = poly_1 * poly_12 
    poly_25 = poly_1 * poly_18 
    poly_26 = poly_1 * poly_13 
    poly_27 = poly_2 * poly_14 
    poly_28 = poly_1 * poly_15 
    poly_29 = poly_2 * poly_16 
    poly_30 = poly_1 * poly_19 
    poly_31 = poly_2 * poly_19 
    poly_32 = poly_1 * poly_17 
    poly_33 = poly_2 * poly_18 
    poly_34 = poly_3 * poly_19 
    poly_35 = poly_1 * poly_20 
    poly_36 = poly_1 * poly_21 
    poly_37 = poly_1 * poly_27 
    poly_38 = poly_1 * poly_22 
    poly_39 = poly_1 * poly_29 
    poly_40 = poly_1 * poly_31 
    poly_41 = poly_1 * poly_23 
    poly_42 = poly_1 * poly_24 
    poly_43 = poly_1 * poly_25 
    poly_44 = poly_1 * poly_33 
    poly_45 = poly_1 * poly_26 
    poly_46 = poly_2 * poly_27 
    poly_47 = poly_1 * poly_28 
    poly_48 = poly_2 * poly_29 
    poly_49 = poly_1 * poly_30 
    poly_50 = poly_2 * poly_31 
    poly_51 = poly_1 * poly_34 
    poly_52 = poly_2 * poly_34 
    poly_53 = poly_1 * poly_32 
    poly_54 = poly_2 * poly_33 
    poly_55 = poly_3 * poly_34 
    poly_56 = poly_1 * poly_35 
    poly_57 = poly_1 * poly_36 
    poly_58 = poly_1 * poly_37 
    poly_59 = poly_1 * poly_46 
    poly_60 = poly_1 * poly_38 
    poly_61 = poly_1 * poly_39 
    poly_62 = poly_1 * poly_48 
    poly_63 = poly_1 * poly_40 
    poly_64 = poly_1 * poly_50 
    poly_65 = poly_1 * poly_52 
    poly_66 = poly_1 * poly_41 
    poly_67 = poly_1 * poly_42 
    poly_68 = poly_1 * poly_43 
    poly_69 = poly_1 * poly_44 
    poly_70 = poly_1 * poly_54 
    poly_71 = poly_1 * poly_45 
    poly_72 = poly_2 * poly_46 
    poly_73 = poly_1 * poly_47 
    poly_74 = poly_2 * poly_48 
    poly_75 = poly_1 * poly_49 
    poly_76 = poly_2 * poly_50 
    poly_77 = poly_1 * poly_51 
    poly_78 = poly_2 * poly_52 
    poly_79 = poly_1 * poly_55 
    poly_80 = poly_2 * poly_55 
    poly_81 = poly_1 * poly_53 
    poly_82 = poly_2 * poly_54 
    poly_83 = poly_3 * poly_55 
    poly_84 = poly_1 * poly_56 
    poly_85 = poly_1 * poly_57 
    poly_86 = poly_1 * poly_58 
    poly_87 = poly_1 * poly_59 
    poly_88 = poly_1 * poly_72 
    poly_89 = poly_1 * poly_60 
    poly_90 = poly_1 * poly_61 
    poly_91 = poly_1 * poly_62 
    poly_92 = poly_1 * poly_74 
    poly_93 = poly_1 * poly_63 
    poly_94 = poly_1 * poly_64 
    poly_95 = poly_1 * poly_76 
    poly_96 = poly_1 * poly_65 
    poly_97 = poly_1 * poly_78 
    poly_98 = poly_1 * poly_80 
    poly_99 = poly_1 * poly_66 
    poly_100 = poly_1 * poly_67 
    poly_101 = poly_1 * poly_68 
    poly_102 = poly_1 * poly_69 
    poly_103 = poly_1 * poly_70 
    poly_104 = poly_1 * poly_82 
    poly_105 = poly_1 * poly_71 
    poly_106 = poly_2 * poly_72 
    poly_107 = poly_1 * poly_73 
    poly_108 = poly_2 * poly_74 
    poly_109 = poly_1 * poly_75 
    poly_110 = poly_2 * poly_76 
    poly_111 = poly_1 * poly_77 
    poly_112 = poly_2 * poly_78 
    poly_113 = poly_1 * poly_79 
    poly_114 = poly_2 * poly_80 
    poly_115 = poly_1 * poly_83 
    poly_116 = poly_2 * poly_83 
    poly_117 = poly_1 * poly_81 
    poly_118 = poly_2 * poly_82 
    poly_119 = poly_3 * poly_83 
    poly_120 = poly_1 * poly_84 
    poly_121 = poly_1 * poly_85 
    poly_122 = poly_1 * poly_86 
    poly_123 = poly_1 * poly_87 
    poly_124 = poly_1 * poly_88 
    poly_125 = poly_1 * poly_106 
    poly_126 = poly_1 * poly_89 
    poly_127 = poly_1 * poly_90 
    poly_128 = poly_1 * poly_91 
    poly_129 = poly_1 * poly_92 
    poly_130 = poly_1 * poly_108 
    poly_131 = poly_1 * poly_93 
    poly_132 = poly_1 * poly_94 
    poly_133 = poly_1 * poly_95 
    poly_134 = poly_1 * poly_110 
    poly_135 = poly_1 * poly_96 
    poly_136 = poly_1 * poly_97 
    poly_137 = poly_1 * poly_112 
    poly_138 = poly_1 * poly_98 
    poly_139 = poly_1 * poly_114 
    poly_140 = poly_1 * poly_116 
    poly_141 = poly_1 * poly_99 
    poly_142 = poly_1 * poly_100 
    poly_143 = poly_1 * poly_101 
    poly_144 = poly_1 * poly_102 
    poly_145 = poly_1 * poly_103 
    poly_146 = poly_1 * poly_104 
    poly_147 = poly_1 * poly_118 
    poly_148 = poly_1 * poly_105 
    poly_149 = poly_2 * poly_106 
    poly_150 = poly_1 * poly_107 
    poly_151 = poly_2 * poly_108 
    poly_152 = poly_1 * poly_109 
    poly_153 = poly_2 * poly_110 
    poly_154 = poly_1 * poly_111 
    poly_155 = poly_2 * poly_112 
    poly_156 = poly_1 * poly_113 
    poly_157 = poly_2 * poly_114 
    poly_158 = poly_1 * poly_115 
    poly_159 = poly_2 * poly_116 
    poly_160 = poly_1 * poly_119 
    poly_161 = poly_2 * poly_119 
    poly_162 = poly_1 * poly_117 
    poly_163 = poly_2 * poly_118 
    poly_164 = poly_3 * poly_119 

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
    poly_161,    poly_162,    poly_163,    poly_164,    ]) 

    return poly 



