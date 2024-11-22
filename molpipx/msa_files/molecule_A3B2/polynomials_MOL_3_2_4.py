import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A3B2.monomials_MOL_3_2_4 import f_monomials as f_monos 

# File created from ./MOL_3_2_4.POLY 

N_POLYS = 139

# Total number of monomials = 139 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(139) 

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
    poly_136,    poly_137,    poly_138,    ]) 

    return poly 



