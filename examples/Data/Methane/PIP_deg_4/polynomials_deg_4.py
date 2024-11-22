import jax.numpy as jnp 
from jax import jit

from monomials_deg_4 import f_monomials as f_monos 


# File created from ./MOL_4_1_4.POLY 


N_POLYS = 83

# Total number of monomials = 83 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(83) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) + jnp.take(mono,2) + jnp.take(mono,3) + jnp.take(mono,4) 
    poly_2 = jnp.take(mono,5) + jnp.take(mono,6) + jnp.take(mono,7) + jnp.take(mono,8) + jnp.take(mono,9) + jnp.take(mono,10) 
    poly_3 = jnp.take(mono,11) + jnp.take(mono,12) + jnp.take(mono,13) + jnp.take(mono,14) + jnp.take(mono,15) + jnp.take(mono,16) 
    poly_4 = jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) + jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) + jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) + jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) 
    poly_5 = jnp.take(mono,29) + jnp.take(mono,30) + jnp.take(mono,31) 
    poly_6 = poly_1 * poly_2 - poly_4 
    poly_7 = jnp.take(mono,32) + jnp.take(mono,33) + jnp.take(mono,34) + jnp.take(mono,35) + jnp.take(mono,36) + jnp.take(mono,37) + jnp.take(mono,38) + jnp.take(mono,39) + jnp.take(mono,40) + jnp.take(mono,41) + jnp.take(mono,42) + jnp.take(mono,43) 
    poly_8 = poly_1 * poly_1 - poly_3 - poly_3 
    poly_9 = poly_2 * poly_2 - poly_7 - poly_5 - poly_7 - poly_5 
    poly_10 = jnp.take(mono,44) + jnp.take(mono,45) + jnp.take(mono,46) + jnp.take(mono,47) 
    poly_11 = jnp.take(mono,48) + jnp.take(mono,49) + jnp.take(mono,50) + jnp.take(mono,51) + jnp.take(mono,52) + jnp.take(mono,53) 
    poly_12 = jnp.take(mono,54) + jnp.take(mono,55) + jnp.take(mono,56) + jnp.take(mono,57) + jnp.take(mono,58) + jnp.take(mono,59) + jnp.take(mono,60) + jnp.take(mono,61) + jnp.take(mono,62) + jnp.take(mono,63) + jnp.take(mono,64) + jnp.take(mono,65) + jnp.take(mono,66) + jnp.take(mono,67) + jnp.take(mono,68) + jnp.take(mono,69) + jnp.take(mono,70) + jnp.take(mono,71) + jnp.take(mono,72) + jnp.take(mono,73) + jnp.take(mono,74) + jnp.take(mono,75) + jnp.take(mono,76) + jnp.take(mono,77) 
    poly_13 = poly_1 * poly_5 
    poly_14 = poly_2 * poly_3 - poly_12 - poly_11 
    poly_15 = jnp.take(mono,78) + jnp.take(mono,79) + jnp.take(mono,80) + jnp.take(mono,81) + jnp.take(mono,82) + jnp.take(mono,83) + jnp.take(mono,84) + jnp.take(mono,85) + jnp.take(mono,86) + jnp.take(mono,87) + jnp.take(mono,88) + jnp.take(mono,89) 
    poly_16 = jnp.take(mono,90) + jnp.take(mono,91) + jnp.take(mono,92) + jnp.take(mono,93) + jnp.take(mono,94) + jnp.take(mono,95) + jnp.take(mono,96) + jnp.take(mono,97) + jnp.take(mono,98) + jnp.take(mono,99) + jnp.take(mono,100) + jnp.take(mono,101) + jnp.take(mono,102) + jnp.take(mono,103) + jnp.take(mono,104) + jnp.take(mono,105) + jnp.take(mono,106) + jnp.take(mono,107) + jnp.take(mono,108) + jnp.take(mono,109) + jnp.take(mono,110) + jnp.take(mono,111) + jnp.take(mono,112) + jnp.take(mono,113) 
    poly_17 = jnp.take(mono,114) + jnp.take(mono,115) + jnp.take(mono,116) + jnp.take(mono,117) + jnp.take(mono,118) + jnp.take(mono,119) + jnp.take(mono,120) + jnp.take(mono,121) + jnp.take(mono,122) + jnp.take(mono,123) + jnp.take(mono,124) + jnp.take(mono,125) 
    poly_18 = jnp.take(mono,126) + jnp.take(mono,127) + jnp.take(mono,128) + jnp.take(mono,129) 
    poly_19 = poly_1 * poly_7 - poly_16 - poly_15 
    poly_20 = jnp.take(mono,130) + jnp.take(mono,131) + jnp.take(mono,132) + jnp.take(mono,133) 
    poly_21 = poly_1 * poly_3 - poly_10 - poly_10 - poly_10 
    poly_22 = poly_1 * poly_4 - poly_12 - poly_11 - poly_11 
    poly_23 = poly_2 * poly_8 - poly_22 
    poly_24 = poly_2 * poly_4 - poly_16 - poly_15 - poly_13 - poly_15 
    poly_25 = poly_2 * poly_5 - poly_17 
    poly_26 = poly_1 * poly_9 - poly_24 
    poly_27 = poly_2 * poly_7 - poly_18 - poly_20 - poly_17 - poly_18 - poly_20 - poly_17 - poly_18 - poly_20 
    poly_28 = poly_1 * poly_8 - poly_21 
    poly_29 = poly_2 * poly_9 - poly_27 - poly_25 
    poly_30 = jnp.take(mono,134) 
    poly_31 = jnp.take(mono,135) + jnp.take(mono,136) + jnp.take(mono,137) + jnp.take(mono,138) + jnp.take(mono,139) + jnp.take(mono,140) + jnp.take(mono,141) + jnp.take(mono,142) + jnp.take(mono,143) + jnp.take(mono,144) + jnp.take(mono,145) + jnp.take(mono,146) 
    poly_32 = jnp.take(mono,147) + jnp.take(mono,148) + jnp.take(mono,149) + jnp.take(mono,150) + jnp.take(mono,151) + jnp.take(mono,152) + jnp.take(mono,153) + jnp.take(mono,154) + jnp.take(mono,155) + jnp.take(mono,156) + jnp.take(mono,157) + jnp.take(mono,158) 
    poly_33 = poly_2 * poly_10 - poly_31 
    poly_34 = poly_3 * poly_5 - poly_32 
    poly_35 = jnp.take(mono,159) + jnp.take(mono,160) + jnp.take(mono,161) + jnp.take(mono,162) + jnp.take(mono,163) + jnp.take(mono,164) + jnp.take(mono,165) + jnp.take(mono,166) + jnp.take(mono,167) + jnp.take(mono,168) + jnp.take(mono,169) + jnp.take(mono,170) + jnp.take(mono,171) + jnp.take(mono,172) + jnp.take(mono,173) + jnp.take(mono,174) + jnp.take(mono,175) + jnp.take(mono,176) + jnp.take(mono,177) + jnp.take(mono,178) + jnp.take(mono,179) + jnp.take(mono,180) + jnp.take(mono,181) + jnp.take(mono,182) 
    poly_36 = jnp.take(mono,183) + jnp.take(mono,184) + jnp.take(mono,185) + jnp.take(mono,186) + jnp.take(mono,187) + jnp.take(mono,188) + jnp.take(mono,189) + jnp.take(mono,190) + jnp.take(mono,191) + jnp.take(mono,192) + jnp.take(mono,193) + jnp.take(mono,194) 
    poly_37 = jnp.take(mono,195) + jnp.take(mono,196) + jnp.take(mono,197) + jnp.take(mono,198) + jnp.take(mono,199) + jnp.take(mono,200) + jnp.take(mono,201) + jnp.take(mono,202) + jnp.take(mono,203) + jnp.take(mono,204) + jnp.take(mono,205) + jnp.take(mono,206) + jnp.take(mono,207) + jnp.take(mono,208) + jnp.take(mono,209) + jnp.take(mono,210) + jnp.take(mono,211) + jnp.take(mono,212) + jnp.take(mono,213) + jnp.take(mono,214) + jnp.take(mono,215) + jnp.take(mono,216) + jnp.take(mono,217) + jnp.take(mono,218) 
    poly_38 = jnp.take(mono,219) + jnp.take(mono,220) + jnp.take(mono,221) 
    poly_39 = jnp.take(mono,222) + jnp.take(mono,223) + jnp.take(mono,224) + jnp.take(mono,225) 
    poly_40 = jnp.take(mono,226) + jnp.take(mono,227) + jnp.take(mono,228) + jnp.take(mono,229) + jnp.take(mono,230) + jnp.take(mono,231) + jnp.take(mono,232) + jnp.take(mono,233) + jnp.take(mono,234) + jnp.take(mono,235) + jnp.take(mono,236) + jnp.take(mono,237) 
    poly_41 = poly_3 * poly_7 - poly_36 - poly_40 - poly_35 
    poly_42 = poly_1 * poly_17 - poly_37 
    poly_43 = poly_1 * poly_18 - poly_39 
    poly_44 = jnp.take(mono,238) + jnp.take(mono,239) + jnp.take(mono,240) + jnp.take(mono,241) + jnp.take(mono,242) + jnp.take(mono,243) + jnp.take(mono,244) + jnp.take(mono,245) + jnp.take(mono,246) + jnp.take(mono,247) + jnp.take(mono,248) + jnp.take(mono,249) 
    poly_45 = jnp.take(mono,250) + jnp.take(mono,251) + jnp.take(mono,252) + jnp.take(mono,253) + jnp.take(mono,254) + jnp.take(mono,255) + jnp.take(mono,256) + jnp.take(mono,257) + jnp.take(mono,258) + jnp.take(mono,259) + jnp.take(mono,260) + jnp.take(mono,261) 
    poly_46 = poly_1 * poly_20 - poly_44 
    poly_47 = poly_1 * poly_10 - poly_30 - poly_30 - poly_30 - poly_30 
    poly_48 = poly_1 * poly_11 - poly_31 
    poly_49 = poly_3 * poly_4 - poly_33 - poly_31 - poly_48 - poly_31 
    poly_50 = poly_1 * poly_12 - poly_33 - poly_31 - poly_49 - poly_33 - poly_31 
    poly_51 = poly_5 * poly_8 
    poly_52 = poly_1 * poly_14 - poly_33 
    poly_53 = poly_1 * poly_15 - poly_40 - poly_35 
    poly_54 = poly_1 * poly_16 - poly_41 - poly_36 - poly_35 - poly_36 
    poly_55 = poly_1 * poly_19 - poly_41 - poly_40 
    poly_56 = poly_2 * poly_11 - poly_35 - poly_34 
    poly_57 = poly_4 * poly_5 - poly_37 
    poly_58 = poly_2 * poly_12 - poly_41 - poly_36 - poly_40 - poly_35 - poly_32 - poly_36 - poly_40 - poly_32 
    poly_59 = poly_1 * poly_25 - poly_57 
    poly_60 = poly_2 * poly_14 - poly_41 - poly_34 
    poly_61 = poly_2 * poly_15 - poly_39 - poly_44 - poly_37 - poly_39 - poly_39 
    poly_62 = poly_4 * poly_7 - poly_43 - poly_39 - poly_44 - poly_42 - poly_37 - poly_61 - poly_39 - poly_44 - poly_39 
    poly_63 = poly_5 * poly_7 - poly_45 
    poly_64 = poly_2 * poly_16 - poly_43 - poly_44 - poly_42 - poly_37 - poly_62 - poly_43 - poly_44 
    poly_65 = poly_2 * poly_17 - poly_45 - poly_38 - poly_63 - poly_45 - poly_38 - poly_38 - poly_38 
    poly_66 = poly_2 * poly_18 - poly_45 
    poly_67 = poly_1 * poly_27 - poly_64 - poly_62 - poly_61 
    poly_68 = poly_2 * poly_20 - poly_45 
    poly_69 = poly_3 * poly_3 - poly_30 - poly_47 - poly_30 - poly_47 - poly_30 - poly_30 - poly_30 - poly_30 
    poly_70 = poly_3 * poly_8 - poly_47 
    poly_71 = poly_1 * poly_22 - poly_49 - poly_48 
    poly_72 = poly_2 * poly_28 - poly_71 
    poly_73 = poly_1 * poly_24 - poly_58 - poly_56 - poly_56 
    poly_74 = poly_5 * poly_5 - poly_38 - poly_38 
    poly_75 = poly_8 * poly_9 - poly_73 
    poly_76 = poly_7 * poly_7 - poly_45 - poly_38 - poly_66 - poly_68 - poly_65 - poly_45 - poly_38 - poly_66 - poly_68 - poly_65 - poly_45 - poly_38 - poly_45 - poly_38 
    poly_77 = poly_2 * poly_24 - poly_62 - poly_61 - poly_57 
    poly_78 = poly_5 * poly_9 - poly_65 
    poly_79 = poly_1 * poly_29 - poly_77 
    poly_80 = poly_7 * poly_9 - poly_66 - poly_68 - poly_63 
    poly_81 = poly_1 * poly_28 - poly_70 
    poly_82 = poly_2 * poly_29 - poly_80 - poly_78 

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
    poly_81,    poly_82,    ]) 

    return poly 



