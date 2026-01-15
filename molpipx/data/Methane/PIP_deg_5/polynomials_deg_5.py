import jax.numpy as jnp 
from jax import jit

from monomials_deg_5 import f_monomials as f_monos 

# File created from ./MOL_4_1_5.POLY 

N_POLYS = 208

# Total number of monomials = 208 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(208) 

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
    poly_83 = poly_30 * poly_2 
    poly_84 = poly_5 * poly_10 
    poly_85 = jnp.take(mono,262) + jnp.take(mono,263) + jnp.take(mono,264) + jnp.take(mono,265) + jnp.take(mono,266) + jnp.take(mono,267) + jnp.take(mono,268) + jnp.take(mono,269) + jnp.take(mono,270) + jnp.take(mono,271) + jnp.take(mono,272) + jnp.take(mono,273) 
    poly_86 = jnp.take(mono,274) + jnp.take(mono,275) + jnp.take(mono,276) + jnp.take(mono,277) + jnp.take(mono,278) + jnp.take(mono,279) + jnp.take(mono,280) + jnp.take(mono,281) + jnp.take(mono,282) + jnp.take(mono,283) + jnp.take(mono,284) + jnp.take(mono,285) 
    poly_87 = jnp.take(mono,286) + jnp.take(mono,287) + jnp.take(mono,288) + jnp.take(mono,289) + jnp.take(mono,290) + jnp.take(mono,291) + jnp.take(mono,292) + jnp.take(mono,293) + jnp.take(mono,294) + jnp.take(mono,295) + jnp.take(mono,296) + jnp.take(mono,297) + jnp.take(mono,298) + jnp.take(mono,299) + jnp.take(mono,300) + jnp.take(mono,301) + jnp.take(mono,302) + jnp.take(mono,303) + jnp.take(mono,304) + jnp.take(mono,305) + jnp.take(mono,306) + jnp.take(mono,307) + jnp.take(mono,308) + jnp.take(mono,309) 
    poly_88 = poly_7 * poly_10 - poly_87 - poly_85 
    poly_89 = jnp.take(mono,310) + jnp.take(mono,311) + jnp.take(mono,312) + jnp.take(mono,313) + jnp.take(mono,314) + jnp.take(mono,315) + jnp.take(mono,316) + jnp.take(mono,317) + jnp.take(mono,318) + jnp.take(mono,319) + jnp.take(mono,320) + jnp.take(mono,321) + jnp.take(mono,322) + jnp.take(mono,323) + jnp.take(mono,324) + jnp.take(mono,325) + jnp.take(mono,326) + jnp.take(mono,327) + jnp.take(mono,328) + jnp.take(mono,329) + jnp.take(mono,330) + jnp.take(mono,331) + jnp.take(mono,332) + jnp.take(mono,333) 
    poly_90 = jnp.take(mono,334) + jnp.take(mono,335) + jnp.take(mono,336) + jnp.take(mono,337) + jnp.take(mono,338) + jnp.take(mono,339) + jnp.take(mono,340) + jnp.take(mono,341) + jnp.take(mono,342) + jnp.take(mono,343) + jnp.take(mono,344) + jnp.take(mono,345) + jnp.take(mono,346) + jnp.take(mono,347) + jnp.take(mono,348) + jnp.take(mono,349) + jnp.take(mono,350) + jnp.take(mono,351) + jnp.take(mono,352) + jnp.take(mono,353) + jnp.take(mono,354) + jnp.take(mono,355) + jnp.take(mono,356) + jnp.take(mono,357) 
    poly_91 = poly_1 * poly_38 
    poly_92 = poly_3 * poly_17 - poly_89 - poly_90 - poly_86 
    poly_93 = jnp.take(mono,358) + jnp.take(mono,359) + jnp.take(mono,360) + jnp.take(mono,361) + jnp.take(mono,362) + jnp.take(mono,363) + jnp.take(mono,364) + jnp.take(mono,365) + jnp.take(mono,366) + jnp.take(mono,367) + jnp.take(mono,368) + jnp.take(mono,369) 
    poly_94 = poly_3 * poly_18 - poly_93 
    poly_95 = jnp.take(mono,370) + jnp.take(mono,371) + jnp.take(mono,372) + jnp.take(mono,373) + jnp.take(mono,374) + jnp.take(mono,375) + jnp.take(mono,376) + jnp.take(mono,377) + jnp.take(mono,378) + jnp.take(mono,379) + jnp.take(mono,380) + jnp.take(mono,381) 
    poly_96 = jnp.take(mono,382) + jnp.take(mono,383) + jnp.take(mono,384) + jnp.take(mono,385) + jnp.take(mono,386) + jnp.take(mono,387) + jnp.take(mono,388) + jnp.take(mono,389) + jnp.take(mono,390) + jnp.take(mono,391) + jnp.take(mono,392) + jnp.take(mono,393) 
    poly_97 = jnp.take(mono,394) + jnp.take(mono,395) + jnp.take(mono,396) + jnp.take(mono,397) + jnp.take(mono,398) + jnp.take(mono,399) + jnp.take(mono,400) + jnp.take(mono,401) + jnp.take(mono,402) + jnp.take(mono,403) + jnp.take(mono,404) + jnp.take(mono,405) + jnp.take(mono,406) + jnp.take(mono,407) + jnp.take(mono,408) + jnp.take(mono,409) + jnp.take(mono,410) + jnp.take(mono,411) + jnp.take(mono,412) + jnp.take(mono,413) + jnp.take(mono,414) + jnp.take(mono,415) + jnp.take(mono,416) + jnp.take(mono,417) 
    poly_98 = jnp.take(mono,418) + jnp.take(mono,419) + jnp.take(mono,420) + jnp.take(mono,421) + jnp.take(mono,422) + jnp.take(mono,423) 
    poly_99 = poly_3 * poly_20 - poly_95 
    poly_100 = poly_1 * poly_45 - poly_97 - poly_96 
    poly_101 = poly_30 * poly_1 
    poly_102 = jnp.take(mono,424) + jnp.take(mono,425) + jnp.take(mono,426) + jnp.take(mono,427) + jnp.take(mono,428) + jnp.take(mono,429) + jnp.take(mono,430) + jnp.take(mono,431) + jnp.take(mono,432) + jnp.take(mono,433) + jnp.take(mono,434) + jnp.take(mono,435) + jnp.take(mono,436) + jnp.take(mono,437) + jnp.take(mono,438) + jnp.take(mono,439) + jnp.take(mono,440) + jnp.take(mono,441) + jnp.take(mono,442) + jnp.take(mono,443) + jnp.take(mono,444) + jnp.take(mono,445) + jnp.take(mono,446) + jnp.take(mono,447) 
    poly_103 = poly_4 * poly_10 - poly_83 - poly_102 - poly_83 
    poly_104 = poly_1 * poly_31 - poly_83 - poly_102 - poly_83 
    poly_105 = poly_1 * poly_32 - poly_84 - poly_84 
    poly_106 = poly_1 * poly_33 - poly_83 - poly_103 - poly_83 
    poly_107 = poly_1 * poly_34 - poly_84 
    poly_108 = jnp.take(mono,448) + jnp.take(mono,449) + jnp.take(mono,450) + jnp.take(mono,451) + jnp.take(mono,452) + jnp.take(mono,453) + jnp.take(mono,454) + jnp.take(mono,455) + jnp.take(mono,456) + jnp.take(mono,457) + jnp.take(mono,458) + jnp.take(mono,459) + jnp.take(mono,460) + jnp.take(mono,461) + jnp.take(mono,462) + jnp.take(mono,463) + jnp.take(mono,464) + jnp.take(mono,465) + jnp.take(mono,466) + jnp.take(mono,467) + jnp.take(mono,468) + jnp.take(mono,469) + jnp.take(mono,470) + jnp.take(mono,471) 
    poly_109 = poly_1 * poly_35 - poly_87 - poly_85 - poly_108 - poly_85 
    poly_110 = poly_1 * poly_36 - poly_88 - poly_85 
    poly_111 = poly_1 * poly_37 - poly_89 - poly_90 - poly_86 - poly_86 
    poly_112 = poly_1 * poly_39 - poly_93 
    poly_113 = poly_3 * poly_15 - poly_87 - poly_85 - poly_108 
    poly_114 = poly_3 * poly_16 - poly_88 - poly_87 - poly_85 - poly_110 - poly_109 - poly_88 - poly_85 
    poly_115 = poly_1 * poly_40 - poly_87 - poly_113 
    poly_116 = poly_3 * poly_19 - poly_88 - poly_87 - poly_115 
    poly_117 = poly_8 * poly_17 - poly_111 
    poly_118 = poly_8 * poly_18 - poly_112 
    poly_119 = poly_1 * poly_44 - poly_99 - poly_95 - poly_95 
    poly_120 = poly_1 * poly_46 - poly_99 
    poly_121 = poly_5 * poly_11 - poly_86 
    poly_122 = poly_6 * poly_11 - poly_87 - poly_109 - poly_107 
    poly_123 = poly_5 * poly_12 - poly_89 - poly_90 
    poly_124 = poly_9 * poly_10 - poly_122 
    poly_125 = poly_5 * poly_14 - poly_92 
    poly_126 = poly_7 * poly_11 - poly_93 - poly_95 - poly_90 
    poly_127 = poly_5 * poly_15 - poly_96 
    poly_128 = jnp.take(mono,472) + jnp.take(mono,473) + jnp.take(mono,474) + jnp.take(mono,475) + jnp.take(mono,476) + jnp.take(mono,477) + jnp.take(mono,478) + jnp.take(mono,479) + jnp.take(mono,480) + jnp.take(mono,481) + jnp.take(mono,482) + jnp.take(mono,483) + jnp.take(mono,484) + jnp.take(mono,485) + jnp.take(mono,486) + jnp.take(mono,487) + jnp.take(mono,488) + jnp.take(mono,489) + jnp.take(mono,490) + jnp.take(mono,491) + jnp.take(mono,492) + jnp.take(mono,493) + jnp.take(mono,494) + jnp.take(mono,495) 
    poly_129 = poly_2 * poly_35 - poly_93 - poly_95 - poly_90 - poly_86 - poly_126 - poly_93 - poly_95 - poly_86 
    poly_130 = poly_2 * poly_36 - poly_94 - poly_95 - poly_89 
    poly_131 = poly_5 * poly_16 - poly_97 - poly_128 
    poly_132 = poly_2 * poly_37 - poly_97 - poly_96 - poly_91 - poly_131 - poly_127 - poly_96 - poly_91 
    poly_133 = poly_2 * poly_38 - poly_98 
    poly_134 = poly_2 * poly_39 - poly_96 
    poly_135 = poly_4 * poly_18 - poly_97 - poly_134 
    poly_136 = poly_5 * poly_18 
    poly_137 = poly_2 * poly_40 - poly_93 - poly_99 - poly_89 
    poly_138 = poly_4 * poly_19 - poly_93 - poly_99 - poly_90 - poly_118 - poly_137 - poly_117 - poly_99 
    poly_139 = poly_5 * poly_19 - poly_100 
    poly_140 = poly_7 * poly_14 - poly_94 - poly_99 - poly_90 
    poly_141 = poly_1 * poly_65 - poly_132 
    poly_142 = poly_1 * poly_66 - poly_135 - poly_134 
    poly_143 = poly_4 * poly_20 - poly_100 - poly_96 
    poly_144 = poly_5 * poly_20 
    poly_145 = poly_2 * poly_44 - poly_97 - poly_96 - poly_143 
    poly_146 = poly_2 * poly_45 - poly_98 - poly_136 - poly_144 - poly_98 - poly_98 - poly_98 
    poly_147 = poly_2 * poly_46 - poly_100 
    poly_148 = poly_3 * poly_10 - poly_101 - poly_101 - poly_101 
    poly_149 = poly_8 * poly_10 - poly_101 
    poly_150 = poly_3 * poly_11 - poly_83 - poly_102 
    poly_151 = poly_8 * poly_11 - poly_104 
    poly_152 = poly_3 * poly_22 - poly_103 - poly_102 - poly_151 
    poly_153 = poly_1 * poly_49 - poly_103 - poly_102 - poly_152 - poly_103 
    poly_154 = poly_2 * poly_69 - poly_153 - poly_150 
    poly_155 = poly_8 * poly_12 - poly_106 - poly_102 - poly_152 
    poly_156 = poly_5 * poly_28 
    poly_157 = poly_8 * poly_14 - poly_103 
    poly_158 = poly_1 * poly_53 - poly_113 - poly_108 
    poly_159 = poly_1 * poly_54 - poly_114 - poly_110 - poly_109 
    poly_160 = poly_1 * poly_55 - poly_116 - poly_115 
    poly_161 = poly_1 * poly_56 - poly_122 
    poly_162 = poly_5 * poly_22 - poly_111 
    poly_163 = poly_3 * poly_24 - poly_124 - poly_122 - poly_161 - poly_122 
    poly_164 = poly_1 * poly_74 
    poly_165 = poly_1 * poly_58 - poly_124 - poly_122 - poly_163 - poly_124 - poly_122 
    poly_166 = poly_5 * poly_23 - poly_117 
    poly_167 = poly_1 * poly_60 - poly_124 
    poly_168 = poly_1 * poly_61 - poly_137 - poly_129 - poly_126 
    poly_169 = poly_1 * poly_62 - poly_138 - poly_130 - poly_126 
    poly_170 = poly_5 * poly_17 - poly_98 - poly_133 - poly_98 
    poly_171 = poly_1 * poly_64 - poly_140 - poly_130 - poly_129 
    poly_172 = poly_1 * poly_67 - poly_140 - poly_138 - poly_137 
    poly_173 = poly_7 * poly_15 - poly_97 - poly_96 - poly_91 - poly_134 - poly_143 - poly_132 - poly_96 - poly_134 
    poly_174 = poly_7 * poly_16 - poly_100 - poly_97 - poly_96 - poly_91 - poly_142 - poly_135 - poly_145 - poly_143 - poly_141 - poly_132 - poly_100 - poly_97 - poly_96 - poly_91 - poly_135 - poly_145 
    poly_175 = poly_7 * poly_17 - poly_98 - poly_146 - poly_136 - poly_144 - poly_133 - poly_98 - poly_136 - poly_144 - poly_133 - poly_98 - poly_98 
    poly_176 = poly_7 * poly_18 - poly_98 - poly_146 - poly_98 
    poly_177 = poly_1 * poly_76 - poly_174 - poly_173 
    poly_178 = poly_7 * poly_20 - poly_98 - poly_146 - poly_98 
    poly_179 = poly_2 * poly_56 - poly_126 - poly_121 
    poly_180 = poly_5 * poly_24 - poly_132 
    poly_181 = poly_2 * poly_58 - poly_138 - poly_130 - poly_137 - poly_129 - poly_123 
    poly_182 = poly_1 * poly_78 - poly_180 
    poly_183 = poly_2 * poly_60 - poly_140 - poly_125 
    poly_184 = poly_9 * poly_15 - poly_134 - poly_145 - poly_131 
    poly_185 = poly_2 * poly_62 - poly_135 - poly_143 - poly_132 - poly_128 - poly_174 - poly_135 
    poly_186 = poly_5 * poly_27 - poly_146 - poly_175 
    poly_187 = poly_9 * poly_16 - poly_142 - poly_143 - poly_139 - poly_127 - poly_185 
    poly_188 = poly_2 * poly_65 - poly_146 - poly_133 - poly_175 
    poly_189 = poly_9 * poly_18 - poly_144 
    poly_190 = poly_1 * poly_80 - poly_187 - poly_185 - poly_184 
    poly_191 = poly_9 * poly_20 - poly_136 
    poly_192 = poly_1 * poly_69 - poly_148 
    poly_193 = poly_3 * poly_28 - poly_149 
    poly_194 = poly_1 * poly_71 - poly_152 - poly_151 
    poly_195 = poly_2 * poly_81 - poly_194 
    poly_196 = poly_1 * poly_73 - poly_163 - poly_161 
    poly_197 = poly_9 * poly_28 - poly_196 
    poly_198 = poly_1 * poly_77 - poly_181 - poly_179 - poly_179 
    poly_199 = poly_2 * poly_74 - poly_170 
    poly_200 = poly_8 * poly_29 - poly_198 
    poly_201 = poly_2 * poly_76 - poly_176 - poly_178 - poly_175 
    poly_202 = poly_2 * poly_77 - poly_185 - poly_184 - poly_180 
    poly_203 = poly_5 * poly_29 - poly_188 
    poly_204 = poly_1 * poly_82 - poly_202 
    poly_205 = poly_7 * poly_29 - poly_189 - poly_191 - poly_186 
    poly_206 = poly_1 * poly_81 - poly_193 
    poly_207 = poly_2 * poly_82 - poly_205 - poly_203 

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
    poly_206,    poly_207,    ]) 

    return poly 



