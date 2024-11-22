import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A4B.monomials_MOL_4_1_7 import f_monomials as f_monos 

# File created from ./MOL_4_1_7.POLY 

N_POLYS = 1101

# Total number of monomials = 1101 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(1101) 

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
    poly_208 = poly_30 * poly_5 
    poly_209 = poly_30 * poly_7 
    poly_210 = jnp.take(mono,496) + jnp.take(mono,497) + jnp.take(mono,498) + jnp.take(mono,499) + jnp.take(mono,500) + jnp.take(mono,501) + jnp.take(mono,502) + jnp.take(mono,503) + jnp.take(mono,504) + jnp.take(mono,505) + jnp.take(mono,506) + jnp.take(mono,507) + jnp.take(mono,508) + jnp.take(mono,509) + jnp.take(mono,510) + jnp.take(mono,511) + jnp.take(mono,512) + jnp.take(mono,513) + jnp.take(mono,514) + jnp.take(mono,515) + jnp.take(mono,516) + jnp.take(mono,517) + jnp.take(mono,518) + jnp.take(mono,519) 
    poly_211 = jnp.take(mono,520) + jnp.take(mono,521) + jnp.take(mono,522) + jnp.take(mono,523) + jnp.take(mono,524) + jnp.take(mono,525) 
    poly_212 = poly_10 * poly_17 - poly_210 
    poly_213 = poly_3 * poly_38 - poly_211 
    poly_214 = jnp.take(mono,526) + jnp.take(mono,527) + jnp.take(mono,528) + jnp.take(mono,529) + jnp.take(mono,530) + jnp.take(mono,531) + jnp.take(mono,532) + jnp.take(mono,533) + jnp.take(mono,534) + jnp.take(mono,535) + jnp.take(mono,536) + jnp.take(mono,537) 
    poly_215 = poly_10 * poly_18 - poly_214 
    poly_216 = jnp.take(mono,538) + jnp.take(mono,539) + jnp.take(mono,540) + jnp.take(mono,541) 
    poly_217 = jnp.take(mono,542) + jnp.take(mono,543) + jnp.take(mono,544) + jnp.take(mono,545) + jnp.take(mono,546) + jnp.take(mono,547) + jnp.take(mono,548) + jnp.take(mono,549) + jnp.take(mono,550) + jnp.take(mono,551) + jnp.take(mono,552) + jnp.take(mono,553) + jnp.take(mono,554) + jnp.take(mono,555) + jnp.take(mono,556) + jnp.take(mono,557) + jnp.take(mono,558) + jnp.take(mono,559) + jnp.take(mono,560) + jnp.take(mono,561) + jnp.take(mono,562) + jnp.take(mono,563) + jnp.take(mono,564) + jnp.take(mono,565) 
    poly_218 = jnp.take(mono,566) + jnp.take(mono,567) + jnp.take(mono,568) + jnp.take(mono,569) + jnp.take(mono,570) + jnp.take(mono,571) + jnp.take(mono,572) + jnp.take(mono,573) + jnp.take(mono,574) + jnp.take(mono,575) + jnp.take(mono,576) + jnp.take(mono,577) 
    poly_219 = jnp.take(mono,578) + jnp.take(mono,579) + jnp.take(mono,580) + jnp.take(mono,581) + jnp.take(mono,582) + jnp.take(mono,583) + jnp.take(mono,584) + jnp.take(mono,585) + jnp.take(mono,586) + jnp.take(mono,587) + jnp.take(mono,588) + jnp.take(mono,589) 
    poly_220 = jnp.take(mono,590) 
    poly_221 = poly_10 * poly_20 - poly_216 
    poly_222 = jnp.take(mono,591) + jnp.take(mono,592) + jnp.take(mono,593) + jnp.take(mono,594) + jnp.take(mono,595) + jnp.take(mono,596) + jnp.take(mono,597) + jnp.take(mono,598) + jnp.take(mono,599) + jnp.take(mono,600) + jnp.take(mono,601) + jnp.take(mono,602) 
    poly_223 = poly_3 * poly_45 - poly_218 - poly_222 - poly_217 
    poly_224 = poly_1 * poly_98 - poly_219 
    poly_225 = poly_30 * poly_4 
    poly_226 = jnp.take(mono,603) + jnp.take(mono,604) + jnp.take(mono,605) + jnp.take(mono,606) + jnp.take(mono,607) + jnp.take(mono,608) + jnp.take(mono,609) + jnp.take(mono,610) + jnp.take(mono,611) + jnp.take(mono,612) + jnp.take(mono,613) + jnp.take(mono,614) 
    poly_227 = poly_30 * poly_6 
    poly_228 = poly_5 * poly_47 - poly_226 
    poly_229 = jnp.take(mono,615) + jnp.take(mono,616) + jnp.take(mono,617) + jnp.take(mono,618) + jnp.take(mono,619) + jnp.take(mono,620) + jnp.take(mono,621) + jnp.take(mono,622) + jnp.take(mono,623) + jnp.take(mono,624) + jnp.take(mono,625) + jnp.take(mono,626) 
    poly_230 = poly_1 * poly_85 - poly_209 - poly_229 
    poly_231 = poly_1 * poly_86 - poly_210 
    poly_232 = poly_10 * poly_15 - poly_209 - poly_229 
    poly_233 = jnp.take(mono,627) + jnp.take(mono,628) + jnp.take(mono,629) + jnp.take(mono,630) + jnp.take(mono,631) + jnp.take(mono,632) + jnp.take(mono,633) + jnp.take(mono,634) + jnp.take(mono,635) + jnp.take(mono,636) + jnp.take(mono,637) + jnp.take(mono,638) + jnp.take(mono,639) + jnp.take(mono,640) + jnp.take(mono,641) + jnp.take(mono,642) + jnp.take(mono,643) + jnp.take(mono,644) + jnp.take(mono,645) + jnp.take(mono,646) + jnp.take(mono,647) + jnp.take(mono,648) + jnp.take(mono,649) + jnp.take(mono,650) 
    poly_234 = poly_10 * poly_16 - poly_209 - poly_233 - poly_230 - poly_209 
    poly_235 = jnp.take(mono,651) + jnp.take(mono,652) + jnp.take(mono,653) + jnp.take(mono,654) + jnp.take(mono,655) + jnp.take(mono,656) + jnp.take(mono,657) + jnp.take(mono,658) + jnp.take(mono,659) + jnp.take(mono,660) + jnp.take(mono,661) + jnp.take(mono,662) + jnp.take(mono,663) + jnp.take(mono,664) + jnp.take(mono,665) + jnp.take(mono,666) + jnp.take(mono,667) + jnp.take(mono,668) + jnp.take(mono,669) + jnp.take(mono,670) + jnp.take(mono,671) + jnp.take(mono,672) + jnp.take(mono,673) + jnp.take(mono,674) 
    poly_236 = poly_3 * poly_37 - poly_212 - poly_210 - poly_235 - poly_231 - poly_210 
    poly_237 = poly_3 * poly_39 - poly_214 
    poly_238 = poly_1 * poly_87 - poly_209 - poly_233 - poly_232 - poly_209 
    poly_239 = poly_1 * poly_88 - poly_209 - poly_234 
    poly_240 = poly_1 * poly_89 - poly_212 - poly_210 - poly_235 
    poly_241 = poly_1 * poly_90 - poly_212 - poly_210 - poly_236 
    poly_242 = poly_8 * poly_38 
    poly_243 = poly_1 * poly_92 - poly_212 
    poly_244 = poly_1 * poly_93 - poly_214 - poly_237 - poly_214 
    poly_245 = poly_18 * poly_21 - poly_244 - poly_237 
    poly_246 = poly_1 * poly_95 - poly_221 - poly_216 - poly_216 - poly_216 
    poly_247 = poly_1 * poly_96 - poly_222 - poly_217 
    poly_248 = poly_1 * poly_97 - poly_223 - poly_218 - poly_217 - poly_218 
    poly_249 = poly_3 * poly_44 - poly_221 - poly_216 - poly_246 - poly_221 - poly_216 - poly_216 
    poly_250 = poly_3 * poly_46 - poly_221 
    poly_251 = poly_1 * poly_100 - poly_223 - poly_222 
    poly_252 = poly_5 * poly_31 - poly_210 
    poly_253 = poly_30 * poly_9 
    poly_254 = poly_5 * poly_33 - poly_212 
    poly_255 = jnp.take(mono,675) + jnp.take(mono,676) + jnp.take(mono,677) + jnp.take(mono,678) + jnp.take(mono,679) + jnp.take(mono,680) + jnp.take(mono,681) + jnp.take(mono,682) + jnp.take(mono,683) + jnp.take(mono,684) + jnp.take(mono,685) + jnp.take(mono,686) + jnp.take(mono,687) + jnp.take(mono,688) + jnp.take(mono,689) + jnp.take(mono,690) + jnp.take(mono,691) + jnp.take(mono,692) + jnp.take(mono,693) + jnp.take(mono,694) + jnp.take(mono,695) + jnp.take(mono,696) + jnp.take(mono,697) + jnp.take(mono,698) 
    poly_256 = poly_2 * poly_85 - poly_214 - poly_216 - poly_210 - poly_216 - poly_216 
    poly_257 = poly_5 * poly_35 - poly_217 - poly_255 
    poly_258 = poly_5 * poly_36 - poly_218 
    poly_259 = poly_2 * poly_86 - poly_217 - poly_213 - poly_257 
    poly_260 = poly_4 * poly_38 - poly_219 
    poly_261 = poly_11 * poly_18 - poly_218 
    poly_262 = poly_5 * poly_39 
    poly_263 = jnp.take(mono,699) + jnp.take(mono,700) + jnp.take(mono,701) + jnp.take(mono,702) + jnp.take(mono,703) + jnp.take(mono,704) + jnp.take(mono,705) + jnp.take(mono,706) + jnp.take(mono,707) + jnp.take(mono,708) + jnp.take(mono,709) + jnp.take(mono,710) 
    poly_264 = poly_11 * poly_19 - poly_221 - poly_244 - poly_241 
    poly_265 = poly_5 * poly_40 - poly_222 
    poly_266 = poly_7 * poly_32 - poly_223 - poly_217 - poly_258 - poly_265 - poly_257 
    poly_267 = poly_14 * poly_15 - poly_214 - poly_249 - poly_236 
    poly_268 = poly_10 * poly_27 - poly_267 - poly_264 - poly_256 
    poly_269 = poly_5 * poly_41 - poly_223 - poly_266 
    poly_270 = poly_2 * poly_89 - poly_223 - poly_217 - poly_211 - poly_258 - poly_265 - poly_211 - poly_211 - poly_211 
    poly_271 = poly_2 * poly_90 - poly_218 - poly_222 - poly_213 - poly_269 - poly_255 - poly_218 - poly_222 - poly_213 
    poly_272 = poly_1 * poly_133 - poly_260 
    poly_273 = poly_2 * poly_92 - poly_223 - poly_213 - poly_266 
    poly_274 = poly_6 * poly_39 - poly_222 - poly_247 
    poly_275 = poly_12 * poly_18 - poly_223 - poly_217 - poly_274 
    poly_276 = poly_5 * poly_43 - poly_263 
    poly_277 = poly_14 * poly_18 - poly_222 
    poly_278 = poly_11 * poly_20 - poly_222 
    poly_279 = jnp.take(mono,711) + jnp.take(mono,712) + jnp.take(mono,713) + jnp.take(mono,714) + jnp.take(mono,715) + jnp.take(mono,716) + jnp.take(mono,717) + jnp.take(mono,718) + jnp.take(mono,719) + jnp.take(mono,720) + jnp.take(mono,721) + jnp.take(mono,722) + jnp.take(mono,723) + jnp.take(mono,724) + jnp.take(mono,725) + jnp.take(mono,726) + jnp.take(mono,727) + jnp.take(mono,728) + jnp.take(mono,729) + jnp.take(mono,730) + jnp.take(mono,731) + jnp.take(mono,732) + jnp.take(mono,733) + jnp.take(mono,734) 
    poly_280 = poly_2 * poly_95 - poly_218 - poly_217 - poly_278 
    poly_281 = poly_5 * poly_44 - poly_279 
    poly_282 = poly_2 * poly_96 - poly_219 - poly_262 - poly_281 - poly_219 
    poly_283 = poly_4 * poly_45 - poly_224 - poly_219 - poly_263 - poly_282 - poly_262 - poly_279 - poly_224 - poly_219 
    poly_284 = jnp.take(mono,735) + jnp.take(mono,736) + jnp.take(mono,737) + jnp.take(mono,738) + jnp.take(mono,739) + jnp.take(mono,740) + jnp.take(mono,741) + jnp.take(mono,742) + jnp.take(mono,743) + jnp.take(mono,744) + jnp.take(mono,745) + jnp.take(mono,746) + jnp.take(mono,747) + jnp.take(mono,748) + jnp.take(mono,749) + jnp.take(mono,750) + jnp.take(mono,751) + jnp.take(mono,752) + jnp.take(mono,753) + jnp.take(mono,754) + jnp.take(mono,755) + jnp.take(mono,756) + jnp.take(mono,757) + jnp.take(mono,758) 
    poly_285 = poly_2 * poly_97 - poly_224 - poly_219 - poly_283 - poly_276 - poly_279 - poly_224 - poly_219 
    poly_286 = poly_2 * poly_98 - poly_220 - poly_284 - poly_220 - poly_220 - poly_220 - poly_220 - poly_220 
    poly_287 = poly_4 * poly_46 - poly_222 - poly_251 
    poly_288 = poly_5 * poly_46 
    poly_289 = poly_14 * poly_20 - poly_218 
    poly_290 = poly_1 * poly_146 - poly_285 - poly_283 - poly_282 
    poly_291 = poly_30 * poly_3 
    poly_292 = poly_30 * poly_8 
    poly_293 = poly_10 * poly_11 - poly_225 
    poly_294 = jnp.take(mono,759) + jnp.take(mono,760) + jnp.take(mono,761) + jnp.take(mono,762) + jnp.take(mono,763) + jnp.take(mono,764) + jnp.take(mono,765) + jnp.take(mono,766) + jnp.take(mono,767) + jnp.take(mono,768) + jnp.take(mono,769) + jnp.take(mono,770) + jnp.take(mono,771) + jnp.take(mono,772) + jnp.take(mono,773) + jnp.take(mono,774) + jnp.take(mono,775) + jnp.take(mono,776) + jnp.take(mono,777) + jnp.take(mono,778) + jnp.take(mono,779) + jnp.take(mono,780) + jnp.take(mono,781) + jnp.take(mono,782) 
    poly_295 = poly_10 * poly_22 - poly_225 - poly_294 
    poly_296 = poly_3 * poly_31 - poly_227 - poly_225 - poly_293 - poly_225 
    poly_297 = poly_3 * poly_32 - poly_208 - poly_226 - poly_228 - poly_208 - poly_226 - poly_208 - poly_208 
    poly_298 = poly_1 * poly_103 - poly_225 - poly_295 
    poly_299 = poly_10 * poly_14 - poly_227 
    poly_300 = poly_5 * poly_69 - poly_297 
    poly_301 = poly_1 * poly_104 - poly_227 - poly_296 
    poly_302 = poly_8 * poly_32 - poly_228 
    poly_303 = poly_8 * poly_33 - poly_225 - poly_295 
    poly_304 = poly_5 * poly_70 - poly_302 
    poly_305 = jnp.take(mono,783) + jnp.take(mono,784) + jnp.take(mono,785) + jnp.take(mono,786) + jnp.take(mono,787) + jnp.take(mono,788) + jnp.take(mono,789) + jnp.take(mono,790) + jnp.take(mono,791) + jnp.take(mono,792) + jnp.take(mono,793) + jnp.take(mono,794) + jnp.take(mono,795) + jnp.take(mono,796) + jnp.take(mono,797) + jnp.take(mono,798) + jnp.take(mono,799) + jnp.take(mono,800) + jnp.take(mono,801) + jnp.take(mono,802) + jnp.take(mono,803) + jnp.take(mono,804) + jnp.take(mono,805) + jnp.take(mono,806) 
    poly_306 = poly_1 * poly_108 - poly_232 - poly_229 - poly_305 - poly_229 
    poly_307 = poly_3 * poly_36 - poly_209 - poly_234 - poly_230 
    poly_308 = poly_1 * poly_109 - poly_233 - poly_230 - poly_306 
    poly_309 = poly_8 * poly_36 - poly_239 - poly_229 
    poly_310 = poly_1 * poly_111 - poly_235 - poly_236 - poly_231 
    poly_311 = poly_1 * poly_112 - poly_237 
    poly_312 = poly_3 * poly_53 - poly_232 - poly_229 - poly_305 
    poly_313 = poly_3 * poly_54 - poly_234 - poly_233 - poly_230 - poly_309 - poly_308 
    poly_314 = poly_1 * poly_113 - poly_232 - poly_312 
    poly_315 = poly_1 * poly_114 - poly_234 - poly_233 - poly_313 
    poly_316 = poly_1 * poly_115 - poly_238 - poly_314 
    poly_317 = poly_3 * poly_55 - poly_239 - poly_238 - poly_316 
    poly_318 = poly_17 * poly_28 - poly_310 
    poly_319 = poly_18 * poly_28 - poly_311 
    poly_320 = poly_1 * poly_119 - poly_249 - poly_246 
    poly_321 = poly_1 * poly_120 - poly_250 
    poly_322 = poly_1 * poly_121 - poly_252 
    poly_323 = poly_2 * poly_102 - poly_233 - poly_230 - poly_232 - poly_229 - poly_228 - poly_229 
    poly_324 = poly_5 * poly_49 - poly_235 - poly_236 
    poly_325 = poly_5 * poly_32 - poly_211 - poly_213 - poly_211 
    poly_326 = poly_2 * poly_103 - poly_234 - poly_232 - poly_226 
    poly_327 = poly_3 * poly_74 - poly_325 
    poly_328 = poly_1 * poly_122 - poly_253 - poly_323 - poly_253 
    poly_329 = poly_5 * poly_50 - poly_240 - poly_241 
    poly_330 = poly_1 * poly_124 - poly_253 - poly_326 - poly_253 
    poly_331 = poly_1 * poly_125 - poly_254 
    poly_332 = poly_11 * poly_15 - poly_216 - poly_210 - poly_237 - poly_216 - poly_216 
    poly_333 = poly_1 * poly_126 - poly_264 - poly_256 - poly_332 
    poly_334 = poly_5 * poly_53 - poly_247 
    poly_335 = poly_1 * poly_128 - poly_266 - poly_258 - poly_255 
    poly_336 = poly_2 * poly_108 - poly_237 - poly_246 - poly_236 - poly_231 - poly_332 - poly_237 
    poly_337 = poly_4 * poly_36 - poly_215 - poly_214 - poly_212 - poly_256 - poly_246 - poly_235 - poly_215 - poly_215 
    poly_338 = poly_5 * poly_37 - poly_219 - poly_260 - poly_219 
    poly_339 = poly_1 * poly_129 - poly_267 - poly_256 - poly_336 
    poly_340 = poly_1 * poly_130 - poly_268 - poly_256 - poly_337 
    poly_341 = poly_5 * poly_54 - poly_248 - poly_335 
    poly_342 = poly_1 * poly_132 - poly_270 - poly_271 - poly_259 - poly_259 
    poly_343 = poly_5 * poly_38 - poly_220 - poly_220 - poly_220 
    poly_344 = poly_2 * poly_112 - poly_247 
    poly_345 = poly_1 * poly_135 - poly_275 - poly_261 
    poly_346 = poly_2 * poly_113 - poly_237 - poly_249 - poly_235 
    poly_347 = poly_3 * poly_62 - poly_268 - poly_264 - poly_256 - poly_337 - poly_333 
    poly_348 = poly_1 * poly_170 - poly_338 
    poly_349 = poly_3 * poly_64 - poly_268 - poly_267 - poly_256 - poly_340 - poly_339 
    poly_350 = poly_1 * poly_137 - poly_267 - poly_264 - poly_346 
    poly_351 = poly_1 * poly_138 - poly_268 - poly_264 - poly_347 
    poly_352 = poly_5 * poly_55 - poly_251 
    poly_353 = poly_1 * poly_140 - poly_268 - poly_267 - poly_349 
    poly_354 = poly_8 * poly_65 - poly_342 
    poly_355 = poly_8 * poly_66 - poly_345 - poly_344 
    poly_356 = poly_20 * poly_22 - poly_251 - poly_247 
    poly_357 = poly_5 * poly_45 - poly_284 
    poly_358 = poly_1 * poly_145 - poly_289 - poly_280 
    poly_359 = poly_2 * poly_120 - poly_251 
    poly_360 = poly_15 * poly_16 - poly_223 - poly_218 - poly_217 - poly_213 - poly_274 - poly_261 - poly_280 - poly_270 - poly_247 - poly_356 - poly_342 - poly_218 - poly_213 - poly_261 - poly_247 
    poly_361 = poly_7 * poly_36 - poly_223 - poly_217 - poly_211 - poly_275 - poly_280 - poly_270 - poly_211 
    poly_362 = poly_15 * poly_17 - poly_219 - poly_282 - poly_262 - poly_279 - poly_260 - poly_219 - poly_262 
    poly_363 = poly_7 * poly_37 - poly_224 - poly_219 - poly_283 - poly_276 - poly_282 - poly_262 - poly_279 - poly_281 - poly_272 - poly_260 - poly_362 - poly_224 - poly_219 - poly_262 - poly_281 
    poly_364 = poly_7 * poly_38 - poly_284 
    poly_365 = poly_16 * poly_17 - poly_224 - poly_219 - poly_285 - poly_283 - poly_263 - poly_276 - poly_279 - poly_281 - poly_272 - poly_260 - poly_363 - poly_224 - poly_219 - poly_263 - poly_281 
    poly_366 = poly_7 * poly_39 - poly_219 - poly_282 
    poly_367 = poly_16 * poly_18 - poly_224 - poly_285 - poly_282 - poly_224 
    poly_368 = poly_17 * poly_18 - poly_284 
    poly_369 = poly_7 * poly_40 - poly_223 - poly_217 - poly_211 - poly_274 - poly_287 - poly_270 - poly_211 
    poly_370 = poly_3 * poly_76 - poly_361 - poly_369 - poly_360 
    poly_371 = poly_1 * poly_175 - poly_365 - poly_363 - poly_362 
    poly_372 = poly_1 * poly_176 - poly_367 - poly_366 
    poly_373 = poly_15 * poly_20 - poly_224 - poly_282 
    poly_374 = poly_16 * poly_20 - poly_219 - poly_290 - poly_283 - poly_219 
    poly_375 = poly_17 * poly_20 - poly_284 
    poly_376 = poly_18 * poly_20 - poly_220 - poly_220 - poly_220 - poly_220 
    poly_377 = poly_1 * poly_178 - poly_374 - poly_373 
    poly_378 = poly_5 * poly_56 - poly_259 
    poly_379 = poly_2 * poly_122 - poly_264 - poly_256 - poly_252 
    poly_380 = poly_5 * poly_58 - poly_270 - poly_271 
    poly_381 = poly_10 * poly_29 - poly_379 
    poly_382 = poly_5 * poly_60 - poly_273 
    poly_383 = poly_7 * poly_56 - poly_261 - poly_278 - poly_255 
    poly_384 = poly_5 * poly_61 - poly_282 - poly_362 
    poly_385 = poly_5 * poly_62 - poly_283 - poly_363 
    poly_386 = poly_2 * poly_129 - poly_274 - poly_280 - poly_271 - poly_257 - poly_360 
    poly_387 = poly_9 * poly_36 - poly_277 - poly_278 - poly_265 
    poly_388 = poly_5 * poly_64 - poly_285 - poly_365 
    poly_389 = poly_2 * poly_132 - poly_283 - poly_282 - poly_260 - poly_363 - poly_362 
    poly_390 = poly_9 * poly_38 - poly_286 
    poly_391 = poly_9 * poly_39 - poly_281 
    poly_392 = poly_18 * poly_24 - poly_279 - poly_391 
    poly_393 = poly_5 * poly_66 - poly_368 
    poly_394 = poly_9 * poly_40 - poly_261 - poly_289 - poly_258 
    poly_395 = poly_2 * poly_138 - poly_275 - poly_287 - poly_271 - poly_266 - poly_370 
    poly_396 = poly_5 * poly_67 - poly_290 - poly_371 
    poly_397 = poly_7 * poly_60 - poly_277 - poly_289 - poly_269 
    poly_398 = poly_1 * poly_188 - poly_389 
    poly_399 = poly_1 * poly_189 - poly_392 - poly_391 
    poly_400 = poly_20 * poly_24 - poly_263 - poly_262 
    poly_401 = poly_5 * poly_68 - poly_375 
    poly_402 = poly_2 * poly_145 - poly_285 - poly_281 - poly_374 
    poly_403 = poly_9 * poly_45 - poly_284 - poly_393 - poly_401 
    poly_404 = poly_9 * poly_46 - poly_263 
    poly_405 = poly_10 * poly_10 - poly_291 - poly_291 
    poly_406 = poly_1 * poly_148 - poly_291 - poly_405 - poly_291 - poly_405 - poly_405 
    poly_407 = poly_10 * poly_28 - poly_292 
    poly_408 = poly_1 * poly_150 - poly_293 
    poly_409 = poly_11 * poly_28 - poly_301 
    poly_410 = poly_3 * poly_71 - poly_295 - poly_294 - poly_409 
    poly_411 = poly_4 * poly_69 - poly_299 - poly_296 - poly_408 
    poly_412 = poly_1 * poly_153 - poly_298 - poly_296 - poly_411 
    poly_413 = poly_1 * poly_154 - poly_299 
    poly_414 = poly_12 * poly_28 - poly_303 - poly_294 - poly_410 
    poly_415 = poly_5 * poly_81 
    poly_416 = poly_14 * poly_28 - poly_295 
    poly_417 = poly_1 * poly_158 - poly_312 - poly_305 
    poly_418 = poly_1 * poly_159 - poly_313 - poly_309 - poly_308 
    poly_419 = poly_1 * poly_160 - poly_317 - poly_316 
    poly_420 = poly_2 * poly_150 - poly_306 - poly_300 
    poly_421 = poly_8 * poly_56 - poly_328 
    poly_422 = poly_5 * poly_71 - poly_310 
    poly_423 = poly_3 * poly_73 - poly_326 - poly_323 - poly_421 
    poly_424 = poly_1 * poly_163 - poly_326 - poly_323 - poly_423 - poly_326 
    poly_425 = poly_8 * poly_74 
    poly_426 = poly_2 * poly_154 - poly_315 - poly_300 
    poly_427 = poly_8 * poly_58 - poly_330 - poly_323 - poly_423 
    poly_428 = poly_5 * poly_72 - poly_318 
    poly_429 = poly_8 * poly_60 - poly_326 
    poly_430 = poly_1 * poly_168 - poly_346 - poly_336 - poly_332 
    poly_431 = poly_1 * poly_169 - poly_347 - poly_337 - poly_333 
    poly_432 = poly_1 * poly_171 - poly_349 - poly_340 - poly_339 
    poly_433 = poly_1 * poly_172 - poly_353 - poly_351 - poly_350 
    poly_434 = poly_1 * poly_173 - poly_369 - poly_360 
    poly_435 = poly_1 * poly_174 - poly_370 - poly_361 - poly_360 - poly_361 
    poly_436 = poly_5 * poly_65 - poly_286 - poly_390 - poly_286 
    poly_437 = poly_18 * poly_18 - poly_286 - poly_286 
    poly_438 = poly_1 * poly_177 - poly_370 - poly_369 
    poly_439 = poly_20 * poly_20 - poly_286 - poly_286 
    poly_440 = poly_1 * poly_179 - poly_379 
    poly_441 = poly_5 * poly_73 - poly_342 
    poly_442 = poly_4 * poly_74 - poly_338 
    poly_443 = poly_3 * poly_77 - poly_381 - poly_379 - poly_440 - poly_379 
    poly_444 = poly_1 * poly_199 - poly_442 
    poly_445 = poly_1 * poly_181 - poly_381 - poly_379 - poly_443 - poly_381 - poly_379 
    poly_446 = poly_5 * poly_75 - poly_354 
    poly_447 = poly_1 * poly_183 - poly_381 
    poly_448 = poly_1 * poly_184 - poly_394 - poly_386 - poly_383 
    poly_449 = poly_1 * poly_185 - poly_395 - poly_387 - poly_383 
    poly_450 = poly_7 * poly_74 - poly_357 
    poly_451 = poly_1 * poly_187 - poly_397 - poly_387 - poly_386 
    poly_452 = poly_1 * poly_190 - poly_397 - poly_395 - poly_394 
    poly_453 = poly_2 * poly_173 - poly_366 - poly_373 - poly_362 
    poly_454 = poly_4 * poly_76 - poly_372 - poly_366 - poly_374 - poly_371 - poly_363 - poly_453 
    poly_455 = poly_5 * poly_76 - poly_376 
    poly_456 = poly_2 * poly_174 - poly_367 - poly_374 - poly_365 - poly_363 - poly_454 
    poly_457 = poly_7 * poly_65 - poly_284 - poly_403 - poly_368 - poly_375 - poly_364 - poly_364 
    poly_458 = poly_18 * poly_27 - poly_284 - poly_403 - poly_375 
    poly_459 = poly_1 * poly_201 - poly_456 - poly_454 - poly_453 
    poly_460 = poly_20 * poly_27 - poly_284 - poly_403 - poly_368 
    poly_461 = poly_2 * poly_179 - poly_383 - poly_378 
    poly_462 = poly_5 * poly_77 - poly_389 
    poly_463 = poly_2 * poly_181 - poly_395 - poly_387 - poly_394 - poly_386 - poly_380 
    poly_464 = poly_1 * poly_203 - poly_462 
    poly_465 = poly_2 * poly_183 - poly_397 - poly_382 
    poly_466 = poly_15 * poly_29 - poly_391 - poly_402 - poly_388 
    poly_467 = poly_2 * poly_185 - poly_392 - poly_400 - poly_389 - poly_385 - poly_454 - poly_392 
    poly_468 = poly_5 * poly_80 - poly_403 - poly_457 
    poly_469 = poly_16 * poly_29 - poly_399 - poly_400 - poly_396 - poly_384 - poly_467 
    poly_470 = poly_2 * poly_188 - poly_403 - poly_390 - poly_457 
    poly_471 = poly_18 * poly_29 - poly_401 
    poly_472 = poly_1 * poly_205 - poly_469 - poly_467 - poly_466 
    poly_473 = poly_20 * poly_29 - poly_393 
    poly_474 = poly_3 * poly_69 - poly_291 - poly_406 
    poly_475 = poly_1 * poly_192 - poly_406 - poly_474 - poly_474 
    poly_476 = poly_3 * poly_81 - poly_407 
    poly_477 = poly_1 * poly_194 - poly_410 - poly_409 
    poly_478 = poly_2 * poly_206 - poly_477 
    poly_479 = poly_1 * poly_196 - poly_423 - poly_421 
    poly_480 = poly_9 * poly_81 - poly_479 
    poly_481 = poly_1 * poly_198 - poly_443 - poly_440 
    poly_482 = poly_5 * poly_74 - poly_343 
    poly_483 = poly_28 * poly_29 - poly_481 
    poly_484 = poly_7 * poly_76 - poly_368 - poly_375 - poly_364 - poly_458 - poly_460 - poly_457 
    poly_485 = poly_1 * poly_202 - poly_463 - poly_461 - poly_461 
    poly_486 = poly_5 * poly_78 - poly_390 
    poly_487 = poly_8 * poly_82 - poly_485 
    poly_488 = poly_2 * poly_201 - poly_458 - poly_460 - poly_457 - poly_455 - poly_484 - poly_484 
    poly_489 = poly_2 * poly_202 - poly_467 - poly_466 - poly_462 
    poly_490 = poly_5 * poly_82 - poly_470 
    poly_491 = poly_1 * poly_207 - poly_489 
    poly_492 = poly_7 * poly_82 - poly_471 - poly_473 - poly_468 
    poly_493 = poly_1 * poly_206 - poly_476 
    poly_494 = poly_2 * poly_207 - poly_492 - poly_490 
    poly_495 = poly_30 * poly_17 
    poly_496 = poly_10 * poly_38 
    poly_497 = poly_30 * poly_18 
    poly_498 = jnp.take(mono,807) + jnp.take(mono,808) + jnp.take(mono,809) + jnp.take(mono,810) + jnp.take(mono,811) + jnp.take(mono,812) + jnp.take(mono,813) + jnp.take(mono,814) + jnp.take(mono,815) + jnp.take(mono,816) + jnp.take(mono,817) + jnp.take(mono,818) 
    poly_499 = jnp.take(mono,819) + jnp.take(mono,820) + jnp.take(mono,821) + jnp.take(mono,822) + jnp.take(mono,823) + jnp.take(mono,824) 
    poly_500 = poly_30 * poly_20 
    poly_501 = jnp.take(mono,825) + jnp.take(mono,826) + jnp.take(mono,827) + jnp.take(mono,828) + jnp.take(mono,829) + jnp.take(mono,830) + jnp.take(mono,831) + jnp.take(mono,832) + jnp.take(mono,833) + jnp.take(mono,834) + jnp.take(mono,835) + jnp.take(mono,836) + jnp.take(mono,837) + jnp.take(mono,838) + jnp.take(mono,839) + jnp.take(mono,840) + jnp.take(mono,841) + jnp.take(mono,842) + jnp.take(mono,843) + jnp.take(mono,844) + jnp.take(mono,845) + jnp.take(mono,846) + jnp.take(mono,847) + jnp.take(mono,848) 
    poly_502 = poly_10 * poly_45 - poly_501 - poly_498 
    poly_503 = jnp.take(mono,849) + jnp.take(mono,850) + jnp.take(mono,851) + jnp.take(mono,852) + jnp.take(mono,853) + jnp.take(mono,854) + jnp.take(mono,855) + jnp.take(mono,856) + jnp.take(mono,857) + jnp.take(mono,858) + jnp.take(mono,859) + jnp.take(mono,860) + jnp.take(mono,861) + jnp.take(mono,862) + jnp.take(mono,863) + jnp.take(mono,864) + jnp.take(mono,865) + jnp.take(mono,866) + jnp.take(mono,867) + jnp.take(mono,868) + jnp.take(mono,869) + jnp.take(mono,870) + jnp.take(mono,871) + jnp.take(mono,872) 
    poly_504 = poly_220 * poly_1 
    poly_505 = poly_3 * poly_98 - poly_503 - poly_499 
    poly_506 = poly_30 * poly_13 
    poly_507 = poly_30 * poly_15 
    poly_508 = poly_30 * poly_16 
    poly_509 = jnp.take(mono,873) + jnp.take(mono,874) + jnp.take(mono,875) + jnp.take(mono,876) + jnp.take(mono,877) + jnp.take(mono,878) + jnp.take(mono,879) + jnp.take(mono,880) + jnp.take(mono,881) + jnp.take(mono,882) + jnp.take(mono,883) + jnp.take(mono,884) + jnp.take(mono,885) + jnp.take(mono,886) + jnp.take(mono,887) + jnp.take(mono,888) + jnp.take(mono,889) + jnp.take(mono,890) + jnp.take(mono,891) + jnp.take(mono,892) + jnp.take(mono,893) + jnp.take(mono,894) + jnp.take(mono,895) + jnp.take(mono,896) 
    poly_510 = jnp.take(mono,897) + jnp.take(mono,898) + jnp.take(mono,899) + jnp.take(mono,900) + jnp.take(mono,901) + jnp.take(mono,902) + jnp.take(mono,903) + jnp.take(mono,904) + jnp.take(mono,905) + jnp.take(mono,906) + jnp.take(mono,907) + jnp.take(mono,908) + jnp.take(mono,909) + jnp.take(mono,910) + jnp.take(mono,911) + jnp.take(mono,912) + jnp.take(mono,913) + jnp.take(mono,914) + jnp.take(mono,915) + jnp.take(mono,916) + jnp.take(mono,917) + jnp.take(mono,918) + jnp.take(mono,919) + jnp.take(mono,920) 
    poly_511 = poly_10 * poly_37 - poly_495 - poly_509 - poly_510 - poly_495 
    poly_512 = poly_10 * poly_39 - poly_497 
    poly_513 = poly_30 * poly_19 
    poly_514 = poly_1 * poly_210 - poly_495 - poly_509 - poly_510 - poly_495 
    poly_515 = poly_1 * poly_211 - poly_496 
    poly_516 = jnp.take(mono,921) + jnp.take(mono,922) + jnp.take(mono,923) + jnp.take(mono,924) + jnp.take(mono,925) + jnp.take(mono,926) + jnp.take(mono,927) + jnp.take(mono,928) + jnp.take(mono,929) + jnp.take(mono,930) + jnp.take(mono,931) + jnp.take(mono,932) + jnp.take(mono,933) + jnp.take(mono,934) + jnp.take(mono,935) + jnp.take(mono,936) + jnp.take(mono,937) + jnp.take(mono,938) + jnp.take(mono,939) + jnp.take(mono,940) + jnp.take(mono,941) + jnp.take(mono,942) + jnp.take(mono,943) + jnp.take(mono,944) 
    poly_517 = poly_10 * poly_42 - poly_495 - poly_516 - poly_514 - poly_495 
    poly_518 = poly_21 * poly_38 - poly_515 
    poly_519 = poly_1 * poly_214 - poly_497 - poly_512 - poly_497 - poly_497 
    poly_520 = poly_1 * poly_215 - poly_497 
    poly_521 = poly_1 * poly_216 - poly_500 
    poly_522 = jnp.take(mono,945) + jnp.take(mono,946) + jnp.take(mono,947) + jnp.take(mono,948) + jnp.take(mono,949) + jnp.take(mono,950) + jnp.take(mono,951) + jnp.take(mono,952) + jnp.take(mono,953) + jnp.take(mono,954) + jnp.take(mono,955) + jnp.take(mono,956) + jnp.take(mono,957) + jnp.take(mono,958) + jnp.take(mono,959) + jnp.take(mono,960) + jnp.take(mono,961) + jnp.take(mono,962) + jnp.take(mono,963) + jnp.take(mono,964) + jnp.take(mono,965) + jnp.take(mono,966) + jnp.take(mono,967) + jnp.take(mono,968) 
    poly_523 = poly_1 * poly_217 - poly_501 - poly_498 - poly_522 - poly_498 
    poly_524 = poly_1 * poly_218 - poly_502 - poly_498 
    poly_525 = poly_1 * poly_219 - poly_503 - poly_499 - poly_499 
    poly_526 = poly_10 * poly_44 - poly_500 - poly_521 - poly_500 - poly_500 
    poly_527 = poly_3 * poly_96 - poly_501 - poly_498 - poly_522 
    poly_528 = poly_3 * poly_97 - poly_502 - poly_501 - poly_498 - poly_524 - poly_523 - poly_502 - poly_498 
    poly_529 = poly_10 * poly_46 - poly_500 
    poly_530 = poly_1 * poly_222 - poly_501 - poly_527 
    poly_531 = poly_3 * poly_100 - poly_502 - poly_501 - poly_530 
    poly_532 = poly_8 * poly_98 - poly_525 
    poly_533 = poly_30 * poly_25 
    poly_534 = poly_5 * poly_85 - poly_498 
    poly_535 = poly_11 * poly_38 - poly_499 
    poly_536 = jnp.take(mono,969) + jnp.take(mono,970) + jnp.take(mono,971) + jnp.take(mono,972) + jnp.take(mono,973) + jnp.take(mono,974) + jnp.take(mono,975) + jnp.take(mono,976) + jnp.take(mono,977) + jnp.take(mono,978) + jnp.take(mono,979) + jnp.take(mono,980) 
    poly_537 = jnp.take(mono,981) + jnp.take(mono,982) + jnp.take(mono,983) + jnp.take(mono,984) + jnp.take(mono,985) + jnp.take(mono,986) + jnp.take(mono,987) + jnp.take(mono,988) + jnp.take(mono,989) + jnp.take(mono,990) + jnp.take(mono,991) + jnp.take(mono,992) + jnp.take(mono,993) + jnp.take(mono,994) + jnp.take(mono,995) + jnp.take(mono,996) + jnp.take(mono,997) + jnp.take(mono,998) + jnp.take(mono,999) + jnp.take(mono,1000) + jnp.take(mono,1001) + jnp.take(mono,1002) + jnp.take(mono,1003) + jnp.take(mono,1004) 
    poly_538 = poly_30 * poly_27 
    poly_539 = poly_5 * poly_87 - poly_501 - poly_537 
    poly_540 = poly_5 * poly_88 - poly_502 
    poly_541 = poly_2 * poly_210 - poly_501 - poly_498 - poly_496 - poly_534 - poly_539 - poly_498 - poly_496 
    poly_542 = poly_2 * poly_211 - poly_505 - poly_499 
    poly_543 = poly_12 * poly_38 - poly_503 - poly_542 
    poly_544 = poly_10 * poly_65 - poly_541 
    poly_545 = poly_14 * poly_38 - poly_505 
    poly_546 = poly_18 * poly_31 - poly_502 - poly_498 
    poly_547 = poly_5 * poly_93 - poly_536 
    poly_548 = poly_18 * poly_32 - poly_547 
    poly_549 = poly_14 * poly_39 - poly_527 
    poly_550 = poly_2 * poly_215 - poly_502 
    poly_551 = poly_5 * poly_94 - poly_548 
    poly_552 = jnp.take(mono,1005) + jnp.take(mono,1006) + jnp.take(mono,1007) + jnp.take(mono,1008) + jnp.take(mono,1009) + jnp.take(mono,1010) + jnp.take(mono,1011) + jnp.take(mono,1012) + jnp.take(mono,1013) + jnp.take(mono,1014) + jnp.take(mono,1015) + jnp.take(mono,1016) 
    poly_553 = poly_2 * poly_216 - poly_498 
    poly_554 = poly_5 * poly_95 - poly_552 
    poly_555 = poly_11 * poly_45 - poly_503 - poly_536 - poly_552 
    poly_556 = jnp.take(mono,1017) + jnp.take(mono,1018) + jnp.take(mono,1019) + jnp.take(mono,1020) + jnp.take(mono,1021) + jnp.take(mono,1022) + jnp.take(mono,1023) + jnp.take(mono,1024) + jnp.take(mono,1025) + jnp.take(mono,1026) + jnp.take(mono,1027) + jnp.take(mono,1028) + jnp.take(mono,1029) + jnp.take(mono,1030) + jnp.take(mono,1031) + jnp.take(mono,1032) + jnp.take(mono,1033) + jnp.take(mono,1034) + jnp.take(mono,1035) + jnp.take(mono,1036) + jnp.take(mono,1037) + jnp.take(mono,1038) + jnp.take(mono,1039) + jnp.take(mono,1040) 
    poly_557 = jnp.take(mono,1041) + jnp.take(mono,1042) + jnp.take(mono,1043) + jnp.take(mono,1044) + jnp.take(mono,1045) + jnp.take(mono,1046) + jnp.take(mono,1047) + jnp.take(mono,1048) + jnp.take(mono,1049) + jnp.take(mono,1050) + jnp.take(mono,1051) + jnp.take(mono,1052) + jnp.take(mono,1053) + jnp.take(mono,1054) + jnp.take(mono,1055) + jnp.take(mono,1056) + jnp.take(mono,1057) + jnp.take(mono,1058) + jnp.take(mono,1059) + jnp.take(mono,1060) + jnp.take(mono,1061) + jnp.take(mono,1062) + jnp.take(mono,1063) + jnp.take(mono,1064) 
    poly_558 = poly_2 * poly_217 - poly_503 - poly_499 - poly_555 - poly_547 - poly_554 - poly_499 - poly_499 - poly_499 
    poly_559 = poly_2 * poly_218 - poly_503 - poly_551 - poly_552 
    poly_560 = jnp.take(mono,1065) + jnp.take(mono,1066) + jnp.take(mono,1067) + jnp.take(mono,1068) + jnp.take(mono,1069) + jnp.take(mono,1070) + jnp.take(mono,1071) + jnp.take(mono,1072) + jnp.take(mono,1073) + jnp.take(mono,1074) + jnp.take(mono,1075) + jnp.take(mono,1076) + jnp.take(mono,1077) + jnp.take(mono,1078) + jnp.take(mono,1079) + jnp.take(mono,1080) + jnp.take(mono,1081) + jnp.take(mono,1082) + jnp.take(mono,1083) + jnp.take(mono,1084) + jnp.take(mono,1085) + jnp.take(mono,1086) + jnp.take(mono,1087) + jnp.take(mono,1088) 
    poly_561 = poly_2 * poly_219 - poly_504 - poly_560 - poly_556 - poly_504 - poly_504 
    poly_562 = poly_220 * poly_2 
    poly_563 = poly_11 * poly_46 - poly_530 
    poly_564 = poly_20 * poly_32 - poly_554 
    poly_565 = poly_10 * poly_68 - poly_563 - poly_553 
    poly_566 = poly_5 * poly_99 - poly_564 
    poly_567 = poly_2 * poly_222 - poly_503 - poly_536 - poly_566 
    poly_568 = poly_12 * poly_45 - poly_505 - poly_503 - poly_499 - poly_559 - poly_548 - poly_567 - poly_558 - poly_547 - poly_564 - poly_554 - poly_505 - poly_503 - poly_499 - poly_505 - poly_499 - poly_505 - poly_499 
    poly_569 = poly_1 * poly_284 - poly_557 - poly_560 - poly_556 
    poly_570 = poly_14 * poly_45 - poly_503 - poly_551 - poly_566 
    poly_571 = poly_1 * poly_286 - poly_561 
    poly_572 = poly_30 * poly_11 
    poly_573 = poly_30 * poly_22 
    poly_574 = poly_30 * poly_12 
    poly_575 = poly_10 * poly_32 - poly_506 - poly_506 
    poly_576 = poly_30 * poly_14 
    poly_577 = poly_5 * poly_148 - poly_575 
    poly_578 = poly_1 * poly_226 - poly_506 - poly_575 
    poly_579 = poly_30 * poly_23 
    poly_580 = poly_5 * poly_149 - poly_578 
    poly_581 = jnp.take(mono,1089) + jnp.take(mono,1090) + jnp.take(mono,1091) + jnp.take(mono,1092) + jnp.take(mono,1093) + jnp.take(mono,1094) + jnp.take(mono,1095) + jnp.take(mono,1096) + jnp.take(mono,1097) + jnp.take(mono,1098) + jnp.take(mono,1099) + jnp.take(mono,1100) 
    poly_582 = poly_1 * poly_229 - poly_507 - poly_581 
    poly_583 = poly_3 * poly_85 - poly_508 - poly_507 - poly_582 
    poly_584 = poly_8 * poly_85 - poly_513 - poly_581 
    poly_585 = poly_3 * poly_86 - poly_495 - poly_509 - poly_510 
    poly_586 = poly_8 * poly_86 - poly_514 
    poly_587 = poly_10 * poly_53 - poly_507 - poly_581 
    poly_588 = poly_10 * poly_35 - poly_508 - poly_507 - poly_582 - poly_507 
    poly_589 = poly_10 * poly_36 - poly_508 - poly_583 
    poly_590 = jnp.take(mono,1101) + jnp.take(mono,1102) + jnp.take(mono,1103) + jnp.take(mono,1104) + jnp.take(mono,1105) + jnp.take(mono,1106) + jnp.take(mono,1107) + jnp.take(mono,1108) + jnp.take(mono,1109) + jnp.take(mono,1110) + jnp.take(mono,1111) + jnp.take(mono,1112) + jnp.take(mono,1113) + jnp.take(mono,1114) + jnp.take(mono,1115) + jnp.take(mono,1116) + jnp.take(mono,1117) + jnp.take(mono,1118) + jnp.take(mono,1119) + jnp.take(mono,1120) + jnp.take(mono,1121) + jnp.take(mono,1122) + jnp.take(mono,1123) + jnp.take(mono,1124) 
    poly_591 = poly_10 * poly_54 - poly_508 - poly_590 - poly_584 
    poly_592 = jnp.take(mono,1125) + jnp.take(mono,1126) + jnp.take(mono,1127) + jnp.take(mono,1128) + jnp.take(mono,1129) + jnp.take(mono,1130) + jnp.take(mono,1131) + jnp.take(mono,1132) + jnp.take(mono,1133) + jnp.take(mono,1134) + jnp.take(mono,1135) + jnp.take(mono,1136) + jnp.take(mono,1137) + jnp.take(mono,1138) + jnp.take(mono,1139) + jnp.take(mono,1140) + jnp.take(mono,1141) + jnp.take(mono,1142) + jnp.take(mono,1143) + jnp.take(mono,1144) + jnp.take(mono,1145) + jnp.take(mono,1146) + jnp.take(mono,1147) + jnp.take(mono,1148) 
    poly_593 = poly_3 * poly_111 - poly_511 - poly_509 - poly_510 - poly_592 - poly_586 
    poly_594 = poly_3 * poly_112 - poly_512 
    poly_595 = poly_10 * poly_40 - poly_513 - poly_507 
    poly_596 = poly_1 * poly_233 - poly_508 - poly_590 - poly_588 
    poly_597 = poly_3 * poly_88 - poly_513 - poly_508 - poly_589 
    poly_598 = poly_1 * poly_235 - poly_511 - poly_509 - poly_592 
    poly_599 = poly_1 * poly_236 - poly_511 - poly_510 - poly_593 
    poly_600 = poly_3 * poly_92 - poly_495 - poly_516 - poly_517 
    poly_601 = poly_21 * poly_39 - poly_519 - poly_594 
    poly_602 = poly_18 * poly_69 - poly_601 
    poly_603 = poly_8 * poly_87 - poly_508 - poly_590 - poly_587 
    poly_604 = poly_1 * poly_239 - poly_513 - poly_597 
    poly_605 = poly_1 * poly_240 - poly_516 - poly_514 - poly_598 
    poly_606 = poly_1 * poly_241 - poly_517 - poly_514 - poly_599 
    poly_607 = poly_28 * poly_38 
    poly_608 = poly_8 * poly_92 - poly_511 
    poly_609 = poly_1 * poly_244 - poly_519 - poly_601 
    poly_610 = poly_8 * poly_94 - poly_520 - poly_512 
    poly_611 = poly_3 * poly_95 - poly_500 - poly_526 - poly_521 - poly_500 - poly_521 - poly_500 
    poly_612 = poly_8 * poly_95 - poly_529 - poly_521 
    poly_613 = poly_1 * poly_247 - poly_527 - poly_522 
    poly_614 = poly_1 * poly_248 - poly_528 - poly_524 - poly_523 
    poly_615 = poly_3 * poly_119 - poly_526 - poly_521 - poly_612 
    poly_616 = poly_20 * poly_69 - poly_611 
    poly_617 = poly_3 * poly_120 - poly_529 
    poly_618 = poly_1 * poly_251 - poly_531 - poly_530 
    poly_619 = poly_5 * poly_102 - poly_509 - poly_510 
    poly_620 = poly_30 * poly_24 
    poly_621 = poly_5 * poly_103 - poly_511 
    poly_622 = poly_10 * poly_74 
    poly_623 = poly_5 * poly_104 - poly_514 
    poly_624 = poly_30 * poly_26 
    poly_625 = poly_5 * poly_106 - poly_516 - poly_517 
    poly_626 = poly_15 * poly_34 - poly_498 - poly_539 - poly_527 
    poly_627 = poly_1 * poly_255 - poly_537 - poly_534 - poly_626 
    poly_628 = poly_2 * poly_229 - poly_512 - poly_521 - poly_510 
    poly_629 = poly_11 * poly_36 - poly_497 - poly_511 - poly_611 - poly_497 - poly_497 
    poly_630 = poly_5 * poly_108 - poly_522 - poly_626 
    poly_631 = jnp.take(mono,1149) + jnp.take(mono,1150) + jnp.take(mono,1151) + jnp.take(mono,1152) + jnp.take(mono,1153) + jnp.take(mono,1154) + jnp.take(mono,1155) + jnp.take(mono,1156) + jnp.take(mono,1157) + jnp.take(mono,1158) + jnp.take(mono,1159) + jnp.take(mono,1160) + jnp.take(mono,1161) + jnp.take(mono,1162) + jnp.take(mono,1163) + jnp.take(mono,1164) + jnp.take(mono,1165) + jnp.take(mono,1166) + jnp.take(mono,1167) + jnp.take(mono,1168) + jnp.take(mono,1169) + jnp.take(mono,1170) + jnp.take(mono,1171) + jnp.take(mono,1172) 
    poly_632 = poly_5 * poly_86 - poly_499 - poly_535 - poly_499 
    poly_633 = poly_1 * poly_256 - poly_538 - poly_629 - poly_628 
    poly_634 = poly_5 * poly_109 - poly_523 - poly_627 
    poly_635 = poly_5 * poly_110 - poly_524 - poly_631 
    poly_636 = poly_1 * poly_259 - poly_541 
    poly_637 = poly_22 * poly_38 - poly_525 
    poly_638 = poly_11 * poly_39 - poly_498 
    poly_639 = poly_1 * poly_261 - poly_546 - poly_638 
    poly_640 = poly_5 * poly_112 
    poly_641 = poly_1 * poly_263 - poly_548 - poly_536 
    poly_642 = poly_11 * poly_40 - poly_500 - poly_514 - poly_601 - poly_500 - poly_500 
    poly_643 = poly_11 * poly_41 - poly_538 - poly_519 - poly_526 - poly_517 - poly_599 
    poly_644 = poly_5 * poly_113 - poly_527 
    poly_645 = poly_3 * poly_128 - poly_540 - poly_537 - poly_534 - poly_631 - poly_627 
    poly_646 = poly_5 * poly_89 - poly_503 - poly_542 
    poly_647 = poly_5 * poly_92 - poly_505 - poly_545 - poly_505 
    poly_648 = poly_10 * poly_61 - poly_538 - poly_642 - poly_628 
    poly_649 = poly_10 * poly_62 - poly_538 - poly_643 - poly_629 
    poly_650 = poly_5 * poly_90 - poly_503 - poly_543 
    poly_651 = poly_2 * poly_233 - poly_519 - poly_526 - poly_517 - poly_510 - poly_643 
    poly_652 = poly_10 * poly_64 - poly_538 - poly_651 - poly_633 
    poly_653 = poly_5 * poly_114 - poly_528 - poly_645 
    poly_654 = poly_2 * poly_235 - poly_528 - poly_522 - poly_515 - poly_635 - poly_644 - poly_515 
    poly_655 = poly_3 * poly_132 - poly_544 - poly_541 - poly_654 - poly_636 - poly_541 
    poly_656 = poly_1 * poly_343 
    poly_657 = poly_6 * poly_112 - poly_527 - poly_613 
    poly_658 = poly_3 * poly_135 - poly_550 - poly_546 - poly_639 
    poly_659 = poly_1 * poly_264 - poly_538 - poly_643 - poly_642 
    poly_660 = poly_5 * poly_115 - poly_530 
    poly_661 = poly_19 * poly_32 - poly_501 - poly_531 - poly_540 - poly_539 - poly_660 
    poly_662 = poly_1 * poly_267 - poly_538 - poly_651 - poly_648 
    poly_663 = poly_1 * poly_268 - poly_538 - poly_652 - poly_649 
    poly_664 = poly_5 * poly_116 - poly_531 - poly_661 
    poly_665 = poly_1 * poly_270 - poly_544 - poly_541 - poly_654 
    poly_666 = poly_1 * poly_271 - poly_544 - poly_541 - poly_655 
    poly_667 = poly_8 * poly_133 - poly_637 
    poly_668 = poly_1 * poly_273 - poly_544 
    poly_669 = poly_23 * poly_39 - poly_530 - poly_613 
    poly_670 = poly_18 * poly_50 - poly_531 - poly_522 - poly_669 
    poly_671 = poly_5 * poly_118 - poly_641 
    poly_672 = poly_1 * poly_277 - poly_550 - poly_549 
    poly_673 = poly_1 * poly_278 - poly_563 - poly_553 
    poly_674 = poly_1 * poly_279 - poly_564 - poly_552 - poly_554 - poly_552 
    poly_675 = poly_5 * poly_97 - poly_557 - poly_560 
    poly_676 = poly_4 * poly_95 - poly_502 - poly_501 - poly_498 - poly_553 - poly_522 - poly_673 - poly_553 
    poly_677 = poly_5 * poly_96 - poly_556 
    poly_678 = poly_1 * poly_280 - poly_565 - poly_553 - poly_676 - poly_553 
    poly_679 = poly_5 * poly_119 - poly_674 
    poly_680 = poly_1 * poly_282 - poly_567 - poly_558 - poly_555 
    poly_681 = poly_1 * poly_283 - poly_568 - poly_559 - poly_555 
    poly_682 = poly_5 * poly_98 - poly_562 
    poly_683 = poly_1 * poly_285 - poly_570 - poly_559 - poly_558 
    poly_684 = poly_22 * poly_46 - poly_527 - poly_618 
    poly_685 = poly_5 * poly_100 - poly_569 
    poly_686 = poly_2 * poly_249 - poly_528 - poly_527 - poly_684 
    poly_687 = poly_4 * poly_120 - poly_530 - poly_618 
    poly_688 = poly_5 * poly_120 
    poly_689 = poly_14 * poly_46 - poly_502 
    poly_690 = poly_1 * poly_290 - poly_570 - poly_568 - poly_567 
    poly_691 = poly_15 * poly_36 - poly_502 - poly_496 - poly_546 - poly_522 - poly_676 - poly_654 - poly_502 
    poly_692 = poly_7 * poly_86 - poly_503 - poly_555 - poly_547 - poly_554 - poly_543 
    poly_693 = poly_15 * poly_38 - poly_556 
    poly_694 = poly_17 * poly_35 - poly_503 - poly_499 - poly_558 - poly_555 - poly_536 - poly_547 - poly_552 - poly_554 - poly_543 - poly_535 - poly_692 - poly_499 - poly_536 - poly_552 - poly_535 - poly_499 - poly_499 
    poly_695 = poly_17 * poly_36 - poly_503 - poly_559 - poly_548 - poly_554 - poly_542 
    poly_696 = poly_16 * poly_38 - poly_557 - poly_560 
    poly_697 = poly_16 * poly_39 - poly_503 - poly_558 - poly_680 
    poly_698 = poly_18 * poly_36 - poly_505 - poly_558 - poly_505 
    poly_699 = poly_17 * poly_39 - poly_556 
    poly_700 = poly_18 * poly_37 - poly_557 - poly_560 - poly_699 
    poly_701 = poly_18 * poly_38 
    poly_702 = jnp.take(mono,1173) + jnp.take(mono,1174) + jnp.take(mono,1175) + jnp.take(mono,1176) + jnp.take(mono,1177) + jnp.take(mono,1178) + jnp.take(mono,1179) + jnp.take(mono,1180) + jnp.take(mono,1181) + jnp.take(mono,1182) + jnp.take(mono,1183) + jnp.take(mono,1184) + jnp.take(mono,1185) + jnp.take(mono,1186) + jnp.take(mono,1187) + jnp.take(mono,1188) + jnp.take(mono,1189) + jnp.take(mono,1190) + jnp.take(mono,1191) + jnp.take(mono,1192) + jnp.take(mono,1193) + jnp.take(mono,1194) + jnp.take(mono,1195) + jnp.take(mono,1196) 
    poly_703 = poly_15 * poly_41 - poly_501 - poly_549 - poly_546 - poly_565 - poly_544 - poly_528 - poly_524 - poly_527 - poly_518 - poly_684 - poly_655 - poly_549 - poly_527 
    poly_704 = poly_10 * poly_76 - poly_703 - poly_691 
    poly_705 = poly_17 * poly_40 - poly_503 - poly_567 - poly_547 - poly_564 - poly_542 
    poly_706 = poly_19 * poly_37 - poly_503 - poly_568 - poly_567 - poly_547 - poly_564 - poly_566 - poly_532 - poly_543 - poly_671 - poly_705 - poly_667 - poly_566 - poly_532 
    poly_707 = poly_19 * poly_38 - poly_569 
    poly_708 = poly_3 * poly_175 - poly_695 - poly_706 - poly_694 - poly_705 - poly_692 
    poly_709 = poly_19 * poly_39 - poly_567 - poly_525 
    poly_710 = poly_3 * poly_176 - poly_698 - poly_709 - poly_697 
    poly_711 = poly_1 * poly_368 - poly_702 - poly_700 - poly_699 
    poly_712 = poly_20 * poly_35 - poly_503 - poly_567 - poly_555 
    poly_713 = poly_20 * poly_36 - poly_499 - poly_568 - poly_499 
    poly_714 = jnp.take(mono,1197) + jnp.take(mono,1198) + jnp.take(mono,1199) + jnp.take(mono,1200) + jnp.take(mono,1201) + jnp.take(mono,1202) + jnp.take(mono,1203) + jnp.take(mono,1204) + jnp.take(mono,1205) + jnp.take(mono,1206) + jnp.take(mono,1207) + jnp.take(mono,1208) + jnp.take(mono,1209) + jnp.take(mono,1210) + jnp.take(mono,1211) + jnp.take(mono,1212) + jnp.take(mono,1213) + jnp.take(mono,1214) + jnp.take(mono,1215) + jnp.take(mono,1216) + jnp.take(mono,1217) + jnp.take(mono,1218) + jnp.take(mono,1219) + jnp.take(mono,1220) 
    poly_715 = poly_20 * poly_37 - poly_569 - poly_556 - poly_714 
    poly_716 = poly_20 * poly_38 
    poly_717 = poly_17 * poly_44 - poly_557 - poly_560 - poly_556 - poly_714 - poly_715 
    poly_718 = poly_20 * poly_39 - poly_504 
    poly_719 = poly_18 * poly_44 - poly_504 - poly_718 - poly_504 - poly_504 
    poly_720 = poly_7 * poly_98 - poly_562 - poly_701 - poly_716 - poly_562 - poly_562 - poly_562 
    poly_721 = poly_15 * poly_46 - poly_567 - poly_532 
    poly_722 = poly_3 * poly_178 - poly_713 - poly_721 - poly_712 
    poly_723 = poly_17 * poly_46 - poly_569 
    poly_724 = poly_18 * poly_46 - poly_504 
    poly_725 = poly_5 * poly_122 - poly_541 
    poly_726 = poly_30 * poly_29 
    poly_727 = poly_5 * poly_124 - poly_544 
    poly_728 = poly_5 * poly_126 - poly_555 - poly_692 
    poly_729 = poly_9 * poly_85 - poly_549 - poly_553 - poly_539 
    poly_730 = poly_5 * poly_129 - poly_558 - poly_694 
    poly_731 = poly_5 * poly_130 - poly_559 - poly_695 
    poly_732 = poly_2 * poly_259 - poly_555 - poly_535 - poly_692 
    poly_733 = poly_24 * poly_38 - poly_561 
    poly_734 = poly_18 * poly_56 - poly_552 
    poly_735 = poly_5 * poly_134 - poly_699 
    poly_736 = poly_5 * poly_135 - poly_700 
    poly_737 = poly_19 * poly_56 - poly_563 - poly_639 - poly_627 
    poly_738 = poly_5 * poly_137 - poly_567 - poly_705 
    poly_739 = poly_5 * poly_138 - poly_568 - poly_706 
    poly_740 = poly_15 * poly_60 - poly_549 - poly_686 - poly_653 
    poly_741 = poly_9 * poly_88 - poly_550 - poly_563 - poly_537 
    poly_742 = poly_5 * poly_140 - poly_570 - poly_708 
    poly_743 = poly_2 * poly_270 - poly_568 - poly_558 - poly_542 - poly_695 - poly_705 
    poly_744 = poly_2 * poly_271 - poly_559 - poly_567 - poly_543 - poly_706 - poly_694 
    poly_745 = poly_1 * poly_390 - poly_733 
    poly_746 = poly_2 * poly_273 - poly_570 - poly_545 - poly_708 
    poly_747 = poly_26 * poly_39 - poly_566 - poly_679 
    poly_748 = poly_18 * poly_58 - poly_564 - poly_554 - poly_747 
    poly_749 = poly_1 * poly_393 - poly_736 - poly_735 
    poly_750 = poly_18 * poly_60 - poly_566 
    poly_751 = poly_20 * poly_56 - poly_536 
    poly_752 = poly_5 * poly_143 - poly_714 - poly_715 
    poly_753 = poly_9 * poly_95 - poly_551 - poly_547 - poly_751 
    poly_754 = poly_5 * poly_145 - poly_717 
    poly_755 = poly_9 * poly_96 - poly_560 - poly_735 - poly_754 
    poly_756 = poly_2 * poly_283 - poly_561 - poly_557 - poly_719 - poly_700 - poly_714 - poly_561 
    poly_757 = jnp.take(mono,1221) + jnp.take(mono,1222) + jnp.take(mono,1223) + jnp.take(mono,1224) + jnp.take(mono,1225) + jnp.take(mono,1226) + jnp.take(mono,1227) + jnp.take(mono,1228) + jnp.take(mono,1229) + jnp.take(mono,1230) + jnp.take(mono,1231) + jnp.take(mono,1232) + jnp.take(mono,1233) + jnp.take(mono,1234) + jnp.take(mono,1235) + jnp.take(mono,1236) + jnp.take(mono,1237) + jnp.take(mono,1238) + jnp.take(mono,1239) + jnp.take(mono,1240) + jnp.take(mono,1241) + jnp.take(mono,1242) + jnp.take(mono,1243) + jnp.take(mono,1244) 
    poly_758 = poly_9 * poly_97 - poly_569 - poly_556 - poly_756 - poly_749 - poly_752 
    poly_759 = poly_2 * poly_286 - poly_562 - poly_720 
    poly_760 = poly_24 * poly_46 - poly_536 - poly_641 
    poly_761 = poly_5 * poly_147 - poly_723 
    poly_762 = poly_20 * poly_60 - poly_551 
    poly_763 = poly_1 * poly_403 - poly_758 - poly_756 - poly_755 
    poly_764 = poly_30 * poly_10 
    poly_765 = poly_30 * poly_21 
    poly_766 = poly_30 * poly_28 
    poly_767 = poly_10 * poly_48 - poly_573 - poly_572 - poly_572 
    poly_768 = jnp.take(mono,1245) + jnp.take(mono,1246) + jnp.take(mono,1247) + jnp.take(mono,1248) + jnp.take(mono,1249) + jnp.take(mono,1250) + jnp.take(mono,1251) + jnp.take(mono,1252) + jnp.take(mono,1253) + jnp.take(mono,1254) + jnp.take(mono,1255) + jnp.take(mono,1256) + jnp.take(mono,1257) + jnp.take(mono,1258) + jnp.take(mono,1259) + jnp.take(mono,1260) + jnp.take(mono,1261) + jnp.take(mono,1262) + jnp.take(mono,1263) + jnp.take(mono,1264) + jnp.take(mono,1265) + jnp.take(mono,1266) + jnp.take(mono,1267) + jnp.take(mono,1268) 
    poly_769 = poly_10 * poly_71 - poly_573 - poly_768 
    poly_770 = poly_1 * poly_293 - poly_572 - poly_767 - poly_572 
    poly_771 = poly_1 * poly_294 - poly_573 - poly_768 - poly_767 - poly_573 
    poly_772 = poly_1 * poly_295 - poly_573 - poly_769 
    poly_773 = poly_2 * poly_405 - poly_770 
    poly_774 = poly_3 * poly_104 - poly_579 - poly_574 - poly_770 
    poly_775 = poly_1 * poly_297 - poly_575 
    poly_776 = poly_8 * poly_103 - poly_572 - poly_769 - poly_572 
    poly_777 = poly_1 * poly_299 - poly_576 - poly_773 - poly_576 
    poly_778 = poly_1 * poly_300 - poly_577 
    poly_779 = poly_1 * poly_301 - poly_579 - poly_774 
    poly_780 = poly_28 * poly_32 - poly_580 
    poly_781 = poly_10 * poly_72 - poly_579 - poly_779 
    poly_782 = poly_5 * poly_193 - poly_780 
    poly_783 = jnp.take(mono,1269) + jnp.take(mono,1270) + jnp.take(mono,1271) + jnp.take(mono,1272) + jnp.take(mono,1273) + jnp.take(mono,1274) + jnp.take(mono,1275) + jnp.take(mono,1276) + jnp.take(mono,1277) + jnp.take(mono,1278) + jnp.take(mono,1279) + jnp.take(mono,1280) + jnp.take(mono,1281) + jnp.take(mono,1282) + jnp.take(mono,1283) + jnp.take(mono,1284) + jnp.take(mono,1285) + jnp.take(mono,1286) + jnp.take(mono,1287) + jnp.take(mono,1288) + jnp.take(mono,1289) + jnp.take(mono,1290) + jnp.take(mono,1291) + jnp.take(mono,1292) 
    poly_784 = poly_1 * poly_305 - poly_587 - poly_581 - poly_783 - poly_581 
    poly_785 = poly_1 * poly_306 - poly_588 - poly_582 - poly_784 
    poly_786 = poly_1 * poly_307 - poly_589 - poly_583 
    poly_787 = poly_1 * poly_308 - poly_590 - poly_584 - poly_785 
    poly_788 = poly_28 * poly_36 - poly_604 - poly_581 
    poly_789 = poly_1 * poly_310 - poly_592 - poly_593 - poly_586 
    poly_790 = poly_1 * poly_311 - poly_594 
    poly_791 = poly_3 * poly_158 - poly_587 - poly_581 - poly_783 
    poly_792 = poly_3 * poly_159 - poly_591 - poly_590 - poly_584 - poly_788 - poly_787 
    poly_793 = poly_1 * poly_312 - poly_587 - poly_791 
    poly_794 = poly_1 * poly_313 - poly_591 - poly_590 - poly_792 
    poly_795 = poly_1 * poly_314 - poly_595 - poly_793 
    poly_796 = poly_1 * poly_315 - poly_597 - poly_596 - poly_794 
    poly_797 = poly_1 * poly_316 - poly_603 - poly_795 
    poly_798 = poly_3 * poly_160 - poly_604 - poly_603 - poly_797 
    poly_799 = poly_17 * poly_81 - poly_789 
    poly_800 = poly_18 * poly_81 - poly_790 
    poly_801 = poly_1 * poly_320 - poly_615 - poly_612 
    poly_802 = poly_1 * poly_321 - poly_617 
    poly_803 = poly_5 * poly_150 - poly_585 
    poly_804 = poly_5 * poly_151 - poly_586 
    poly_805 = poly_10 * poly_56 - poly_620 
    poly_806 = poly_2 * poly_294 - poly_590 - poly_584 - poly_587 - poly_581 - poly_580 - poly_581 
    poly_807 = poly_5 * poly_152 - poly_592 - poly_593 
    poly_808 = poly_10 * poly_73 - poly_620 - poly_806 
    poly_809 = poly_3 * poly_122 - poly_624 - poly_620 - poly_805 - poly_620 
    poly_810 = poly_5 * poly_153 - poly_598 - poly_599 
    poly_811 = poly_1 * poly_325 - poly_622 - poly_622 
    poly_812 = poly_1 * poly_326 - poly_620 - poly_808 
    poly_813 = poly_1 * poly_327 - poly_622 
    poly_814 = poly_10 * poly_60 - poly_624 
    poly_815 = poly_5 * poly_154 - poly_600 
    poly_816 = poly_1 * poly_328 - poly_624 - poly_809 
    poly_817 = poly_5 * poly_155 - poly_605 - poly_606 
    poly_818 = poly_8 * poly_124 - poly_620 - poly_808 
    poly_819 = poly_5 * poly_157 - poly_608 
    poly_820 = poly_11 * poly_53 - poly_521 - poly_509 - poly_594 
    poly_821 = poly_1 * poly_332 - poly_642 - poly_628 - poly_820 
    poly_822 = poly_1 * poly_333 - poly_643 - poly_629 - poly_821 
    poly_823 = poly_5 * poly_158 - poly_613 
    poly_824 = poly_1 * poly_335 - poly_645 - poly_631 - poly_627 
    poly_825 = poly_2 * poly_305 - poly_594 - poly_612 - poly_593 - poly_586 - poly_820 - poly_594 
    poly_826 = poly_22 * poly_36 - poly_520 - poly_512 - poly_517 - poly_628 - poly_612 - poly_592 
    poly_827 = poly_1 * poly_336 - poly_648 - poly_628 - poly_825 
    poly_828 = poly_1 * poly_337 - poly_649 - poly_629 - poly_826 
    poly_829 = poly_5 * poly_111 - poly_525 - poly_637 - poly_525 
    poly_830 = poly_1 * poly_339 - poly_651 - poly_633 - poly_827 
    poly_831 = poly_1 * poly_340 - poly_652 - poly_633 - poly_828 
    poly_832 = poly_5 * poly_159 - poly_614 - poly_824 
    poly_833 = poly_1 * poly_342 - poly_654 - poly_655 - poly_636 
    poly_834 = poly_2 * poly_311 - poly_613 
    poly_835 = poly_1 * poly_345 - poly_658 - poly_639 
    poly_836 = poly_2 * poly_312 - poly_594 - poly_615 - poly_592 
    poly_837 = poly_3 * poly_169 - poly_649 - poly_643 - poly_629 - poly_826 - poly_822 
    poly_838 = poly_3 * poly_171 - poly_652 - poly_651 - poly_633 - poly_831 - poly_830 
    poly_839 = poly_1 * poly_346 - poly_648 - poly_642 - poly_836 
    poly_840 = poly_1 * poly_347 - poly_649 - poly_643 - poly_837 
    poly_841 = poly_8 * poly_170 - poly_829 
    poly_842 = poly_1 * poly_349 - poly_652 - poly_651 - poly_838 
    poly_843 = poly_1 * poly_350 - poly_662 - poly_659 - poly_839 
    poly_844 = poly_1 * poly_351 - poly_663 - poly_659 - poly_840 
    poly_845 = poly_5 * poly_160 - poly_618 
    poly_846 = poly_1 * poly_353 - poly_663 - poly_662 - poly_842 
    poly_847 = poly_28 * poly_65 - poly_833 
    poly_848 = poly_18 * poly_72 - poly_618 - poly_613 
    poly_849 = poly_20 * poly_71 - poly_618 - poly_613 
    poly_850 = poly_1 * poly_358 - poly_686 - poly_678 
    poly_851 = poly_2 * poly_321 - poly_618 
    poly_852 = poly_7 * poly_108 - poly_524 - poly_523 - poly_527 - poly_522 - poly_518 - poly_657 - poly_638 - poly_676 - poly_655 - poly_673 - poly_636 - poly_527 - poly_638 
    poly_853 = poly_1 * poly_360 - poly_703 - poly_691 - poly_852 - poly_691 
    poly_854 = poly_1 * poly_361 - poly_704 - poly_691 
    poly_855 = poly_1 * poly_362 - poly_694 - poly_705 - poly_692 
    poly_856 = poly_1 * poly_363 - poly_695 - poly_706 - poly_692 
    poly_857 = poly_1 * poly_365 - poly_708 - poly_695 - poly_694 
    poly_858 = poly_5 * poly_132 - poly_561 - poly_733 - poly_561 
    poly_859 = poly_17 * poly_38 - poly_562 - poly_682 - poly_562 
    poly_860 = poly_1 * poly_366 - poly_709 - poly_697 
    poly_861 = poly_1 * poly_367 - poly_710 - poly_698 - poly_697 - poly_698 
    poly_862 = poly_18 * poly_39 - poly_561 
    poly_863 = poly_3 * poly_173 - poly_703 - poly_691 - poly_852 
    poly_864 = poly_14 * poly_62 - poly_563 - poly_541 - poly_741 - poly_658 - poly_627 - poly_563 
    poly_865 = poly_1 * poly_436 - poly_858 
    poly_866 = poly_1 * poly_437 - poly_862 
    poly_867 = poly_1 * poly_369 - poly_703 - poly_863 
    poly_868 = poly_3 * poly_177 - poly_704 - poly_703 - poly_867 
    poly_869 = poly_1 * poly_371 - poly_708 - poly_706 - poly_705 
    poly_870 = poly_1 * poly_372 - poly_710 - poly_709 
    poly_871 = poly_1 * poly_373 - poly_721 - poly_712 
    poly_872 = poly_1 * poly_374 - poly_722 - poly_713 - poly_712 - poly_713 
    poly_873 = poly_5 * poly_146 - poly_720 - poly_757 
    poly_874 = poly_18 * poly_45 - poly_562 - poly_720 - poly_562 
    poly_875 = poly_1 * poly_377 - poly_722 - poly_721 
    poly_876 = poly_20 * poly_44 - poly_571 - poly_561 - poly_561 
    poly_877 = poly_20 * poly_45 - poly_562 - poly_720 - poly_562 
    poly_878 = poly_1 * poly_439 - poly_876 
    poly_879 = poly_1 * poly_378 - poly_725 
    poly_880 = poly_5 * poly_121 - poly_535 
    poly_881 = poly_2 * poly_323 - poly_643 - poly_629 - poly_642 - poly_628 - poly_619 
    poly_882 = poly_5 * poly_163 - poly_654 - poly_655 
    poly_883 = poly_5 * poly_123 - poly_542 - poly_543 
    poly_884 = poly_2 * poly_326 - poly_649 - poly_648 - poly_621 
    poly_885 = poly_5 * poly_125 - poly_545 
    poly_886 = poly_1 * poly_379 - poly_726 - poly_881 - poly_726 
    poly_887 = poly_5 * poly_165 - poly_665 - poly_666 
    poly_888 = poly_1 * poly_381 - poly_726 - poly_884 - poly_726 
    poly_889 = poly_1 * poly_382 - poly_727 
    poly_890 = poly_15 * poly_56 - poly_553 - poly_534 - poly_638 
    poly_891 = poly_1 * poly_383 - poly_737 - poly_729 - poly_890 
    poly_892 = poly_5 * poly_168 - poly_680 - poly_855 
    poly_893 = poly_5 * poly_169 - poly_681 - poly_856 
    poly_894 = poly_15 * poly_74 - poly_677 
    poly_895 = poly_5 * poly_128 - poly_557 - poly_696 
    poly_896 = poly_2 * poly_336 - poly_657 - poly_676 - poly_655 - poly_630 - poly_852 
    poly_897 = poly_2 * poly_337 - poly_658 - poly_676 - poly_654 - poly_631 - poly_854 
    poly_898 = poly_5 * poly_131 - poly_560 - poly_696 
    poly_899 = poly_1 * poly_386 - poly_740 - poly_729 - poly_896 
    poly_900 = poly_1 * poly_387 - poly_741 - poly_729 - poly_897 
    poly_901 = poly_5 * poly_171 - poly_683 - poly_857 
    poly_902 = poly_1 * poly_389 - poly_743 - poly_744 - poly_732 - poly_732 
    poly_903 = poly_25 * poly_38 - poly_562 
    poly_904 = poly_9 * poly_112 - poly_679 
    poly_905 = poly_1 * poly_392 - poly_748 - poly_734 
    poly_906 = poly_18 * poly_74 
    poly_907 = poly_9 * poly_113 - poly_638 - poly_686 - poly_635 
    poly_908 = poly_2 * poly_347 - poly_658 - poly_684 - poly_655 - poly_645 - poly_864 
    poly_909 = poly_19 * poly_74 - poly_685 
    poly_910 = poly_3 * poly_187 - poly_741 - poly_740 - poly_729 - poly_900 - poly_899 
    poly_911 = poly_1 * poly_394 - poly_740 - poly_737 - poly_907 
    poly_912 = poly_1 * poly_395 - poly_741 - poly_737 - poly_908 
    poly_913 = poly_5 * poly_172 - poly_690 - poly_869 
    poly_914 = poly_1 * poly_397 - poly_741 - poly_740 - poly_910 
    poly_915 = poly_8 * poly_188 - poly_902 
    poly_916 = poly_8 * poly_189 - poly_905 - poly_904 
    poly_917 = poly_20 * poly_73 - poly_641 - poly_640 
    poly_918 = poly_20 * poly_74 
    poly_919 = poly_1 * poly_402 - poly_762 - poly_753 
    poly_920 = poly_9 * poly_120 - poly_641 
    poly_921 = poly_11 * poly_76 - poly_709 - poly_713 - poly_706 
    poly_922 = poly_5 * poly_173 - poly_718 
    poly_923 = poly_7 * poly_128 - poly_557 - poly_556 - poly_702 - poly_736 - poly_717 - poly_685 - poly_675 - poly_752 - poly_733 - poly_656 - poly_865 - poly_736 - poly_685 
    poly_924 = poly_2 * poly_360 - poly_697 - poly_712 - poly_694 - poly_692 - poly_921 
    poly_925 = poly_2 * poly_361 - poly_698 - poly_713 - poly_695 
    poly_926 = poly_5 * poly_174 - poly_719 - poly_923 
    poly_927 = poly_15 * poly_65 - poly_560 - poly_755 - poly_699 - poly_717 - poly_696 
    poly_928 = poly_2 * poly_363 - poly_700 - poly_715 - poly_696 - poly_926 - poly_858 
    poly_929 = poly_27 * poly_38 - poly_720 - poly_757 
    poly_930 = poly_2 * poly_365 - poly_702 - poly_717 - poly_696 - poly_923 - poly_865 
    poly_931 = poly_18 * poly_61 - poly_556 - poly_756 - poly_714 
    poly_932 = poly_18 * poly_62 - poly_557 - poly_755 - poly_717 
    poly_933 = poly_5 * poly_176 - poly_874 
    poly_934 = poly_18 * poly_64 - poly_569 - poly_758 - poly_715 
    poly_935 = poly_18 * poly_65 - poly_716 - poly_716 
    poly_936 = poly_2 * poly_369 - poly_709 - poly_721 - poly_705 
    poly_937 = poly_4 * poly_177 - poly_709 - poly_722 - poly_706 - poly_870 - poly_936 - poly_869 
    poly_938 = poly_5 * poly_177 - poly_724 
    poly_939 = poly_14 * poly_76 - poly_698 - poly_721 - poly_694 
    poly_940 = poly_1 * poly_457 - poly_930 - poly_928 - poly_927 
    poly_941 = poly_1 * poly_458 - poly_934 - poly_932 - poly_931 
    poly_942 = poly_2 * poly_373 - poly_718 - poly_714 - poly_876 
    poly_943 = poly_20 * poly_62 - poly_556 - poly_756 - poly_702 
    poly_944 = poly_5 * poly_178 - poly_877 
    poly_945 = poly_20 * poly_64 - poly_560 - poly_763 - poly_700 
    poly_946 = poly_20 * poly_65 - poly_701 - poly_701 
    poly_947 = poly_2 * poly_376 - poly_720 - poly_874 - poly_877 
    poly_948 = poly_1 * poly_460 - poly_945 - poly_943 - poly_942 
    poly_949 = poly_5 * poly_179 - poly_732 
    poly_950 = poly_2 * poly_379 - poly_737 - poly_729 - poly_725 
    poly_951 = poly_5 * poly_181 - poly_743 - poly_744 
    poly_952 = poly_10 * poly_82 - poly_950 
    poly_953 = poly_5 * poly_183 - poly_746 
    poly_954 = poly_7 * poly_179 - poly_734 - poly_751 - poly_728 
    poly_955 = poly_5 * poly_184 - poly_755 - poly_927 
    poly_956 = poly_5 * poly_185 - poly_756 - poly_928 
    poly_957 = poly_2 * poly_386 - poly_747 - poly_753 - poly_744 - poly_730 - poly_924 
    poly_958 = poly_29 * poly_36 - poly_750 - poly_751 - poly_738 
    poly_959 = poly_5 * poly_187 - poly_758 - poly_930 
    poly_960 = poly_2 * poly_389 - poly_756 - poly_755 - poly_733 - poly_928 - poly_927 
    poly_961 = poly_29 * poly_38 - poly_759 
    poly_962 = poly_29 * poly_39 - poly_754 
    poly_963 = poly_18 * poly_77 - poly_752 - poly_962 
    poly_964 = poly_5 * poly_189 - poly_935 
    poly_965 = poly_29 * poly_40 - poly_734 - poly_762 - poly_731 
    poly_966 = poly_2 * poly_395 - poly_748 - poly_760 - poly_744 - poly_739 - poly_937 
    poly_967 = poly_5 * poly_190 - poly_763 - poly_940 
    poly_968 = poly_7 * poly_183 - poly_750 - poly_762 - poly_742 
    poly_969 = poly_1 * poly_470 - poly_960 
    poly_970 = poly_1 * poly_471 - poly_963 - poly_962 
    poly_971 = poly_20 * poly_77 - poly_736 - poly_735 
    poly_972 = poly_5 * poly_191 - poly_946 
    poly_973 = poly_2 * poly_402 - poly_758 - poly_754 - poly_945 
    poly_974 = poly_29 * poly_45 - poly_757 - poly_964 - poly_972 
    poly_975 = poly_29 * poly_46 - poly_736 
    poly_976 = poly_1 * poly_405 - poly_764 
    poly_977 = poly_10 * poly_69 - poly_765 
    poly_978 = poly_8 * poly_148 - poly_764 - poly_976 - poly_764 - poly_764 
    poly_979 = poly_10 * poly_81 - poly_766 
    poly_980 = poly_3 * poly_150 - poly_572 - poly_767 
    poly_981 = poly_8 * poly_150 - poly_770 
    poly_982 = poly_11 * poly_81 - poly_779 
    poly_983 = poly_3 * poly_194 - poly_769 - poly_768 - poly_982 
    poly_984 = poly_1 * poly_410 - poly_769 - poly_768 - poly_983 - poly_769 
    poly_985 = poly_1 * poly_411 - poly_772 - poly_771 - poly_984 
    poly_986 = poly_2 * poly_474 - poly_985 - poly_980 
    poly_987 = poly_1 * poly_412 - poly_776 - poly_774 - poly_985 
    poly_988 = poly_8 * poly_154 - poly_773 
    poly_989 = poly_8 * poly_155 - poly_777 - poly_774 - poly_985 
    poly_990 = poly_5 * poly_206 
    poly_991 = poly_14 * poly_81 - poly_769 
    poly_992 = poly_1 * poly_417 - poly_791 - poly_783 
    poly_993 = poly_1 * poly_418 - poly_792 - poly_788 - poly_787 
    poly_994 = poly_1 * poly_419 - poly_798 - poly_797 
    poly_995 = poly_1 * poly_420 - poly_805 
    poly_996 = poly_28 * poly_56 - poly_816 
    poly_997 = poly_5 * poly_194 - poly_789 
    poly_998 = poly_3 * poly_196 - poly_808 - poly_806 - poly_996 
    poly_999 = poly_24 * poly_69 - poly_814 - poly_809 - poly_995 
    poly_1000 = poly_1 * poly_424 - poly_812 - poly_809 - poly_999 
    poly_1001 = poly_28 * poly_74 
    poly_1002 = poly_1 * poly_426 - poly_814 
    poly_1003 = poly_28 * poly_58 - poly_818 - poly_806 - poly_998 
    poly_1004 = poly_5 * poly_195 - poly_799 
    poly_1005 = poly_28 * poly_60 - poly_808 
    poly_1006 = poly_1 * poly_430 - poly_836 - poly_825 - poly_820 
    poly_1007 = poly_1 * poly_431 - poly_837 - poly_826 - poly_822 
    poly_1008 = poly_1 * poly_432 - poly_838 - poly_831 - poly_830 
    poly_1009 = poly_1 * poly_433 - poly_846 - poly_844 - poly_843 
    poly_1010 = poly_1 * poly_434 - poly_863 - poly_852 
    poly_1011 = poly_1 * poly_435 - poly_864 - poly_854 - poly_853 
    poly_1012 = poly_1 * poly_438 - poly_868 - poly_867 
    poly_1013 = poly_2 * poly_420 - poly_821 - poly_803 
    poly_1014 = poly_8 * poly_179 - poly_886 
    poly_1015 = poly_5 * poly_196 - poly_833 
    poly_1016 = poly_5 * poly_162 - poly_637 
    poly_1017 = poly_3 * poly_198 - poly_884 - poly_881 - poly_1014 
    poly_1018 = poly_1 * poly_482 
    poly_1019 = poly_1 * poly_443 - poly_884 - poly_881 - poly_1017 - poly_884 
    poly_1020 = poly_5 * poly_166 - poly_667 
    poly_1021 = poly_2 * poly_426 - poly_842 - poly_815 
    poly_1022 = poly_8 * poly_181 - poly_888 - poly_881 - poly_1017 
    poly_1023 = poly_5 * poly_197 - poly_847 
    poly_1024 = poly_8 * poly_183 - poly_884 
    poly_1025 = poly_1 * poly_448 - poly_907 - poly_896 - poly_890 
    poly_1026 = poly_1 * poly_449 - poly_908 - poly_897 - poly_891 
    poly_1027 = poly_5 * poly_170 - poly_682 - poly_859 
    poly_1028 = poly_1 * poly_451 - poly_910 - poly_900 - poly_899 
    poly_1029 = poly_1 * poly_452 - poly_914 - poly_912 - poly_911 
    poly_1030 = poly_1 * poly_453 - poly_936 - poly_924 - poly_921 
    poly_1031 = poly_1 * poly_454 - poly_937 - poly_925 - poly_921 
    poly_1032 = poly_5 * poly_175 - poly_720 - poly_929 
    poly_1033 = poly_1 * poly_456 - poly_939 - poly_925 - poly_924 
    poly_1034 = poly_2 * poly_436 - poly_873 - poly_859 - poly_1032 
    poly_1035 = poly_2 * poly_437 - poly_874 
    poly_1036 = poly_1 * poly_459 - poly_939 - poly_937 - poly_936 
    poly_1037 = poly_2 * poly_439 - poly_877 
    poly_1038 = poly_7 * poly_173 - poly_699 - poly_714 - poly_693 - poly_931 - poly_942 - poly_927 
    poly_1039 = poly_7 * poly_174 - poly_702 - poly_700 - poly_717 - poly_715 - poly_696 - poly_934 - poly_932 - poly_945 - poly_943 - poly_930 - poly_928 
    poly_1040 = poly_17 * poly_76 - poly_720 - poly_947 - poly_933 - poly_944 - poly_929 
    poly_1041 = poly_18 * poly_76 - poly_701 - poly_946 
    poly_1042 = poly_1 * poly_484 - poly_1039 - poly_1038 
    poly_1043 = poly_20 * poly_76 - poly_716 - poly_935 
    poly_1044 = poly_1 * poly_461 - poly_950 
    poly_1045 = poly_5 * poly_198 - poly_902 
    poly_1046 = poly_5 * poly_180 - poly_733 
    poly_1047 = poly_3 * poly_202 - poly_952 - poly_950 - poly_1044 - poly_950 
    poly_1048 = poly_1 * poly_486 - poly_1046 
    poly_1049 = poly_1 * poly_463 - poly_952 - poly_950 - poly_1047 - poly_952 - poly_950 
    poly_1050 = poly_5 * poly_200 - poly_915 
    poly_1051 = poly_1 * poly_465 - poly_952 
    poly_1052 = poly_1 * poly_466 - poly_965 - poly_957 - poly_954 
    poly_1053 = poly_1 * poly_467 - poly_966 - poly_958 - poly_954 
    poly_1054 = poly_5 * poly_186 - poly_757 - poly_929 
    poly_1055 = poly_1 * poly_469 - poly_968 - poly_958 - poly_957 
    poly_1056 = poly_1 * poly_472 - poly_968 - poly_966 - poly_965 
    poly_1057 = poly_9 * poly_173 - poly_862 - poly_876 - poly_858 - poly_862 - poly_862 
    poly_1058 = poly_2 * poly_454 - poly_932 - poly_943 - poly_928 - poly_923 - poly_1039 
    poly_1059 = poly_5 * poly_201 - poly_947 - poly_1040 
    poly_1060 = poly_2 * poly_456 - poly_934 - poly_945 - poly_930 - poly_926 - poly_1039 
    poly_1061 = poly_7 * poly_188 - poly_757 - poly_974 - poly_935 - poly_946 - poly_929 
    poly_1062 = poly_9 * poly_176 - poly_877 - poly_873 - poly_1035 
    poly_1063 = poly_1 * poly_488 - poly_1060 - poly_1058 - poly_1057 
    poly_1064 = poly_9 * poly_178 - poly_874 - poly_873 - poly_1037 
    poly_1065 = poly_2 * poly_461 - poly_954 - poly_949 
    poly_1066 = poly_5 * poly_202 - poly_960 
    poly_1067 = poly_2 * poly_463 - poly_966 - poly_958 - poly_965 - poly_957 - poly_951 
    poly_1068 = poly_1 * poly_490 - poly_1066 
    poly_1069 = poly_2 * poly_465 - poly_968 - poly_953 
    poly_1070 = poly_15 * poly_82 - poly_962 - poly_973 - poly_959 
    poly_1071 = poly_9 * poly_185 - poly_932 - poly_942 - poly_927 - poly_895 - poly_1039 
    poly_1072 = poly_5 * poly_205 - poly_974 - poly_1061 
    poly_1073 = poly_9 * poly_187 - poly_941 - poly_945 - poly_940 - poly_898 - poly_1039 
    poly_1074 = poly_2 * poly_470 - poly_974 - poly_961 - poly_1061 
    poly_1075 = poly_18 * poly_82 - poly_972 
    poly_1076 = poly_1 * poly_492 - poly_1073 - poly_1071 - poly_1070 
    poly_1077 = poly_20 * poly_82 - poly_964 
    poly_1078 = poly_1 * poly_474 - poly_977 
    poly_1079 = poly_28 * poly_69 - poly_976 
    poly_1080 = poly_3 * poly_206 - poly_979 
    poly_1081 = poly_1 * poly_477 - poly_983 - poly_982 
    poly_1082 = poly_2 * poly_493 - poly_1081 
    poly_1083 = poly_1 * poly_479 - poly_998 - poly_996 
    poly_1084 = poly_9 * poly_206 - poly_1083 
    poly_1085 = poly_1 * poly_481 - poly_1017 - poly_1014 
    poly_1086 = poly_29 * poly_81 - poly_1085 
    poly_1087 = poly_1 * poly_485 - poly_1047 - poly_1044 
    poly_1088 = poly_2 * poly_482 - poly_1027 
    poly_1089 = poly_28 * poly_82 - poly_1087 
    poly_1090 = poly_2 * poly_484 - poly_1041 - poly_1043 - poly_1040 
    poly_1091 = poly_1 * poly_489 - poly_1067 - poly_1065 - poly_1065 
    poly_1092 = poly_5 * poly_203 - poly_961 
    poly_1093 = poly_8 * poly_207 - poly_1091 
    poly_1094 = poly_29 * poly_76 - poly_1035 - poly_1037 - poly_1032 
    poly_1095 = poly_2 * poly_489 - poly_1071 - poly_1070 - poly_1066 
    poly_1096 = poly_5 * poly_207 - poly_1074 
    poly_1097 = poly_1 * poly_494 - poly_1095 
    poly_1098 = poly_7 * poly_207 - poly_1075 - poly_1077 - poly_1072 
    poly_1099 = poly_1 * poly_493 - poly_1080 
    poly_1100 = poly_2 * poly_494 - poly_1098 - poly_1096 

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
    poly_886,    poly_887,    poly_888,    poly_889,    poly_890, 
    poly_891,    poly_892,    poly_893,    poly_894,    poly_895, 
    poly_896,    poly_897,    poly_898,    poly_899,    poly_900, 
    poly_901,    poly_902,    poly_903,    poly_904,    poly_905, 
    poly_906,    poly_907,    poly_908,    poly_909,    poly_910, 
    poly_911,    poly_912,    poly_913,    poly_914,    poly_915, 
    poly_916,    poly_917,    poly_918,    poly_919,    poly_920, 
    poly_921,    poly_922,    poly_923,    poly_924,    poly_925, 
    poly_926,    poly_927,    poly_928,    poly_929,    poly_930, 
    poly_931,    poly_932,    poly_933,    poly_934,    poly_935, 
    poly_936,    poly_937,    poly_938,    poly_939,    poly_940, 
    poly_941,    poly_942,    poly_943,    poly_944,    poly_945, 
    poly_946,    poly_947,    poly_948,    poly_949,    poly_950, 
    poly_951,    poly_952,    poly_953,    poly_954,    poly_955, 
    poly_956,    poly_957,    poly_958,    poly_959,    poly_960, 
    poly_961,    poly_962,    poly_963,    poly_964,    poly_965, 
    poly_966,    poly_967,    poly_968,    poly_969,    poly_970, 
    poly_971,    poly_972,    poly_973,    poly_974,    poly_975, 
    poly_976,    poly_977,    poly_978,    poly_979,    poly_980, 
    poly_981,    poly_982,    poly_983,    poly_984,    poly_985, 
    poly_986,    poly_987,    poly_988,    poly_989,    poly_990, 
    poly_991,    poly_992,    poly_993,    poly_994,    poly_995, 
    poly_996,    poly_997,    poly_998,    poly_999,    poly_1000, 
    poly_1001,    poly_1002,    poly_1003,    poly_1004,    poly_1005, 
    poly_1006,    poly_1007,    poly_1008,    poly_1009,    poly_1010, 
    poly_1011,    poly_1012,    poly_1013,    poly_1014,    poly_1015, 
    poly_1016,    poly_1017,    poly_1018,    poly_1019,    poly_1020, 
    poly_1021,    poly_1022,    poly_1023,    poly_1024,    poly_1025, 
    poly_1026,    poly_1027,    poly_1028,    poly_1029,    poly_1030, 
    poly_1031,    poly_1032,    poly_1033,    poly_1034,    poly_1035, 
    poly_1036,    poly_1037,    poly_1038,    poly_1039,    poly_1040, 
    poly_1041,    poly_1042,    poly_1043,    poly_1044,    poly_1045, 
    poly_1046,    poly_1047,    poly_1048,    poly_1049,    poly_1050, 
    poly_1051,    poly_1052,    poly_1053,    poly_1054,    poly_1055, 
    poly_1056,    poly_1057,    poly_1058,    poly_1059,    poly_1060, 
    poly_1061,    poly_1062,    poly_1063,    poly_1064,    poly_1065, 
    poly_1066,    poly_1067,    poly_1068,    poly_1069,    poly_1070, 
    poly_1071,    poly_1072,    poly_1073,    poly_1074,    poly_1075, 
    poly_1076,    poly_1077,    poly_1078,    poly_1079,    poly_1080, 
    poly_1081,    poly_1082,    poly_1083,    poly_1084,    poly_1085, 
    poly_1086,    poly_1087,    poly_1088,    poly_1089,    poly_1090, 
    poly_1091,    poly_1092,    poly_1093,    poly_1094,    poly_1095, 
    poly_1096,    poly_1097,    poly_1098,    poly_1099,    poly_1100, 
    ]) 

    return poly 



