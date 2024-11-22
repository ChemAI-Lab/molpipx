import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A4B.monomials_MOL_4_1_8 import f_monomials as f_monos 

# File created from ./MOL_4_1_8.POLY 

N_POLYS = 2327

# Total number of monomials = 2327 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(2327) 

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
    poly_1101 = poly_30 * poly_38 
    poly_1102 = poly_30 * poly_45 
    poly_1103 = jnp.take(mono,1293) + jnp.take(mono,1294) + jnp.take(mono,1295) + jnp.take(mono,1296) + jnp.take(mono,1297) + jnp.take(mono,1298) + jnp.take(mono,1299) + jnp.take(mono,1300) + jnp.take(mono,1301) + jnp.take(mono,1302) + jnp.take(mono,1303) + jnp.take(mono,1304) 
    poly_1104 = poly_10 * poly_98 - poly_1103 
    poly_1105 = poly_220 * poly_3 
    poly_1106 = poly_30 * poly_37 
    poly_1107 = poly_30 * poly_39 
    poly_1108 = poly_30 * poly_42 
    poly_1109 = jnp.take(mono,1305) + jnp.take(mono,1306) + jnp.take(mono,1307) + jnp.take(mono,1308) + jnp.take(mono,1309) + jnp.take(mono,1310) + jnp.take(mono,1311) + jnp.take(mono,1312) + jnp.take(mono,1313) + jnp.take(mono,1314) + jnp.take(mono,1315) + jnp.take(mono,1316) + jnp.take(mono,1317) + jnp.take(mono,1318) + jnp.take(mono,1319) + jnp.take(mono,1320) + jnp.take(mono,1321) + jnp.take(mono,1322) + jnp.take(mono,1323) + jnp.take(mono,1324) + jnp.take(mono,1325) + jnp.take(mono,1326) + jnp.take(mono,1327) + jnp.take(mono,1328) 
    poly_1110 = poly_38 * poly_47 - poly_1109 
    poly_1111 = poly_30 * poly_43 
    poly_1112 = jnp.take(mono,1329) + jnp.take(mono,1330) + jnp.take(mono,1331) + jnp.take(mono,1332) + jnp.take(mono,1333) + jnp.take(mono,1334) + jnp.take(mono,1335) + jnp.take(mono,1336) + jnp.take(mono,1337) + jnp.take(mono,1338) + jnp.take(mono,1339) + jnp.take(mono,1340) 
    poly_1113 = poly_1 * poly_498 - poly_1102 - poly_1112 
    poly_1114 = poly_1 * poly_499 - poly_1103 
    poly_1115 = poly_30 * poly_44 
    poly_1116 = poly_10 * poly_96 - poly_1102 - poly_1112 
    poly_1117 = jnp.take(mono,1341) + jnp.take(mono,1342) + jnp.take(mono,1343) + jnp.take(mono,1344) + jnp.take(mono,1345) + jnp.take(mono,1346) + jnp.take(mono,1347) + jnp.take(mono,1348) + jnp.take(mono,1349) + jnp.take(mono,1350) + jnp.take(mono,1351) + jnp.take(mono,1352) + jnp.take(mono,1353) + jnp.take(mono,1354) + jnp.take(mono,1355) + jnp.take(mono,1356) + jnp.take(mono,1357) + jnp.take(mono,1358) + jnp.take(mono,1359) + jnp.take(mono,1360) + jnp.take(mono,1361) + jnp.take(mono,1362) + jnp.take(mono,1363) + jnp.take(mono,1364) 
    poly_1118 = poly_10 * poly_97 - poly_1102 - poly_1117 - poly_1113 - poly_1102 
    poly_1119 = poly_3 * poly_219 - poly_1104 - poly_1103 - poly_1114 - poly_1103 
    poly_1120 = poly_30 * poly_46 
    poly_1121 = poly_1 * poly_501 - poly_1102 - poly_1117 - poly_1116 - poly_1102 
    poly_1122 = poly_1 * poly_502 - poly_1102 - poly_1118 
    poly_1123 = poly_1 * poly_503 - poly_1104 - poly_1103 - poly_1119 - poly_1104 - poly_1103 
    poly_1124 = poly_220 * poly_8 
    poly_1125 = poly_1 * poly_505 - poly_1104 
    poly_1126 = poly_30 * poly_63 
    poly_1127 = poly_31 * poly_38 - poly_1103 
    poly_1128 = poly_30 * poly_65 
    poly_1129 = poly_10 * poly_133 - poly_1127 
    poly_1130 = jnp.take(mono,1365) + jnp.take(mono,1366) + jnp.take(mono,1367) + jnp.take(mono,1368) + jnp.take(mono,1369) + jnp.take(mono,1370) + jnp.take(mono,1371) + jnp.take(mono,1372) + jnp.take(mono,1373) + jnp.take(mono,1374) + jnp.take(mono,1375) + jnp.take(mono,1376) + jnp.take(mono,1377) + jnp.take(mono,1378) + jnp.take(mono,1379) + jnp.take(mono,1380) + jnp.take(mono,1381) + jnp.take(mono,1382) + jnp.take(mono,1383) + jnp.take(mono,1384) + jnp.take(mono,1385) + jnp.take(mono,1386) + jnp.take(mono,1387) + jnp.take(mono,1388) 
    poly_1131 = poly_30 * poly_66 
    poly_1132 = poly_5 * poly_214 - poly_1130 
    poly_1133 = poly_5 * poly_215 
    poly_1134 = poly_5 * poly_216 
    poly_1135 = jnp.take(mono,1389) + jnp.take(mono,1390) + jnp.take(mono,1391) + jnp.take(mono,1392) + jnp.take(mono,1393) + jnp.take(mono,1394) + jnp.take(mono,1395) + jnp.take(mono,1396) + jnp.take(mono,1397) + jnp.take(mono,1398) + jnp.take(mono,1399) + jnp.take(mono,1400) + jnp.take(mono,1401) + jnp.take(mono,1402) + jnp.take(mono,1403) + jnp.take(mono,1404) + jnp.take(mono,1405) + jnp.take(mono,1406) + jnp.take(mono,1407) + jnp.take(mono,1408) + jnp.take(mono,1409) + jnp.take(mono,1410) + jnp.take(mono,1411) + jnp.take(mono,1412) 
    poly_1136 = poly_2 * poly_498 - poly_1103 - poly_1132 - poly_1134 - poly_1103 
    poly_1137 = jnp.take(mono,1413) + jnp.take(mono,1414) + jnp.take(mono,1415) + jnp.take(mono,1416) + jnp.take(mono,1417) + jnp.take(mono,1418) + jnp.take(mono,1419) + jnp.take(mono,1420) + jnp.take(mono,1421) + jnp.take(mono,1422) + jnp.take(mono,1423) + jnp.take(mono,1424) + jnp.take(mono,1425) + jnp.take(mono,1426) + jnp.take(mono,1427) + jnp.take(mono,1428) + jnp.take(mono,1429) + jnp.take(mono,1430) + jnp.take(mono,1431) + jnp.take(mono,1432) + jnp.take(mono,1433) + jnp.take(mono,1434) + jnp.take(mono,1435) + jnp.take(mono,1436) 
    poly_1138 = jnp.take(mono,1437) + jnp.take(mono,1438) + jnp.take(mono,1439) + jnp.take(mono,1440) + jnp.take(mono,1441) + jnp.take(mono,1442) + jnp.take(mono,1443) + jnp.take(mono,1444) + jnp.take(mono,1445) + jnp.take(mono,1446) + jnp.take(mono,1447) + jnp.take(mono,1448) + jnp.take(mono,1449) + jnp.take(mono,1450) + jnp.take(mono,1451) + jnp.take(mono,1452) + jnp.take(mono,1453) + jnp.take(mono,1454) + jnp.take(mono,1455) + jnp.take(mono,1456) + jnp.take(mono,1457) + jnp.take(mono,1458) + jnp.take(mono,1459) + jnp.take(mono,1460) 
    poly_1139 = poly_2 * poly_499 - poly_1105 - poly_1137 
    poly_1140 = poly_220 * poly_4 
    poly_1141 = jnp.take(mono,1461) + jnp.take(mono,1462) + jnp.take(mono,1463) + jnp.take(mono,1464) + jnp.take(mono,1465) + jnp.take(mono,1466) + jnp.take(mono,1467) + jnp.take(mono,1468) + jnp.take(mono,1469) + jnp.take(mono,1470) + jnp.take(mono,1471) + jnp.take(mono,1472) 
    poly_1142 = poly_30 * poly_68 
    poly_1143 = poly_5 * poly_221 - poly_1141 
    poly_1144 = poly_31 * poly_45 - poly_1104 - poly_1103 - poly_1136 - poly_1130 - poly_1141 - poly_1134 - poly_1104 - poly_1103 
    poly_1145 = jnp.take(mono,1473) + jnp.take(mono,1474) + jnp.take(mono,1475) + jnp.take(mono,1476) + jnp.take(mono,1477) + jnp.take(mono,1478) + jnp.take(mono,1479) + jnp.take(mono,1480) + jnp.take(mono,1481) + jnp.take(mono,1482) + jnp.take(mono,1483) + jnp.take(mono,1484) + jnp.take(mono,1485) + jnp.take(mono,1486) + jnp.take(mono,1487) + jnp.take(mono,1488) + jnp.take(mono,1489) + jnp.take(mono,1490) + jnp.take(mono,1491) + jnp.take(mono,1492) + jnp.take(mono,1493) + jnp.take(mono,1494) + jnp.take(mono,1495) + jnp.take(mono,1496) 
    poly_1146 = jnp.take(mono,1497) + jnp.take(mono,1498) + jnp.take(mono,1499) + jnp.take(mono,1500) + jnp.take(mono,1501) + jnp.take(mono,1502) + jnp.take(mono,1503) + jnp.take(mono,1504) + jnp.take(mono,1505) + jnp.take(mono,1506) + jnp.take(mono,1507) + jnp.take(mono,1508) + jnp.take(mono,1509) + jnp.take(mono,1510) + jnp.take(mono,1511) + jnp.take(mono,1512) + jnp.take(mono,1513) + jnp.take(mono,1514) + jnp.take(mono,1515) + jnp.take(mono,1516) + jnp.take(mono,1517) + jnp.take(mono,1518) + jnp.take(mono,1519) + jnp.take(mono,1520) 
    poly_1147 = poly_2 * poly_501 - poly_1104 - poly_1103 - poly_1144 - poly_1130 - poly_1143 - poly_1104 - poly_1103 
    poly_1148 = poly_10 * poly_146 - poly_1147 - poly_1144 - poly_1136 
    poly_1149 = poly_3 * poly_284 - poly_1146 - poly_1138 - poly_1145 - poly_1135 - poly_1137 
    poly_1150 = poly_2 * poly_503 - poly_1105 - poly_1149 - poly_1138 - poly_1145 - poly_1135 - poly_1105 - poly_1105 - poly_1105 
    poly_1151 = poly_220 * poly_6 
    poly_1152 = poly_2 * poly_505 - poly_1105 - poly_1146 
    poly_1153 = poly_30 * poly_32 
    poly_1154 = poly_30 * poly_34 
    poly_1155 = poly_30 * poly_51 
    poly_1156 = poly_30 * poly_53 
    poly_1157 = poly_30 * poly_35 
    poly_1158 = poly_30 * poly_36 
    poly_1159 = poly_30 * poly_54 
    poly_1160 = jnp.take(mono,1521) + jnp.take(mono,1522) + jnp.take(mono,1523) + jnp.take(mono,1524) + jnp.take(mono,1525) + jnp.take(mono,1526) + jnp.take(mono,1527) + jnp.take(mono,1528) + jnp.take(mono,1529) + jnp.take(mono,1530) + jnp.take(mono,1531) + jnp.take(mono,1532) + jnp.take(mono,1533) + jnp.take(mono,1534) + jnp.take(mono,1535) + jnp.take(mono,1536) + jnp.take(mono,1537) + jnp.take(mono,1538) + jnp.take(mono,1539) + jnp.take(mono,1540) + jnp.take(mono,1541) + jnp.take(mono,1542) + jnp.take(mono,1543) + jnp.take(mono,1544) 
    poly_1161 = poly_10 * poly_86 - poly_1106 
    poly_1162 = jnp.take(mono,1545) + jnp.take(mono,1546) + jnp.take(mono,1547) + jnp.take(mono,1548) + jnp.take(mono,1549) + jnp.take(mono,1550) + jnp.take(mono,1551) + jnp.take(mono,1552) + jnp.take(mono,1553) + jnp.take(mono,1554) + jnp.take(mono,1555) + jnp.take(mono,1556) + jnp.take(mono,1557) + jnp.take(mono,1558) + jnp.take(mono,1559) + jnp.take(mono,1560) + jnp.take(mono,1561) + jnp.take(mono,1562) + jnp.take(mono,1563) + jnp.take(mono,1564) + jnp.take(mono,1565) + jnp.take(mono,1566) + jnp.take(mono,1567) + jnp.take(mono,1568) 
    poly_1163 = poly_10 * poly_111 - poly_1106 - poly_1160 - poly_1162 
    poly_1164 = poly_10 * poly_112 - poly_1107 
    poly_1165 = poly_30 * poly_40 
    poly_1166 = poly_30 * poly_41 
    poly_1167 = poly_1 * poly_509 - poly_1106 - poly_1160 - poly_1161 
    poly_1168 = poly_1 * poly_510 - poly_1106 - poly_1162 - poly_1161 
    poly_1169 = poly_3 * poly_211 - poly_1101 - poly_1109 - poly_1101 
    poly_1170 = poly_10 * poly_89 - poly_1108 - poly_1106 - poly_1167 
    poly_1171 = poly_10 * poly_90 - poly_1108 - poly_1106 - poly_1168 
    poly_1172 = poly_10 * poly_92 - poly_1108 
    poly_1173 = poly_38 * poly_69 - poly_1169 
    poly_1174 = poly_39 * poly_47 - poly_1111 - poly_1164 
    poly_1175 = poly_3 * poly_214 - poly_1111 - poly_1107 - poly_1174 - poly_1111 - poly_1107 - poly_1107 
    poly_1176 = poly_3 * poly_215 - poly_1111 
    poly_1177 = poly_30 * poly_55 
    poly_1178 = poly_1 * poly_514 - poly_1108 - poly_1167 - poly_1168 
    poly_1179 = poly_8 * poly_211 - poly_1110 
    poly_1180 = poly_1 * poly_516 - poly_1108 - poly_1172 - poly_1170 
    poly_1181 = poly_8 * poly_212 - poly_1106 - poly_1180 - poly_1163 
    poly_1182 = poly_8 * poly_213 - poly_1109 
    poly_1183 = poly_8 * poly_214 - poly_1111 - poly_1164 
    poly_1184 = poly_8 * poly_215 - poly_1107 
    poly_1185 = poly_3 * poly_216 - poly_1115 
    poly_1186 = poly_8 * poly_216 - poly_1120 
    poly_1187 = jnp.take(mono,1569) + jnp.take(mono,1570) + jnp.take(mono,1571) + jnp.take(mono,1572) + jnp.take(mono,1573) + jnp.take(mono,1574) + jnp.take(mono,1575) + jnp.take(mono,1576) + jnp.take(mono,1577) + jnp.take(mono,1578) + jnp.take(mono,1579) + jnp.take(mono,1580) + jnp.take(mono,1581) + jnp.take(mono,1582) + jnp.take(mono,1583) + jnp.take(mono,1584) + jnp.take(mono,1585) + jnp.take(mono,1586) + jnp.take(mono,1587) + jnp.take(mono,1588) + jnp.take(mono,1589) + jnp.take(mono,1590) + jnp.take(mono,1591) + jnp.take(mono,1592) 
    poly_1188 = poly_1 * poly_522 - poly_1116 - poly_1112 - poly_1187 - poly_1112 
    poly_1189 = poly_3 * poly_218 - poly_1102 - poly_1118 - poly_1113 
    poly_1190 = poly_1 * poly_523 - poly_1117 - poly_1113 - poly_1188 
    poly_1191 = poly_8 * poly_218 - poly_1122 - poly_1112 
    poly_1192 = poly_1 * poly_525 - poly_1119 - poly_1114 
    poly_1193 = poly_10 * poly_95 - poly_1115 - poly_1185 - poly_1115 
    poly_1194 = poly_10 * poly_119 - poly_1115 - poly_1186 
    poly_1195 = poly_3 * poly_247 - poly_1116 - poly_1112 - poly_1187 
    poly_1196 = poly_3 * poly_248 - poly_1118 - poly_1117 - poly_1113 - poly_1191 - poly_1190 
    poly_1197 = poly_20 * poly_148 - poly_1193 - poly_1185 
    poly_1198 = poly_1 * poly_527 - poly_1116 - poly_1195 
    poly_1199 = poly_1 * poly_528 - poly_1118 - poly_1117 - poly_1196 
    poly_1200 = poly_10 * poly_120 - poly_1120 
    poly_1201 = poly_1 * poly_530 - poly_1121 - poly_1198 
    poly_1202 = poly_3 * poly_251 - poly_1122 - poly_1121 - poly_1201 
    poly_1203 = poly_28 * poly_98 - poly_1192 
    poly_1204 = poly_30 * poly_57 
    poly_1205 = poly_30 * poly_74 
    poly_1206 = poly_30 * poly_59 
    poly_1207 = poly_5 * poly_229 - poly_1112 
    poly_1208 = jnp.take(mono,1593) + jnp.take(mono,1594) + jnp.take(mono,1595) + jnp.take(mono,1596) + jnp.take(mono,1597) + jnp.take(mono,1598) + jnp.take(mono,1599) + jnp.take(mono,1600) + jnp.take(mono,1601) + jnp.take(mono,1602) + jnp.take(mono,1603) + jnp.take(mono,1604) + jnp.take(mono,1605) + jnp.take(mono,1606) + jnp.take(mono,1607) + jnp.take(mono,1608) + jnp.take(mono,1609) + jnp.take(mono,1610) + jnp.take(mono,1611) + jnp.take(mono,1612) + jnp.take(mono,1613) + jnp.take(mono,1614) + jnp.take(mono,1615) + jnp.take(mono,1616) 
    poly_1209 = poly_5 * poly_230 - poly_1113 - poly_1208 
    poly_1210 = poly_1 * poly_535 - poly_1127 
    poly_1211 = poly_34 * poly_39 - poly_1132 
    poly_1212 = poly_1 * poly_536 - poly_1130 - poly_1211 
    poly_1213 = jnp.take(mono,1617) + jnp.take(mono,1618) + jnp.take(mono,1619) + jnp.take(mono,1620) + jnp.take(mono,1621) + jnp.take(mono,1622) + jnp.take(mono,1623) + jnp.take(mono,1624) + jnp.take(mono,1625) + jnp.take(mono,1626) + jnp.take(mono,1627) + jnp.take(mono,1628) + jnp.take(mono,1629) + jnp.take(mono,1630) + jnp.take(mono,1631) + jnp.take(mono,1632) + jnp.take(mono,1633) + jnp.take(mono,1634) + jnp.take(mono,1635) + jnp.take(mono,1636) + jnp.take(mono,1637) + jnp.take(mono,1638) + jnp.take(mono,1639) + jnp.take(mono,1640) 
    poly_1214 = poly_11 * poly_92 - poly_1128 - poly_1117 - poly_1173 
    poly_1215 = poly_30 * poly_61 
    poly_1216 = poly_30 * poly_62 
    poly_1217 = poly_5 * poly_232 - poly_1116 - poly_1213 
    poly_1218 = poly_10 * poly_128 - poly_1126 - poly_1214 - poly_1208 
    poly_1219 = poly_5 * poly_210 - poly_1103 - poly_1127 - poly_1103 
    poly_1220 = poly_10 * poly_170 - poly_1219 
    poly_1221 = poly_30 * poly_64 
    poly_1222 = poly_5 * poly_233 - poly_1117 - poly_1214 
    poly_1223 = poly_5 * poly_234 - poly_1118 - poly_1218 
    poly_1224 = poly_2 * poly_509 - poly_1117 - poly_1112 - poly_1109 - poly_1209 - poly_1217 - poly_1112 
    poly_1225 = poly_2 * poly_510 - poly_1113 - poly_1116 - poly_1110 - poly_1222 - poly_1207 - poly_1110 
    poly_1226 = poly_4 * poly_211 - poly_1104 - poly_1114 - poly_1127 
    poly_1227 = poly_38 * poly_49 - poly_1119 - poly_1226 
    poly_1228 = poly_5 * poly_211 - poly_1105 
    poly_1229 = poly_32 * poly_38 - poly_1105 - poly_1228 - poly_1105 
    poly_1230 = poly_10 * poly_132 - poly_1128 - poly_1224 - poly_1225 - poly_1128 
    poly_1231 = poly_34 * poly_38 - poly_1105 
    poly_1232 = poly_31 * poly_39 - poly_1102 - poly_1112 
    poly_1233 = poly_11 * poly_94 - poly_1131 - poly_1118 - poly_1189 
    poly_1234 = poly_5 * poly_237 - poly_1211 
    poly_1235 = poly_3 * poly_263 - poly_1133 - poly_1130 - poly_1212 
    poly_1236 = poly_14 * poly_112 - poly_1195 
    poly_1237 = poly_4 * poly_215 - poly_1131 - poly_1118 
    poly_1238 = poly_7 * poly_226 - poly_1122 - poly_1117 - poly_1112 - poly_1218 - poly_1209 - poly_1217 
    poly_1239 = poly_30 * poly_67 
    poly_1240 = poly_5 * poly_238 - poly_1121 - poly_1238 
    poly_1241 = poly_5 * poly_239 - poly_1122 
    poly_1242 = poly_1 * poly_541 - poly_1128 - poly_1224 - poly_1225 - poly_1128 
    poly_1243 = poly_1 * poly_542 - poly_1129 - poly_1127 - poly_1226 
    poly_1244 = poly_38 * poly_50 - poly_1123 - poly_1243 
    poly_1245 = poly_2 * poly_516 - poly_1122 - poly_1117 - poly_1109 - poly_1218 - poly_1238 - poly_1122 
    poly_1246 = poly_10 * poly_141 - poly_1128 - poly_1245 - poly_1242 - poly_1128 
    poly_1247 = poly_1 * poly_545 - poly_1129 
    poly_1248 = poly_18 * poly_104 - poly_1122 - poly_1112 
    poly_1249 = poly_5 * poly_244 - poly_1212 
    poly_1250 = poly_18 * poly_105 - poly_1235 - poly_1249 - poly_1234 
    poly_1251 = poly_1 * poly_549 - poly_1131 - poly_1236 
    poly_1252 = poly_1 * poly_550 - poly_1131 - poly_1237 
    poly_1253 = poly_1 * poly_551 - poly_1133 - poly_1132 
    poly_1254 = poly_1 * poly_552 - poly_1141 - poly_1134 
    poly_1255 = poly_5 * poly_218 - poly_1138 
    poly_1256 = poly_4 * poly_216 - poly_1102 - poly_1112 
    poly_1257 = jnp.take(mono,1641) + jnp.take(mono,1642) + jnp.take(mono,1643) + jnp.take(mono,1644) + jnp.take(mono,1645) + jnp.take(mono,1646) + jnp.take(mono,1647) + jnp.take(mono,1648) + jnp.take(mono,1649) + jnp.take(mono,1650) + jnp.take(mono,1651) + jnp.take(mono,1652) + jnp.take(mono,1653) + jnp.take(mono,1654) + jnp.take(mono,1655) + jnp.take(mono,1656) + jnp.take(mono,1657) + jnp.take(mono,1658) + jnp.take(mono,1659) + jnp.take(mono,1660) + jnp.take(mono,1661) + jnp.take(mono,1662) + jnp.take(mono,1663) + jnp.take(mono,1664) 
    poly_1258 = poly_5 * poly_217 - poly_1135 - poly_1137 
    poly_1259 = poly_1 * poly_553 - poly_1142 - poly_1256 
    poly_1260 = poly_5 * poly_246 - poly_1254 - poly_1257 
    poly_1261 = poly_11 * poly_96 - poly_1103 - poly_1134 - poly_1211 - poly_1103 
    poly_1262 = poly_1 * poly_555 - poly_1144 - poly_1136 - poly_1261 
    poly_1263 = poly_1 * poly_556 - poly_1145 - poly_1135 - poly_1137 
    poly_1264 = poly_1 * poly_557 - poly_1146 - poly_1138 - poly_1135 
    poly_1265 = poly_2 * poly_522 - poly_1119 - poly_1114 - poly_1261 - poly_1234 - poly_1260 - poly_1114 
    poly_1266 = poly_4 * poly_218 - poly_1104 - poly_1133 - poly_1136 - poly_1132 - poly_1119 - poly_1254 - poly_1104 
    poly_1267 = poly_5 * poly_219 - poly_1140 
    poly_1268 = poly_1 * poly_558 - poly_1147 - poly_1136 - poly_1265 
    poly_1269 = poly_1 * poly_559 - poly_1148 - poly_1136 - poly_1266 
    poly_1270 = poly_1 * poly_560 - poly_1149 - poly_1138 - poly_1137 
    poly_1271 = poly_1 * poly_561 - poly_1150 - poly_1139 - poly_1139 
    poly_1272 = poly_220 * poly_5 
    poly_1273 = poly_46 * poly_48 - poly_1201 - poly_1198 
    poly_1274 = poly_3 * poly_279 - poly_1141 - poly_1143 - poly_1134 - poly_1254 - poly_1257 - poly_1141 - poly_1134 
    poly_1275 = poly_5 * poly_223 - poly_1146 - poly_1149 
    poly_1276 = poly_20 * poly_103 - poly_1122 - poly_1112 
    poly_1277 = poly_5 * poly_222 - poly_1145 
    poly_1278 = poly_10 * poly_145 - poly_1142 - poly_1259 
    poly_1279 = poly_5 * poly_249 - poly_1274 
    poly_1280 = poly_2 * poly_527 - poly_1119 - poly_1211 - poly_1279 
    poly_1281 = poly_3 * poly_283 - poly_1148 - poly_1144 - poly_1136 - poly_1266 - poly_1262 
    poly_1282 = poly_1 * poly_682 - poly_1267 
    poly_1283 = poly_3 * poly_285 - poly_1148 - poly_1147 - poly_1136 - poly_1269 - poly_1268 
    poly_1284 = poly_11 * poly_120 - poly_1201 
    poly_1285 = poly_32 * poly_46 - poly_1143 
    poly_1286 = poly_10 * poly_147 - poly_1142 - poly_1284 
    poly_1287 = poly_5 * poly_250 - poly_1285 
    poly_1288 = poly_1 * poly_567 - poly_1147 - poly_1144 - poly_1280 
    poly_1289 = poly_1 * poly_568 - poly_1148 - poly_1144 - poly_1281 
    poly_1290 = poly_1 * poly_569 - poly_1146 - poly_1149 - poly_1145 
    poly_1291 = poly_1 * poly_570 - poly_1148 - poly_1147 - poly_1283 
    poly_1292 = poly_8 * poly_286 - poly_1271 
    poly_1293 = poly_17 * poly_85 - poly_1103 - poly_1136 - poly_1130 - poly_1134 - poly_1127 - poly_1103 - poly_1134 
    poly_1294 = poly_35 * poly_38 - poly_1135 - poly_1137 
    poly_1295 = poly_36 * poly_38 - poly_1138 
    poly_1296 = poly_18 * poly_85 - poly_1104 - poly_1136 
    poly_1297 = poly_18 * poly_86 - poly_1138 
    poly_1298 = poly_38 * poly_39 
    poly_1299 = jnp.take(mono,1665) + jnp.take(mono,1666) + jnp.take(mono,1667) + jnp.take(mono,1668) + jnp.take(mono,1669) + jnp.take(mono,1670) + jnp.take(mono,1671) + jnp.take(mono,1672) + jnp.take(mono,1673) + jnp.take(mono,1674) + jnp.take(mono,1675) + jnp.take(mono,1676) + jnp.take(mono,1677) + jnp.take(mono,1678) + jnp.take(mono,1679) + jnp.take(mono,1680) + jnp.take(mono,1681) + jnp.take(mono,1682) + jnp.take(mono,1683) + jnp.take(mono,1684) + jnp.take(mono,1685) + jnp.take(mono,1686) + jnp.take(mono,1687) + jnp.take(mono,1688) 
    poly_1300 = jnp.take(mono,1689) + jnp.take(mono,1690) + jnp.take(mono,1691) + jnp.take(mono,1692) + jnp.take(mono,1693) + jnp.take(mono,1694) + jnp.take(mono,1695) + jnp.take(mono,1696) + jnp.take(mono,1697) + jnp.take(mono,1698) + jnp.take(mono,1699) + jnp.take(mono,1700) + jnp.take(mono,1701) + jnp.take(mono,1702) + jnp.take(mono,1703) + jnp.take(mono,1704) + jnp.take(mono,1705) + jnp.take(mono,1706) + jnp.take(mono,1707) + jnp.take(mono,1708) + jnp.take(mono,1709) + jnp.take(mono,1710) + jnp.take(mono,1711) + jnp.take(mono,1712) 
    poly_1301 = jnp.take(mono,1713) + jnp.take(mono,1714) + jnp.take(mono,1715) + jnp.take(mono,1716) + jnp.take(mono,1717) + jnp.take(mono,1718) + jnp.take(mono,1719) + jnp.take(mono,1720) + jnp.take(mono,1721) + jnp.take(mono,1722) + jnp.take(mono,1723) + jnp.take(mono,1724) + jnp.take(mono,1725) + jnp.take(mono,1726) + jnp.take(mono,1727) + jnp.take(mono,1728) + jnp.take(mono,1729) + jnp.take(mono,1730) + jnp.take(mono,1731) + jnp.take(mono,1732) + jnp.take(mono,1733) + jnp.take(mono,1734) + jnp.take(mono,1735) + jnp.take(mono,1736) 
    poly_1302 = poly_30 * poly_76 
    poly_1303 = poly_19 * poly_86 - poly_1144 - poly_1143 - poly_1123 - poly_1249 - poly_1244 
    poly_1304 = poly_38 * poly_40 - poly_1145 
    poly_1305 = poly_15 * poly_92 - poly_1147 - poly_1130 - poly_1119 - poly_1274 - poly_1227 
    poly_1306 = poly_10 * poly_175 - poly_1305 - poly_1293 - poly_1303 
    poly_1307 = poly_38 * poly_41 - poly_1146 - poly_1149 
    poly_1308 = poly_39 * poly_41 - poly_1147 - poly_1119 - poly_1280 
    poly_1309 = poly_7 * poly_215 - poly_1104 - poly_1148 
    poly_1310 = poly_18 * poly_89 - poly_1146 - poly_1137 - poly_1300 
    poly_1311 = poly_18 * poly_90 - poly_1149 - poly_1135 - poly_1299 
    poly_1312 = poly_38 * poly_43 - poly_1301 
    poly_1313 = poly_18 * poly_92 - poly_1145 
    poly_1314 = poly_7 * poly_216 - poly_1103 - poly_1136 
    poly_1315 = poly_20 * poly_86 - poly_1145 
    poly_1316 = jnp.take(mono,1737) + jnp.take(mono,1738) + jnp.take(mono,1739) + jnp.take(mono,1740) + jnp.take(mono,1741) + jnp.take(mono,1742) + jnp.take(mono,1743) + jnp.take(mono,1744) + jnp.take(mono,1745) + jnp.take(mono,1746) + jnp.take(mono,1747) + jnp.take(mono,1748) 
    poly_1317 = jnp.take(mono,1749) + jnp.take(mono,1750) + jnp.take(mono,1751) + jnp.take(mono,1752) + jnp.take(mono,1753) + jnp.take(mono,1754) + jnp.take(mono,1755) + jnp.take(mono,1756) + jnp.take(mono,1757) + jnp.take(mono,1758) + jnp.take(mono,1759) + jnp.take(mono,1760) + jnp.take(mono,1761) + jnp.take(mono,1762) + jnp.take(mono,1763) + jnp.take(mono,1764) + jnp.take(mono,1765) + jnp.take(mono,1766) + jnp.take(mono,1767) + jnp.take(mono,1768) + jnp.take(mono,1769) + jnp.take(mono,1770) + jnp.take(mono,1771) + jnp.take(mono,1772) 
    poly_1318 = poly_17 * poly_95 - poly_1138 - poly_1135 - poly_1137 - poly_1317 - poly_1315 
    poly_1319 = poly_38 * poly_44 - poly_1316 
    poly_1320 = poly_7 * poly_217 - poly_1105 - poly_1150 - poly_1138 - poly_1139 - poly_1145 - poly_1137 - poly_1310 - poly_1297 - poly_1318 - poly_1315 - poly_1105 - poly_1139 - poly_1137 - poly_1105 - poly_1139 - poly_1105 - poly_1139 
    poly_1321 = poly_18 * poly_95 - poly_1105 - poly_1320 - poly_1105 
    poly_1322 = poly_15 * poly_98 - poly_1140 - poly_1298 - poly_1316 - poly_1140 
    poly_1323 = poly_7 * poly_219 - poly_1151 - poly_1140 - poly_1312 - poly_1322 - poly_1298 - poly_1319 - poly_1151 - poly_1140 
    poly_1324 = poly_220 * poly_7 
    poly_1325 = poly_16 * poly_98 - poly_1151 - poly_1140 - poly_1323 - poly_1301 - poly_1319 - poly_1151 - poly_1140 
    poly_1326 = poly_35 * poly_46 - poly_1144 - poly_1123 - poly_1288 
    poly_1327 = poly_10 * poly_178 - poly_1326 - poly_1314 
    poly_1328 = poly_20 * poly_89 - poly_1146 - poly_1137 - poly_1318 
    poly_1329 = poly_20 * poly_90 - poly_1149 - poly_1135 - poly_1317 
    poly_1330 = poly_38 * poly_46 
    poly_1331 = poly_20 * poly_92 - poly_1138 
    poly_1332 = poly_39 * poly_46 - poly_1124 
    poly_1333 = poly_3 * poly_376 - poly_1321 - poly_1332 - poly_1320 
    poly_1334 = poly_1 * poly_720 - poly_1325 - poly_1323 - poly_1322 
    poly_1335 = poly_30 * poly_78 
    poly_1336 = poly_5 * poly_256 - poly_1136 - poly_1293 
    poly_1337 = poly_38 * poly_56 - poly_1139 
    poly_1338 = poly_5 * poly_261 - poly_1297 
    poly_1339 = poly_5 * poly_264 - poly_1144 - poly_1303 
    poly_1340 = poly_30 * poly_80 
    poly_1341 = poly_5 * poly_267 - poly_1147 - poly_1305 
    poly_1342 = poly_5 * poly_268 - poly_1148 - poly_1306 
    poly_1343 = poly_2 * poly_541 - poly_1144 - poly_1136 - poly_1127 - poly_1293 - poly_1303 
    poly_1344 = poly_9 * poly_211 - poly_1152 - poly_1139 
    poly_1345 = poly_38 * poly_58 - poly_1150 - poly_1344 
    poly_1346 = poly_10 * poly_188 - poly_1343 
    poly_1347 = poly_38 * poly_60 - poly_1152 
    poly_1348 = poly_18 * poly_122 - poly_1141 - poly_1134 
    poly_1349 = poly_5 * poly_274 - poly_1299 - poly_1310 
    poly_1350 = poly_5 * poly_275 - poly_1300 - poly_1311 
    poly_1351 = poly_39 * poly_60 - poly_1279 
    poly_1352 = poly_9 * poly_215 - poly_1141 
    poly_1353 = poly_5 * poly_277 - poly_1313 
    poly_1354 = poly_5 * poly_278 - poly_1315 
    poly_1355 = poly_9 * poly_216 - poly_1132 
    poly_1356 = poly_5 * poly_280 - poly_1317 - poly_1318 
    poly_1357 = poly_45 * poly_56 - poly_1135 - poly_1338 - poly_1354 
    poly_1358 = jnp.take(mono,1773) + jnp.take(mono,1774) + jnp.take(mono,1775) + jnp.take(mono,1776) + jnp.take(mono,1777) + jnp.take(mono,1778) + jnp.take(mono,1779) + jnp.take(mono,1780) + jnp.take(mono,1781) + jnp.take(mono,1782) + jnp.take(mono,1783) + jnp.take(mono,1784) + jnp.take(mono,1785) + jnp.take(mono,1786) + jnp.take(mono,1787) + jnp.take(mono,1788) + jnp.take(mono,1789) + jnp.take(mono,1790) + jnp.take(mono,1791) + jnp.take(mono,1792) + jnp.take(mono,1793) + jnp.take(mono,1794) + jnp.take(mono,1795) + jnp.take(mono,1796) 
    poly_1359 = poly_4 * poly_284 - poly_1151 - poly_1140 - poly_1323 - poly_1301 - poly_1322 - poly_1358 - poly_1298 - poly_1316 - poly_1319 - poly_1282 - poly_1267 - poly_1151 - poly_1140 - poly_1298 - poly_1316 
    poly_1360 = poly_2 * poly_558 - poly_1150 - poly_1137 - poly_1320 - poly_1310 - poly_1318 
    poly_1361 = poly_9 * poly_218 - poly_1145 - poly_1353 - poly_1354 
    poly_1362 = jnp.take(mono,1797) + jnp.take(mono,1798) + jnp.take(mono,1799) + jnp.take(mono,1800) + jnp.take(mono,1801) + jnp.take(mono,1802) + jnp.take(mono,1803) + jnp.take(mono,1804) + jnp.take(mono,1805) + jnp.take(mono,1806) + jnp.take(mono,1807) + jnp.take(mono,1808) + jnp.take(mono,1809) + jnp.take(mono,1810) + jnp.take(mono,1811) + jnp.take(mono,1812) + jnp.take(mono,1813) + jnp.take(mono,1814) + jnp.take(mono,1815) + jnp.take(mono,1816) + jnp.take(mono,1817) + jnp.take(mono,1818) + jnp.take(mono,1819) + jnp.take(mono,1820) 
    poly_1363 = poly_2 * poly_561 - poly_1140 - poly_1323 - poly_1322 
    poly_1364 = poly_220 * poly_9 
    poly_1365 = poly_46 * poly_56 - poly_1212 
    poly_1366 = poly_5 * poly_287 - poly_1328 - poly_1329 
    poly_1367 = poly_10 * poly_191 - poly_1365 - poly_1355 
    poly_1368 = poly_5 * poly_289 - poly_1331 
    poly_1369 = poly_9 * poly_222 - poly_1138 - poly_1338 - poly_1368 
    poly_1370 = poly_2 * poly_568 - poly_1150 - poly_1146 - poly_1333 - poly_1300 - poly_1328 
    poly_1371 = poly_1 * poly_757 - poly_1359 - poly_1362 - poly_1358 
    poly_1372 = poly_45 * poly_60 - poly_1149 - poly_1353 - poly_1368 
    poly_1373 = poly_1 * poly_759 - poly_1363 
    poly_1374 = poly_30 * poly_48 
    poly_1375 = poly_30 * poly_71 
    poly_1376 = poly_30 * poly_31 
    poly_1377 = poly_30 * poly_49 
    poly_1378 = poly_30 * poly_33 
    poly_1379 = poly_5 * poly_405 
    poly_1380 = poly_3 * poly_226 - poly_1155 - poly_1153 - poly_1379 - poly_1153 
    poly_1381 = poly_30 * poly_50 
    poly_1382 = poly_10 * poly_105 - poly_1155 - poly_1153 - poly_1380 - poly_1155 - poly_1153 
    poly_1383 = poly_30 * poly_52 
    poly_1384 = poly_5 * poly_406 - poly_1380 - poly_1382 
    poly_1385 = poly_1 * poly_578 - poly_1155 - poly_1380 
    poly_1386 = poly_30 * poly_72 
    poly_1387 = poly_5 * poly_407 - poly_1385 
    poly_1388 = jnp.take(mono,1821) + jnp.take(mono,1822) + jnp.take(mono,1823) + jnp.take(mono,1824) + jnp.take(mono,1825) + jnp.take(mono,1826) + jnp.take(mono,1827) + jnp.take(mono,1828) + jnp.take(mono,1829) + jnp.take(mono,1830) + jnp.take(mono,1831) + jnp.take(mono,1832) 
    poly_1389 = poly_1 * poly_581 - poly_1156 - poly_1388 
    poly_1390 = poly_10 * poly_85 - poly_1158 - poly_1157 
    poly_1391 = poly_8 * poly_229 - poly_1165 - poly_1388 
    poly_1392 = poly_1 * poly_583 - poly_1158 - poly_1390 
    poly_1393 = poly_28 * poly_85 - poly_1177 - poly_1388 
    poly_1394 = poly_1 * poly_585 - poly_1161 
    poly_1395 = poly_28 * poly_86 - poly_1178 
    poly_1396 = poly_10 * poly_158 - poly_1156 - poly_1388 
    poly_1397 = poly_10 * poly_108 - poly_1157 - poly_1156 - poly_1389 - poly_1156 
    poly_1398 = poly_10 * poly_109 - poly_1159 - poly_1157 - poly_1391 
    poly_1399 = poly_10 * poly_110 - poly_1159 - poly_1158 - poly_1392 - poly_1158 
    poly_1400 = jnp.take(mono,1833) + jnp.take(mono,1834) + jnp.take(mono,1835) + jnp.take(mono,1836) + jnp.take(mono,1837) + jnp.take(mono,1838) + jnp.take(mono,1839) + jnp.take(mono,1840) + jnp.take(mono,1841) + jnp.take(mono,1842) + jnp.take(mono,1843) + jnp.take(mono,1844) + jnp.take(mono,1845) + jnp.take(mono,1846) + jnp.take(mono,1847) + jnp.take(mono,1848) + jnp.take(mono,1849) + jnp.take(mono,1850) + jnp.take(mono,1851) + jnp.take(mono,1852) + jnp.take(mono,1853) + jnp.take(mono,1854) + jnp.take(mono,1855) + jnp.take(mono,1856) 
    poly_1401 = poly_10 * poly_159 - poly_1159 - poly_1400 - poly_1393 
    poly_1402 = jnp.take(mono,1857) + jnp.take(mono,1858) + jnp.take(mono,1859) + jnp.take(mono,1860) + jnp.take(mono,1861) + jnp.take(mono,1862) + jnp.take(mono,1863) + jnp.take(mono,1864) + jnp.take(mono,1865) + jnp.take(mono,1866) + jnp.take(mono,1867) + jnp.take(mono,1868) + jnp.take(mono,1869) + jnp.take(mono,1870) + jnp.take(mono,1871) + jnp.take(mono,1872) + jnp.take(mono,1873) + jnp.take(mono,1874) + jnp.take(mono,1875) + jnp.take(mono,1876) + jnp.take(mono,1877) + jnp.take(mono,1878) + jnp.take(mono,1879) + jnp.take(mono,1880) 
    poly_1403 = poly_3 * poly_310 - poly_1163 - poly_1160 - poly_1162 - poly_1402 - poly_1395 
    poly_1404 = poly_3 * poly_311 - poly_1164 
    poly_1405 = poly_10 * poly_113 - poly_1165 - poly_1156 
    poly_1406 = poly_1 * poly_588 - poly_1157 - poly_1398 - poly_1397 
    poly_1407 = poly_1 * poly_589 - poly_1158 - poly_1399 
    poly_1408 = poly_1 * poly_590 - poly_1159 - poly_1400 - poly_1398 
    poly_1409 = poly_1 * poly_591 - poly_1159 - poly_1401 - poly_1399 
    poly_1410 = poly_1 * poly_592 - poly_1163 - poly_1160 - poly_1402 
    poly_1411 = poly_1 * poly_593 - poly_1163 - poly_1162 - poly_1403 
    poly_1412 = poly_39 * poly_69 - poly_1175 
    poly_1413 = poly_10 * poly_115 - poly_1177 - poly_1165 
    poly_1414 = poly_1 * poly_596 - poly_1166 - poly_1408 - poly_1406 
    poly_1415 = poly_3 * poly_239 - poly_1177 - poly_1166 - poly_1407 
    poly_1416 = poly_1 * poly_598 - poly_1170 - poly_1167 - poly_1410 
    poly_1417 = poly_1 * poly_599 - poly_1171 - poly_1168 - poly_1411 
    poly_1418 = poly_1 * poly_600 - poly_1172 
    poly_1419 = poly_1 * poly_601 - poly_1174 - poly_1412 
    poly_1420 = poly_1 * poly_602 - poly_1176 - poly_1175 
    poly_1421 = poly_8 * poly_238 - poly_1166 - poly_1408 - poly_1405 
    poly_1422 = poly_1 * poly_604 - poly_1177 - poly_1415 
    poly_1423 = poly_1 * poly_605 - poly_1180 - poly_1178 - poly_1416 
    poly_1424 = poly_1 * poly_606 - poly_1181 - poly_1178 - poly_1417 
    poly_1425 = poly_38 * poly_81 
    poly_1426 = poly_28 * poly_92 - poly_1163 
    poly_1427 = poly_1 * poly_609 - poly_1183 - poly_1419 
    poly_1428 = poly_18 * poly_193 - poly_1427 - poly_1404 
    poly_1429 = poly_1 * poly_611 - poly_1193 - poly_1185 
    poly_1430 = poly_28 * poly_95 - poly_1200 - poly_1186 
    poly_1431 = poly_1 * poly_613 - poly_1195 - poly_1187 
    poly_1432 = poly_1 * poly_614 - poly_1196 - poly_1191 - poly_1190 
    poly_1433 = poly_3 * poly_320 - poly_1194 - poly_1186 - poly_1430 
    poly_1434 = poly_1 * poly_615 - poly_1194 - poly_1433 
    poly_1435 = poly_46 * poly_69 - poly_1193 
    poly_1436 = poly_3 * poly_321 - poly_1200 
    poly_1437 = poly_1 * poly_618 - poly_1202 - poly_1201 
    poly_1438 = poly_5 * poly_293 - poly_1161 
    poly_1439 = poly_5 * poly_294 - poly_1160 - poly_1162 
    poly_1440 = poly_30 * poly_56 
    poly_1441 = poly_30 * poly_73 
    poly_1442 = poly_5 * poly_295 - poly_1163 
    poly_1443 = poly_5 * poly_296 - poly_1167 - poly_1168 
    poly_1444 = poly_5 * poly_226 - poly_1109 
    poly_1445 = poly_30 * poly_58 
    poly_1446 = poly_5 * poly_298 - poly_1170 - poly_1171 
    poly_1447 = poly_47 * poly_74 - poly_1444 
    poly_1448 = poly_30 * poly_60 
    poly_1449 = poly_5 * poly_299 - poly_1172 
    poly_1450 = poly_5 * poly_301 - poly_1178 
    poly_1451 = poly_30 * poly_75 
    poly_1452 = poly_5 * poly_303 - poly_1180 - poly_1181 
    poly_1453 = poly_34 * poly_53 - poly_1112 - poly_1217 - poly_1195 
    poly_1454 = poly_1 * poly_626 - poly_1213 - poly_1207 - poly_1453 
    poly_1455 = poly_1 * poly_627 - poly_1214 - poly_1208 - poly_1454 
    poly_1456 = poly_2 * poly_581 - poly_1164 - poly_1186 - poly_1162 
    poly_1457 = poly_11 * poly_85 - poly_1107 - poly_1106 - poly_1185 - poly_1107 - poly_1107 
    poly_1458 = poly_22 * poly_85 - poly_1111 - poly_1108 - poly_1164 - poly_1186 - poly_1160 - poly_1456 - poly_1186 
    poly_1459 = poly_5 * poly_305 - poly_1187 - poly_1453 
    poly_1460 = jnp.take(mono,1881) + jnp.take(mono,1882) + jnp.take(mono,1883) + jnp.take(mono,1884) + jnp.take(mono,1885) + jnp.take(mono,1886) + jnp.take(mono,1887) + jnp.take(mono,1888) + jnp.take(mono,1889) + jnp.take(mono,1890) + jnp.take(mono,1891) + jnp.take(mono,1892) + jnp.take(mono,1893) + jnp.take(mono,1894) + jnp.take(mono,1895) + jnp.take(mono,1896) + jnp.take(mono,1897) + jnp.take(mono,1898) + jnp.take(mono,1899) + jnp.take(mono,1900) + jnp.take(mono,1901) + jnp.take(mono,1902) + jnp.take(mono,1903) + jnp.take(mono,1904) 
    poly_1461 = poly_1 * poly_628 - poly_1215 - poly_1457 - poly_1456 
    poly_1462 = poly_1 * poly_629 - poly_1216 - poly_1458 - poly_1457 
    poly_1463 = poly_5 * poly_306 - poly_1188 - poly_1454 
    poly_1464 = poly_5 * poly_307 - poly_1189 
    poly_1465 = poly_1 * poly_632 - poly_1219 
    poly_1466 = poly_1 * poly_633 - poly_1221 - poly_1462 - poly_1461 
    poly_1467 = poly_5 * poly_308 - poly_1190 - poly_1455 
    poly_1468 = poly_5 * poly_309 - poly_1191 - poly_1460 
    poly_1469 = poly_2 * poly_585 - poly_1188 - poly_1173 - poly_1463 
    poly_1470 = poly_8 * poly_259 - poly_1242 
    poly_1471 = poly_38 * poly_71 - poly_1192 
    poly_1472 = poly_11 * poly_112 - poly_1112 
    poly_1473 = poly_18 * poly_150 - poly_1189 
    poly_1474 = poly_1 * poly_639 - poly_1233 - poly_1473 
    poly_1475 = poly_5 * poly_311 
    poly_1476 = poly_1 * poly_641 - poly_1235 - poly_1212 
    poly_1477 = poly_11 * poly_113 - poly_1115 - poly_1167 - poly_1412 
    poly_1478 = poly_10 * poly_126 - poly_1216 - poly_1215 - poly_1457 
    poly_1479 = poly_11 * poly_114 - poly_1221 - poly_1175 - poly_1194 - poly_1172 - poly_1411 - poly_1175 
    poly_1480 = poly_5 * poly_312 - poly_1195 
    poly_1481 = poly_3 * poly_335 - poly_1218 - poly_1214 - poly_1208 - poly_1460 - poly_1455 
    poly_1482 = poly_5 * poly_235 - poly_1119 - poly_1226 
    poly_1483 = poly_10 * poly_168 - poly_1215 - poly_1477 - poly_1456 
    poly_1484 = poly_10 * poly_169 - poly_1216 - poly_1479 - poly_1458 
    poly_1485 = poly_10 * poly_129 - poly_1221 - poly_1215 - poly_1461 
    poly_1486 = poly_2 * poly_589 - poly_1176 - poly_1193 - poly_1170 
    poly_1487 = poly_5 * poly_236 - poly_1119 - poly_1227 
    poly_1488 = poly_2 * poly_590 - poly_1183 - poly_1194 - poly_1181 - poly_1162 - poly_1479 
    poly_1489 = poly_10 * poly_171 - poly_1221 - poly_1488 - poly_1466 
    poly_1490 = poly_5 * poly_313 - poly_1196 - poly_1481 
    poly_1491 = poly_2 * poly_592 - poly_1196 - poly_1187 - poly_1179 - poly_1468 - poly_1480 - poly_1179 
    poly_1492 = poly_3 * poly_342 - poly_1230 - poly_1224 - poly_1225 - poly_1491 - poly_1470 
    poly_1493 = poly_6 * poly_311 - poly_1195 - poly_1431 
    poly_1494 = poly_3 * poly_345 - poly_1237 - poly_1233 - poly_1474 
    poly_1495 = poly_1 * poly_642 - poly_1215 - poly_1478 - poly_1477 
    poly_1496 = poly_1 * poly_643 - poly_1216 - poly_1479 - poly_1478 
    poly_1497 = poly_5 * poly_314 - poly_1198 
    poly_1498 = poly_3 * poly_266 - poly_1126 - poly_1218 - poly_1241 - poly_1214 - poly_1238 
    poly_1499 = poly_5 * poly_240 - poly_1123 - poly_1243 
    poly_1500 = poly_1 * poly_647 - poly_1220 
    poly_1501 = poly_1 * poly_648 - poly_1215 - poly_1485 - poly_1483 
    poly_1502 = poly_1 * poly_649 - poly_1216 - poly_1486 - poly_1484 
    poly_1503 = poly_5 * poly_241 - poly_1123 - poly_1244 
    poly_1504 = poly_1 * poly_651 - poly_1221 - poly_1488 - poly_1485 
    poly_1505 = poly_1 * poly_652 - poly_1221 - poly_1489 - poly_1486 
    poly_1506 = poly_5 * poly_315 - poly_1199 - poly_1498 
    poly_1507 = poly_1 * poly_654 - poly_1230 - poly_1224 - poly_1491 
    poly_1508 = poly_1 * poly_655 - poly_1230 - poly_1225 - poly_1492 
    poly_1509 = poly_8 * poly_343 
    poly_1510 = poly_2 * poly_600 - poly_1199 - poly_1173 - poly_1498 
    poly_1511 = poly_23 * poly_112 - poly_1198 - poly_1431 
    poly_1512 = poly_18 * poly_153 - poly_1199 - poly_1188 - poly_1511 
    poly_1513 = poly_18 * poly_154 - poly_1198 
    poly_1514 = poly_1 * poly_659 - poly_1239 - poly_1496 - poly_1495 
    poly_1515 = poly_5 * poly_316 - poly_1201 
    poly_1516 = poly_8 * poly_266 - poly_1223 - poly_1213 - poly_1481 
    poly_1517 = poly_1 * poly_662 - poly_1239 - poly_1504 - poly_1501 
    poly_1518 = poly_1 * poly_663 - poly_1239 - poly_1505 - poly_1502 
    poly_1519 = poly_5 * poly_317 - poly_1202 - poly_1516 
    poly_1520 = poly_1 * poly_665 - poly_1245 - poly_1242 - poly_1507 
    poly_1521 = poly_1 * poly_666 - poly_1246 - poly_1242 - poly_1508 
    poly_1522 = poly_28 * poly_133 - poly_1471 
    poly_1523 = poly_8 * poly_273 - poly_1230 
    poly_1524 = poly_39 * poly_72 - poly_1201 - poly_1431 
    poly_1525 = poly_1 * poly_670 - poly_1252 - poly_1248 - poly_1512 
    poly_1526 = poly_5 * poly_319 - poly_1476 
    poly_1527 = poly_8 * poly_277 - poly_1237 - poly_1236 
    poly_1528 = poly_20 * poly_150 - poly_1198 
    poly_1529 = poly_8 * poly_278 - poly_1284 - poly_1259 
    poly_1530 = poly_1 * poly_674 - poly_1274 - poly_1254 - poly_1257 
    poly_1531 = poly_5 * poly_248 - poly_1264 - poly_1270 
    poly_1532 = poly_22 * poly_95 - poly_1122 - poly_1121 - poly_1112 - poly_1256 - poly_1187 - poly_1529 
    poly_1533 = poly_1 * poly_676 - poly_1276 - poly_1256 - poly_1532 
    poly_1534 = poly_5 * poly_247 - poly_1263 
    poly_1535 = poly_8 * poly_280 - poly_1286 - poly_1256 - poly_1532 
    poly_1536 = poly_5 * poly_320 - poly_1530 
    poly_1537 = poly_1 * poly_680 - poly_1280 - poly_1265 - poly_1261 
    poly_1538 = poly_1 * poly_681 - poly_1281 - poly_1266 - poly_1262 
    poly_1539 = poly_1 * poly_683 - poly_1283 - poly_1269 - poly_1268 
    poly_1540 = poly_46 * poly_71 - poly_1195 - poly_1437 
    poly_1541 = poly_2 * poly_615 - poly_1196 - poly_1195 - poly_1540 
    poly_1542 = poly_22 * poly_120 - poly_1198 - poly_1437 
    poly_1543 = poly_5 * poly_251 - poly_1290 
    poly_1544 = poly_20 * poly_154 - poly_1189 
    poly_1545 = poly_4 * poly_321 - poly_1201 - poly_1437 
    poly_1546 = poly_5 * poly_321 
    poly_1547 = poly_14 * poly_120 - poly_1122 
    poly_1548 = poly_1 * poly_690 - poly_1291 - poly_1289 - poly_1288 
    poly_1549 = poly_7 * poly_229 - poly_1113 - poly_1116 - poly_1110 - poly_1232 - poly_1256 - poly_1225 
    poly_1550 = poly_1 * poly_691 - poly_1302 - poly_1549 
    poly_1551 = poly_15 * poly_86 - poly_1103 - poly_1134 - poly_1127 - poly_1261 - poly_1234 - poly_1103 - poly_1134 
    poly_1552 = poly_1 * poly_692 - poly_1293 - poly_1303 - poly_1551 
    poly_1553 = poly_38 * poly_53 - poly_1263 
    poly_1554 = poly_17 * poly_108 - poly_1119 - poly_1114 - poly_1265 - poly_1261 - poly_1211 - poly_1234 - poly_1254 - poly_1257 - poly_1227 - poly_1210 - poly_1551 - poly_1114 - poly_1211 
    poly_1555 = poly_36 * poly_37 - poly_1104 - poly_1133 - poly_1136 - poly_1130 - poly_1119 - poly_1129 - poly_1266 - poly_1293 - poly_1257 - poly_1260 - poly_1226 - poly_1104 - poly_1133 
    poly_1556 = poly_1 * poly_694 - poly_1305 - poly_1293 - poly_1554 
    poly_1557 = poly_1 * poly_695 - poly_1306 - poly_1293 - poly_1555 
    poly_1558 = poly_38 * poly_54 - poly_1264 - poly_1270 
    poly_1559 = poly_5 * poly_259 - poly_1139 - poly_1337 - poly_1139 
    poly_1560 = poly_37 * poly_38 - poly_1140 - poly_1267 - poly_1140 
    poly_1561 = poly_16 * poly_112 - poly_1119 - poly_1265 - poly_1537 
    poly_1562 = poly_18 * poly_109 - poly_1123 - poly_1269 - poly_1261 
    poly_1563 = poly_1 * poly_698 - poly_1309 - poly_1296 
    poly_1564 = poly_17 * poly_112 - poly_1263 
    poly_1565 = poly_1 * poly_700 - poly_1300 - poly_1311 - poly_1297 
    poly_1566 = poly_1 * poly_702 - poly_1313 - poly_1300 - poly_1299 
    poly_1567 = poly_10 * poly_173 - poly_1302 - poly_1549 
    poly_1568 = poly_11 * poly_140 - poly_1340 - poly_1251 - poly_1278 - poly_1246 - poly_1506 
    poly_1569 = poly_10 * poly_174 - poly_1302 - poly_1568 - poly_1550 - poly_1302 
    poly_1570 = poly_3 * poly_362 - poly_1305 - poly_1293 - poly_1303 - poly_1554 - poly_1551 
    poly_1571 = poly_3 * poly_363 - poly_1306 - poly_1293 - poly_1303 - poly_1555 - poly_1552 
    poly_1572 = poly_3 * poly_365 - poly_1306 - poly_1305 - poly_1293 - poly_1557 - poly_1556 
    poly_1573 = poly_5 * poly_270 - poly_1150 - poly_1344 
    poly_1574 = poly_5 * poly_271 - poly_1150 - poly_1345 
    poly_1575 = poly_1 * poly_859 - poly_1560 
    poly_1576 = poly_3 * poly_436 - poly_1573 - poly_1574 - poly_1559 
    poly_1577 = poly_18 * poly_113 - poly_1114 - poly_1281 
    poly_1578 = poly_14 * poly_135 - poly_1144 - poly_1352 - poly_1212 
    poly_1579 = poly_39 * poly_43 - poly_1150 - poly_1271 
    poly_1580 = poly_3 * poly_437 - poly_1579 
    poly_1581 = poly_1 * poly_703 - poly_1302 - poly_1568 - poly_1567 - poly_1302 
    poly_1582 = poly_1 * poly_704 - poly_1302 - poly_1569 
    poly_1583 = poly_1 * poly_705 - poly_1305 - poly_1303 - poly_1570 
    poly_1584 = poly_1 * poly_706 - poly_1306 - poly_1303 - poly_1571 
    poly_1585 = poly_38 * poly_55 - poly_1290 
    poly_1586 = poly_3 * poly_371 - poly_1306 - poly_1305 - poly_1303 - poly_1584 - poly_1583 
    poly_1587 = poly_1 * poly_709 - poly_1308 - poly_1577 
    poly_1588 = poly_3 * poly_372 - poly_1309 - poly_1308 - poly_1587 
    poly_1589 = poly_1 * poly_711 - poly_1313 - poly_1311 - poly_1310 
    poly_1590 = poly_20 * poly_108 - poly_1123 - poly_1280 - poly_1261 
    poly_1591 = poly_11 * poly_145 - poly_1147 - poly_1355 - poly_1279 
    poly_1592 = poly_1 * poly_713 - poly_1327 - poly_1314 
    poly_1593 = poly_1 * poly_714 - poly_1328 - poly_1317 - poly_1315 
    poly_1594 = poly_1 * poly_715 - poly_1329 - poly_1318 - poly_1315 
    poly_1595 = poly_1 * poly_717 - poly_1331 - poly_1317 - poly_1318 
    poly_1596 = poly_5 * poly_283 - poly_1323 - poly_1359 
    poly_1597 = poly_5 * poly_282 - poly_1322 - poly_1358 
    poly_1598 = poly_38 * poly_45 - poly_1324 
    poly_1599 = poly_5 * poly_285 - poly_1325 - poly_1362 
    poly_1600 = poly_20 * poly_112 - poly_1124 
    poly_1601 = poly_1 * poly_719 - poly_1333 - poly_1321 - poly_1320 - poly_1321 
    poly_1602 = poly_5 * poly_286 - poly_1364 
    poly_1603 = poly_18 * poly_96 - poly_1140 - poly_1323 
    poly_1604 = poly_18 * poly_97 - poly_1151 - poly_1325 - poly_1322 - poly_1151 
    poly_1605 = poly_18 * poly_98 - poly_1324 
    poly_1606 = poly_20 * poly_113 - poly_1125 - poly_1265 
    poly_1607 = poly_20 * poly_114 - poly_1119 - poly_1291 - poly_1266 
    poly_1608 = poly_5 * poly_290 - poly_1334 - poly_1371 
    poly_1609 = poly_1 * poly_874 - poly_1604 - poly_1603 
    poly_1610 = poly_1 * poly_721 - poly_1326 - poly_1606 
    poly_1611 = poly_3 * poly_377 - poly_1327 - poly_1326 - poly_1610 
    poly_1612 = poly_17 * poly_120 - poly_1290 
    poly_1613 = poly_18 * poly_120 - poly_1124 
    poly_1614 = poly_20 * poly_95 - poly_1150 - poly_1139 - poly_1139 
    poly_1615 = poly_20 * poly_96 - poly_1151 - poly_1322 
    poly_1616 = poly_20 * poly_97 - poly_1140 - poly_1334 - poly_1323 - poly_1140 
    poly_1617 = poly_20 * poly_98 - poly_1324 
    poly_1618 = poly_3 * poly_439 - poly_1614 
    poly_1619 = poly_1 * poly_877 - poly_1616 - poly_1615 
    poly_1620 = poly_5 * poly_323 - poly_1224 - poly_1225 
    poly_1621 = poly_5 * poly_252 - poly_1127 
    poly_1622 = poly_30 * poly_77 
    poly_1623 = poly_5 * poly_326 - poly_1230 
    poly_1624 = poly_5 * poly_254 - poly_1129 
    poly_1625 = poly_5 * poly_328 - poly_1242 
    poly_1626 = poly_30 * poly_79 
    poly_1627 = poly_5 * poly_330 - poly_1245 - poly_1246 
    poly_1628 = poly_5 * poly_332 - poly_1261 - poly_1551 
    poly_1629 = poly_5 * poly_333 - poly_1262 - poly_1552 
    poly_1630 = poly_5 * poly_255 - poly_1135 - poly_1294 
    poly_1631 = poly_9 * poly_229 - poly_1236 - poly_1259 - poly_1222 
    poly_1632 = poly_36 * poly_56 - poly_1131 - poly_1213 - poly_1528 
    poly_1633 = poly_5 * poly_336 - poly_1265 - poly_1554 
    poly_1634 = poly_5 * poly_337 - poly_1266 - poly_1555 
    poly_1635 = poly_5 * poly_257 - poly_1137 - poly_1294 
    poly_1636 = poly_36 * poly_74 - poly_1255 
    poly_1637 = poly_1 * poly_729 - poly_1340 - poly_1632 - poly_1631 
    poly_1638 = poly_5 * poly_339 - poly_1268 - poly_1556 
    poly_1639 = poly_5 * poly_340 - poly_1269 - poly_1557 
    poly_1640 = poly_1 * poly_732 - poly_1343 
    poly_1641 = poly_38 * poly_73 - poly_1271 
    poly_1642 = poly_38 * poly_57 - poly_1140 
    poly_1643 = poly_39 * poly_56 - poly_1134 
    poly_1644 = poly_1 * poly_734 - poly_1348 - poly_1643 
    poly_1645 = poly_5 * poly_344 - poly_1564 
    poly_1646 = poly_5 * poly_345 - poly_1565 
    poly_1647 = poly_39 * poly_74 
    poly_1648 = poly_5 * poly_263 - poly_1301 
    poly_1649 = poly_40 * poly_56 - poly_1142 - poly_1208 - poly_1473 
    poly_1650 = poly_2 * poly_643 - poly_1233 - poly_1273 - poly_1225 - poly_1214 - poly_1568 
    poly_1651 = poly_5 * poly_346 - poly_1280 - poly_1570 
    poly_1652 = poly_5 * poly_347 - poly_1281 - poly_1571 
    poly_1653 = poly_40 * poly_74 - poly_1277 
    poly_1654 = poly_5 * poly_266 - poly_1146 - poly_1307 
    poly_1655 = poly_10 * poly_184 - poly_1340 - poly_1649 - poly_1631 
    poly_1656 = poly_10 * poly_185 - poly_1340 - poly_1650 - poly_1632 
    poly_1657 = poly_5 * poly_269 - poly_1149 - poly_1307 
    poly_1658 = poly_2 * poly_651 - poly_1251 - poly_1278 - poly_1246 - poly_1222 - poly_1568 
    poly_1659 = poly_10 * poly_187 - poly_1340 - poly_1658 - poly_1637 
    poly_1660 = poly_5 * poly_349 - poly_1283 - poly_1572 
    poly_1661 = poly_2 * poly_654 - poly_1281 - poly_1265 - poly_1226 - poly_1555 - poly_1570 
    poly_1662 = poly_2 * poly_655 - poly_1266 - poly_1280 - poly_1227 - poly_1571 - poly_1554 
    poly_1663 = poly_1 * poly_903 - poly_1642 
    poly_1664 = poly_26 * poly_112 - poly_1279 - poly_1536 
    poly_1665 = poly_3 * poly_392 - poly_1352 - poly_1348 - poly_1644 
    poly_1666 = poly_43 * poly_74 - poly_1648 
    poly_1667 = poly_1 * poly_737 - poly_1340 - poly_1650 - poly_1649 
    poly_1668 = poly_5 * poly_350 - poly_1288 - poly_1583 
    poly_1669 = poly_5 * poly_351 - poly_1289 - poly_1584 
    poly_1670 = poly_1 * poly_740 - poly_1340 - poly_1658 - poly_1655 
    poly_1671 = poly_1 * poly_741 - poly_1340 - poly_1659 - poly_1656 
    poly_1672 = poly_5 * poly_353 - poly_1291 - poly_1586 
    poly_1673 = poly_1 * poly_743 - poly_1346 - poly_1343 - poly_1661 
    poly_1674 = poly_1 * poly_744 - poly_1346 - poly_1343 - poly_1662 
    poly_1675 = poly_8 * poly_390 - poly_1641 
    poly_1676 = poly_1 * poly_746 - poly_1346 
    poly_1677 = poly_39 * poly_75 - poly_1287 - poly_1536 
    poly_1678 = poly_18 * poly_165 - poly_1285 - poly_1260 - poly_1677 
    poly_1679 = poly_5 * poly_355 - poly_1566 - poly_1589 
    poly_1680 = poly_1 * poly_750 - poly_1352 - poly_1351 
    poly_1681 = poly_1 * poly_751 - poly_1365 - poly_1355 
    poly_1682 = poly_5 * poly_356 - poly_1593 - poly_1594 
    poly_1683 = poly_5 * poly_279 - poly_1316 - poly_1319 - poly_1316 
    poly_1684 = poly_2 * poly_676 - poly_1266 - poly_1265 - poly_1257 - poly_1592 - poly_1590 
    poly_1685 = poly_5 * poly_281 - poly_1319 
    poly_1686 = poly_1 * poly_753 - poly_1367 - poly_1355 - poly_1684 - poly_1355 
    poly_1687 = poly_5 * poly_358 - poly_1595 
    poly_1688 = poly_1 * poly_755 - poly_1369 - poly_1360 - poly_1357 
    poly_1689 = poly_1 * poly_756 - poly_1370 - poly_1361 - poly_1357 
    poly_1690 = poly_5 * poly_284 - poly_1324 - poly_1598 - poly_1324 
    poly_1691 = poly_1 * poly_758 - poly_1372 - poly_1361 - poly_1360 
    poly_1692 = poly_46 * poly_73 - poly_1211 - poly_1476 
    poly_1693 = poly_46 * poly_74 
    poly_1694 = poly_2 * poly_686 - poly_1283 - poly_1279 - poly_1607 
    poly_1695 = poly_24 * poly_120 - poly_1212 - poly_1476 
    poly_1696 = poly_5 * poly_359 - poly_1612 
    poly_1697 = poly_46 * poly_60 - poly_1133 
    poly_1698 = poly_1 * poly_763 - poly_1372 - poly_1370 - poly_1369 
    poly_1699 = poly_15 * poly_128 - poly_1138 - poly_1299 - poly_1338 - poly_1275 - poly_1318 - poly_1258 - poly_1263 - poly_1229 - poly_1573 - poly_1682 - poly_1641 - poly_1338 - poly_1229 
    poly_1700 = poly_2 * poly_691 - poly_1296 - poly_1314 - poly_1293 
    poly_1701 = poly_5 * poly_360 - poly_1320 - poly_1699 
    poly_1702 = poly_5 * poly_361 - poly_1321 
    poly_1703 = poly_7 * poly_259 - poly_1135 - poly_1357 - poly_1297 - poly_1315 - poly_1294 
    poly_1704 = poly_38 * poly_61 - poly_1322 - poly_1358 
    poly_1705 = poly_38 * poly_62 - poly_1323 - poly_1359 
    poly_1706 = poly_2 * poly_694 - poly_1299 - poly_1317 - poly_1294 - poly_1699 - poly_1574 
    poly_1707 = poly_2 * poly_695 - poly_1300 - poly_1318 - poly_1295 - poly_1702 - poly_1573 - poly_1295 
    poly_1708 = poly_38 * poly_64 - poly_1325 - poly_1362 
    poly_1709 = poly_11 * poly_176 - poly_1321 - poly_1311 - poly_1579 
    poly_1710 = poly_5 * poly_366 - poly_1603 
    poly_1711 = poly_18 * poly_128 - poly_1358 - poly_1282 - poly_1599 
    poly_1712 = poly_18 * poly_129 - poly_1145 - poly_1361 - poly_1315 
    poly_1713 = poly_2 * poly_698 - poly_1321 - poly_1300 - poly_1580 
    poly_1714 = poly_5 * poly_367 - poly_1604 - poly_1711 
    poly_1715 = poly_39 * poly_65 - poly_1319 
    poly_1716 = poly_18 * poly_132 - poly_1316 - poly_1319 - poly_1715 - poly_1316 
    poly_1717 = poly_18 * poly_133 - poly_1598 
    poly_1718 = poly_2 * poly_702 - poly_1325 - poly_1301 - poly_1609 - poly_1711 - poly_1608 - poly_1609 
    poly_1719 = poly_11 * poly_177 - poly_1327 - poly_1587 - poly_1584 
    poly_1720 = poly_5 * poly_369 - poly_1332 
    poly_1721 = poly_32 * poly_76 - poly_1333 - poly_1320 - poly_1702 - poly_1720 - poly_1701 
    poly_1722 = poly_14 * poly_173 - poly_1296 - poly_1606 - poly_1554 
    poly_1723 = poly_2 * poly_704 - poly_1309 - poly_1327 - poly_1306 
    poly_1724 = poly_5 * poly_370 - poly_1333 - poly_1721 
    poly_1725 = poly_2 * poly_705 - poly_1310 - poly_1328 - poly_1304 - poly_1720 - poly_1573 - poly_1304 
    poly_1726 = poly_2 * poly_706 - poly_1311 - poly_1329 - poly_1307 - poly_1724 - poly_1574 
    poly_1727 = poly_38 * poly_67 - poly_1334 - poly_1371 
    poly_1728 = poly_3 * poly_457 - poly_1707 - poly_1726 - poly_1706 - poly_1725 - poly_1703 
    poly_1729 = poly_2 * poly_709 - poly_1332 - poly_1310 - poly_1579 
    poly_1730 = poly_18 * poly_138 - poly_1138 - poly_1369 - poly_1331 
    poly_1731 = poly_5 * poly_372 - poly_1609 
    poly_1732 = poly_14 * poly_176 - poly_1332 - poly_1299 - poly_1580 
    poly_1733 = poly_1 * poly_935 - poly_1718 - poly_1716 - poly_1715 
    poly_1734 = poly_11 * poly_178 - poly_1332 - poly_1329 - poly_1614 
    poly_1735 = poly_5 * poly_373 - poly_1615 
    poly_1736 = poly_20 * poly_128 - poly_1359 - poly_1267 - poly_1608 
    poly_1737 = poly_20 * poly_129 - poly_1138 - poly_1369 - poly_1297 
    poly_1738 = poly_2 * poly_713 - poly_1321 - poly_1318 - poly_1614 
    poly_1739 = poly_5 * poly_374 - poly_1616 - poly_1736 
    poly_1740 = poly_2 * poly_714 - poly_1322 - poly_1316 - poly_1616 - poly_1596 - poly_1735 - poly_1316 
    poly_1741 = poly_20 * poly_132 - poly_1301 - poly_1298 - poly_1740 - poly_1298 
    poly_1742 = poly_20 * poly_133 - poly_1598 
    poly_1743 = poly_2 * poly_717 - poly_1325 - poly_1319 - poly_1616 - poly_1599 - poly_1736 
    poly_1744 = poly_20 * poly_134 - poly_1140 - poly_1603 
    poly_1745 = poly_20 * poly_135 - poly_1140 - poly_1609 
    poly_1746 = poly_7 * poly_284 - poly_1364 - poly_1324 - poly_1272 - poly_1605 - poly_1717 - poly_1617 - poly_1742 - poly_1602 - poly_1598 - poly_1364 - poly_1324 - poly_1272 - poly_1605 - poly_1617 - poly_1602 - poly_1598 - poly_1364 - poly_1324 - poly_1272 - poly_1364 - poly_1324 - poly_1272 - poly_1272 - poly_1272 - poly_1272 - poly_1272 
    poly_1747 = poly_18 * poly_145 - poly_1151 - poly_1615 
    poly_1748 = poly_7 * poly_286 - poly_1324 - poly_1605 - poly_1617 - poly_1324 
    poly_1749 = poly_2 * poly_721 - poly_1332 - poly_1328 - poly_1618 
    poly_1750 = poly_20 * poly_138 - poly_1145 - poly_1361 - poly_1313 
    poly_1751 = poly_5 * poly_377 - poly_1619 
    poly_1752 = poly_14 * poly_178 - poly_1321 - poly_1317 - poly_1618 
    poly_1753 = poly_46 * poly_65 - poly_1301 
    poly_1754 = poly_18 * poly_147 - poly_1151 - poly_1619 
    poly_1755 = poly_5 * poly_379 - poly_1343 
    poly_1756 = poly_30 * poly_82 
    poly_1757 = poly_5 * poly_381 - poly_1346 
    poly_1758 = poly_5 * poly_383 - poly_1357 - poly_1703 
    poly_1759 = poly_29 * poly_85 - poly_1351 - poly_1355 - poly_1341 
    poly_1760 = poly_5 * poly_386 - poly_1360 - poly_1706 
    poly_1761 = poly_5 * poly_387 - poly_1361 - poly_1707 
    poly_1762 = poly_2 * poly_732 - poly_1357 - poly_1337 - poly_1703 
    poly_1763 = poly_38 * poly_77 - poly_1363 
    poly_1764 = poly_18 * poly_179 - poly_1354 
    poly_1765 = poly_5 * poly_391 - poly_1715 
    poly_1766 = poly_5 * poly_392 - poly_1716 
    poly_1767 = poly_19 * poly_179 - poly_1365 - poly_1644 - poly_1629 
    poly_1768 = poly_5 * poly_394 - poly_1369 - poly_1725 
    poly_1769 = poly_5 * poly_395 - poly_1370 - poly_1726 
    poly_1770 = poly_15 * poly_183 - poly_1351 - poly_1694 - poly_1660 
    poly_1771 = poly_10 * poly_205 - poly_1770 - poly_1767 - poly_1759 
    poly_1772 = poly_5 * poly_397 - poly_1372 - poly_1728 
    poly_1773 = poly_2 * poly_743 - poly_1370 - poly_1360 - poly_1344 - poly_1707 - poly_1725 
    poly_1774 = poly_2 * poly_744 - poly_1361 - poly_1369 - poly_1345 - poly_1726 - poly_1706 
    poly_1775 = poly_1 * poly_961 - poly_1763 
    poly_1776 = poly_2 * poly_746 - poly_1372 - poly_1347 - poly_1728 
    poly_1777 = poly_39 * poly_79 - poly_1368 - poly_1687 
    poly_1778 = poly_18 * poly_181 - poly_1366 - poly_1356 - poly_1777 
    poly_1779 = poly_1 * poly_964 - poly_1766 - poly_1765 
    poly_1780 = poly_18 * poly_183 - poly_1368 
    poly_1781 = poly_20 * poly_179 - poly_1338 
    poly_1782 = poly_5 * poly_400 - poly_1740 - poly_1741 
    poly_1783 = poly_29 * poly_95 - poly_1353 - poly_1349 - poly_1781 
    poly_1784 = poly_5 * poly_402 - poly_1743 
    poly_1785 = poly_29 * poly_96 - poly_1362 - poly_1765 - poly_1784 
    poly_1786 = poly_9 * poly_283 - poly_1322 - poly_1282 - poly_1747 - poly_1714 - poly_1735 
    poly_1787 = poly_9 * poly_284 - poly_1324 - poly_1748 - poly_1717 - poly_1742 - poly_1690 - poly_1324 
    poly_1788 = poly_9 * poly_285 - poly_1334 - poly_1267 - poly_1745 - poly_1731 - poly_1736 
    poly_1789 = poly_2 * poly_759 - poly_1364 - poly_1748 
    poly_1790 = poly_46 * poly_77 - poly_1338 - poly_1646 
    poly_1791 = poly_5 * poly_404 - poly_1753 
    poly_1792 = poly_20 * poly_183 - poly_1353 
    poly_1793 = poly_1 * poly_974 - poly_1788 - poly_1786 - poly_1785 
    poly_1794 = poly_30 * poly_30 
    poly_1795 = poly_30 * poly_47 
    poly_1796 = poly_30 * poly_69 
    poly_1797 = poly_30 * poly_70 
    poly_1798 = poly_30 * poly_81 
    poly_1799 = poly_10 * poly_150 - poly_1374 
    poly_1800 = poly_10 * poly_151 - poly_1375 - poly_1374 
    poly_1801 = jnp.take(mono,1905) + jnp.take(mono,1906) + jnp.take(mono,1907) + jnp.take(mono,1908) + jnp.take(mono,1909) + jnp.take(mono,1910) + jnp.take(mono,1911) + jnp.take(mono,1912) + jnp.take(mono,1913) + jnp.take(mono,1914) + jnp.take(mono,1915) + jnp.take(mono,1916) + jnp.take(mono,1917) + jnp.take(mono,1918) + jnp.take(mono,1919) + jnp.take(mono,1920) + jnp.take(mono,1921) + jnp.take(mono,1922) + jnp.take(mono,1923) + jnp.take(mono,1924) + jnp.take(mono,1925) + jnp.take(mono,1926) + jnp.take(mono,1927) + jnp.take(mono,1928) 
    poly_1802 = poly_10 * poly_194 - poly_1375 - poly_1801 
    poly_1803 = poly_11 * poly_148 - poly_1378 - poly_1377 - poly_1799 
    poly_1804 = poly_1 * poly_768 - poly_1375 - poly_1801 - poly_1800 - poly_1375 
    poly_1805 = poly_1 * poly_769 - poly_1375 - poly_1802 
    poly_1806 = poly_4 * poly_405 - poly_1378 - poly_1803 
    poly_1807 = poly_1 * poly_770 - poly_1376 - poly_1803 
    poly_1808 = poly_1 * poly_771 - poly_1377 - poly_1804 - poly_1803 
    poly_1809 = poly_3 * poly_297 - poly_1153 - poly_1380 - poly_1382 
    poly_1810 = poly_8 * poly_295 - poly_1374 - poly_1802 
    poly_1811 = poly_1 * poly_773 - poly_1378 - poly_1806 
    poly_1812 = poly_10 * poly_154 - poly_1383 
    poly_1813 = poly_5 * poly_474 - poly_1809 
    poly_1814 = poly_3 * poly_301 - poly_1386 - poly_1381 - poly_1807 
    poly_1815 = poly_8 * poly_297 - poly_1379 - poly_1379 
    poly_1816 = poly_28 * poly_103 - poly_1374 - poly_1802 
    poly_1817 = poly_8 * poly_299 - poly_1378 - poly_1806 
    poly_1818 = poly_5 * poly_475 - poly_1815 
    poly_1819 = poly_1 * poly_779 - poly_1386 - poly_1814 
    poly_1820 = poly_32 * poly_81 - poly_1387 
    poly_1821 = poly_10 * poly_195 - poly_1386 - poly_1819 
    poly_1822 = poly_5 * poly_476 - poly_1820 
    poly_1823 = jnp.take(mono,1929) + jnp.take(mono,1930) + jnp.take(mono,1931) + jnp.take(mono,1932) + jnp.take(mono,1933) + jnp.take(mono,1934) + jnp.take(mono,1935) + jnp.take(mono,1936) + jnp.take(mono,1937) + jnp.take(mono,1938) + jnp.take(mono,1939) + jnp.take(mono,1940) + jnp.take(mono,1941) + jnp.take(mono,1942) + jnp.take(mono,1943) + jnp.take(mono,1944) + jnp.take(mono,1945) + jnp.take(mono,1946) + jnp.take(mono,1947) + jnp.take(mono,1948) + jnp.take(mono,1949) + jnp.take(mono,1950) + jnp.take(mono,1951) + jnp.take(mono,1952) 
    poly_1824 = poly_1 * poly_783 - poly_1396 - poly_1388 - poly_1823 - poly_1388 
    poly_1825 = poly_1 * poly_784 - poly_1397 - poly_1389 - poly_1824 
    poly_1826 = poly_3 * poly_307 - poly_1158 - poly_1399 - poly_1392 
    poly_1827 = poly_1 * poly_785 - poly_1398 - poly_1391 - poly_1825 
    poly_1828 = poly_8 * poly_307 - poly_1407 - poly_1390 
    poly_1829 = poly_1 * poly_787 - poly_1400 - poly_1393 - poly_1827 
    poly_1830 = poly_36 * poly_81 - poly_1422 - poly_1388 
    poly_1831 = poly_1 * poly_789 - poly_1402 - poly_1403 - poly_1395 
    poly_1832 = poly_1 * poly_790 - poly_1404 
    poly_1833 = poly_3 * poly_417 - poly_1396 - poly_1388 - poly_1823 
    poly_1834 = poly_3 * poly_418 - poly_1401 - poly_1400 - poly_1393 - poly_1830 - poly_1829 
    poly_1835 = poly_1 * poly_791 - poly_1396 - poly_1833 
    poly_1836 = poly_1 * poly_792 - poly_1401 - poly_1400 - poly_1834 
    poly_1837 = poly_1 * poly_793 - poly_1405 - poly_1835 
    poly_1838 = poly_1 * poly_794 - poly_1409 - poly_1408 - poly_1836 
    poly_1839 = poly_1 * poly_795 - poly_1413 - poly_1837 
    poly_1840 = poly_1 * poly_796 - poly_1415 - poly_1414 - poly_1838 
    poly_1841 = poly_1 * poly_797 - poly_1421 - poly_1839 
    poly_1842 = poly_3 * poly_419 - poly_1422 - poly_1421 - poly_1841 
    poly_1843 = poly_17 * poly_206 - poly_1831 
    poly_1844 = poly_18 * poly_206 - poly_1832 
    poly_1845 = poly_1 * poly_801 - poly_1433 - poly_1430 
    poly_1846 = poly_1 * poly_802 - poly_1436 
    poly_1847 = poly_1 * poly_803 - poly_1438 
    poly_1848 = poly_5 * poly_409 - poly_1395 
    poly_1849 = poly_10 * poly_161 - poly_1441 - poly_1440 - poly_1440 
    poly_1850 = poly_11 * poly_152 - poly_1159 - poly_1392 - poly_1396 - poly_1380 - poly_1824 
    poly_1851 = poly_5 * poly_410 - poly_1402 - poly_1403 
    poly_1852 = poly_10 * poly_196 - poly_1441 - poly_1850 
    poly_1853 = poly_1 * poly_805 - poly_1440 - poly_1849 - poly_1440 
    poly_1854 = poly_1 * poly_806 - poly_1441 - poly_1850 - poly_1849 - poly_1441 
    poly_1855 = poly_5 * poly_411 - poly_1410 - poly_1411 
    poly_1856 = poly_5 * poly_297 - poly_1169 - poly_1173 - poly_1169 
    poly_1857 = poly_1 * poly_808 - poly_1441 - poly_1852 
    poly_1858 = poly_9 * poly_405 - poly_1853 
    poly_1859 = poly_5 * poly_300 - poly_1173 
    poly_1860 = poly_3 * poly_328 - poly_1451 - poly_1445 - poly_1853 
    poly_1861 = poly_5 * poly_412 - poly_1416 - poly_1417 
    poly_1862 = poly_8 * poly_325 - poly_1447 
    poly_1863 = poly_8 * poly_326 - poly_1440 - poly_1852 - poly_1440 
    poly_1864 = poly_5 * poly_304 - poly_1182 
    poly_1865 = poly_1 * poly_814 - poly_1448 - poly_1858 - poly_1448 
    poly_1866 = poly_1 * poly_815 - poly_1449 
    poly_1867 = poly_1 * poly_816 - poly_1451 - poly_1860 
    poly_1868 = poly_5 * poly_414 - poly_1423 - poly_1424 
    poly_1869 = poly_10 * poly_197 - poly_1451 - poly_1867 
    poly_1870 = poly_5 * poly_416 - poly_1426 
    poly_1871 = poly_11 * poly_158 - poly_1186 - poly_1160 - poly_1404 
    poly_1872 = poly_1 * poly_820 - poly_1477 - poly_1456 - poly_1871 
    poly_1873 = poly_1 * poly_821 - poly_1478 - poly_1457 - poly_1872 
    poly_1874 = poly_1 * poly_822 - poly_1479 - poly_1458 - poly_1873 
    poly_1875 = poly_5 * poly_417 - poly_1431 
    poly_1876 = poly_1 * poly_824 - poly_1481 - poly_1460 - poly_1455 
    poly_1877 = poly_2 * poly_783 - poly_1404 - poly_1430 - poly_1403 - poly_1395 - poly_1871 - poly_1404 
    poly_1878 = poly_36 * poly_71 - poly_1184 - poly_1164 - poly_1181 - poly_1456 - poly_1430 - poly_1402 
    poly_1879 = poly_1 * poly_825 - poly_1483 - poly_1456 - poly_1877 
    poly_1880 = poly_1 * poly_826 - poly_1484 - poly_1458 - poly_1878 
    poly_1881 = poly_1 * poly_827 - poly_1485 - poly_1461 - poly_1879 
    poly_1882 = poly_1 * poly_828 - poly_1486 - poly_1462 - poly_1880 
    poly_1883 = poly_1 * poly_829 - poly_1482 - poly_1487 - poly_1465 
    poly_1884 = poly_1 * poly_830 - poly_1488 - poly_1466 - poly_1881 
    poly_1885 = poly_1 * poly_831 - poly_1489 - poly_1466 - poly_1882 
    poly_1886 = poly_5 * poly_418 - poly_1432 - poly_1876 
    poly_1887 = poly_1 * poly_833 - poly_1491 - poly_1492 - poly_1470 
    poly_1888 = poly_2 * poly_790 - poly_1431 
    poly_1889 = poly_1 * poly_835 - poly_1494 - poly_1474 
    poly_1890 = poly_2 * poly_791 - poly_1404 - poly_1433 - poly_1402 
    poly_1891 = poly_3 * poly_431 - poly_1484 - poly_1479 - poly_1458 - poly_1878 - poly_1874 
    poly_1892 = poly_3 * poly_432 - poly_1489 - poly_1488 - poly_1466 - poly_1885 - poly_1884 
    poly_1893 = poly_1 * poly_836 - poly_1483 - poly_1477 - poly_1890 
    poly_1894 = poly_1 * poly_837 - poly_1484 - poly_1479 - poly_1891 
    poly_1895 = poly_1 * poly_838 - poly_1489 - poly_1488 - poly_1892 
    poly_1896 = poly_1 * poly_839 - poly_1501 - poly_1495 - poly_1893 
    poly_1897 = poly_1 * poly_840 - poly_1502 - poly_1496 - poly_1894 
    poly_1898 = poly_28 * poly_170 - poly_1883 
    poly_1899 = poly_1 * poly_842 - poly_1505 - poly_1504 - poly_1895 
    poly_1900 = poly_1 * poly_843 - poly_1517 - poly_1514 - poly_1896 
    poly_1901 = poly_1 * poly_844 - poly_1518 - poly_1514 - poly_1897 
    poly_1902 = poly_5 * poly_419 - poly_1437 
    poly_1903 = poly_1 * poly_846 - poly_1518 - poly_1517 - poly_1899 
    poly_1904 = poly_65 * poly_81 - poly_1887 
    poly_1905 = poly_18 * poly_195 - poly_1437 - poly_1431 
    poly_1906 = poly_20 * poly_194 - poly_1437 - poly_1431 
    poly_1907 = poly_1 * poly_850 - poly_1541 - poly_1535 
    poly_1908 = poly_2 * poly_802 - poly_1437 
    poly_1909 = poly_7 * poly_305 - poly_1191 - poly_1190 - poly_1195 - poly_1187 - poly_1182 - poly_1493 - poly_1472 - poly_1532 - poly_1492 - poly_1529 - poly_1470 - poly_1195 - poly_1472 
    poly_1910 = poly_1 * poly_852 - poly_1567 - poly_1549 - poly_1909 - poly_1549 
    poly_1911 = poly_3 * poly_361 - poly_1302 - poly_1569 - poly_1550 
    poly_1912 = poly_1 * poly_853 - poly_1568 - poly_1550 - poly_1910 
    poly_1913 = poly_8 * poly_361 - poly_1582 - poly_1549 
    poly_1914 = poly_1 * poly_855 - poly_1554 - poly_1570 - poly_1551 
    poly_1915 = poly_1 * poly_856 - poly_1555 - poly_1571 - poly_1552 
    poly_1916 = poly_1 * poly_857 - poly_1572 - poly_1557 - poly_1556 
    poly_1917 = poly_5 * poly_342 - poly_1271 - poly_1641 - poly_1271 
    poly_1918 = poly_38 * poly_38 - poly_1272 - poly_1272 
    poly_1919 = poly_1 * poly_860 - poly_1577 - poly_1561 
    poly_1920 = poly_1 * poly_861 - poly_1578 - poly_1563 - poly_1562 
    poly_1921 = poly_1 * poly_862 - poly_1579 
    poly_1922 = poly_3 * poly_434 - poly_1567 - poly_1549 - poly_1909 
    poly_1923 = poly_3 * poly_435 - poly_1569 - poly_1568 - poly_1550 - poly_1913 - poly_1912 
    poly_1924 = poly_1 * poly_863 - poly_1567 - poly_1922 
    poly_1925 = poly_1 * poly_864 - poly_1569 - poly_1568 - poly_1923 
    poly_1926 = poly_8 * poly_436 - poly_1917 
    poly_1927 = poly_8 * poly_437 - poly_1921 
    poly_1928 = poly_1 * poly_867 - poly_1581 - poly_1924 
    poly_1929 = poly_3 * poly_438 - poly_1582 - poly_1581 - poly_1928 
    poly_1930 = poly_1 * poly_869 - poly_1586 - poly_1584 - poly_1583 
    poly_1931 = poly_1 * poly_870 - poly_1588 - poly_1587 
    poly_1932 = poly_1 * poly_871 - poly_1606 - poly_1590 
    poly_1933 = poly_1 * poly_872 - poly_1607 - poly_1592 - poly_1591 
    poly_1934 = poly_1 * poly_875 - poly_1611 - poly_1610 
    poly_1935 = poly_1 * poly_876 - poly_1618 - poly_1614 - poly_1614 
    poly_1936 = poly_5 * poly_376 - poly_1746 
    poly_1937 = poly_1 * poly_878 - poly_1618 
    poly_1938 = poly_5 * poly_420 - poly_1469 
    poly_1939 = poly_5 * poly_421 - poly_1470 
    poly_1940 = poly_1 * poly_880 - poly_1621 
    poly_1941 = poly_10 * poly_179 - poly_1622 
    poly_1942 = poly_2 * poly_806 - poly_1479 - poly_1458 - poly_1477 - poly_1456 - poly_1439 
    poly_1943 = poly_5 * poly_423 - poly_1491 - poly_1492 
    poly_1944 = poly_5 * poly_324 - poly_1226 - poly_1227 
    poly_1945 = poly_5 * poly_325 - poly_1228 - poly_1229 
    poly_1946 = poly_10 * poly_198 - poly_1622 - poly_1942 
    poly_1947 = poly_3 * poly_482 - poly_1945 
    poly_1948 = poly_3 * poly_379 - poly_1626 - poly_1622 - poly_1941 - poly_1622 
    poly_1949 = poly_5 * poly_424 - poly_1507 - poly_1508 
    poly_1950 = poly_5 * poly_329 - poly_1243 - poly_1244 
    poly_1951 = poly_1 * poly_884 - poly_1622 - poly_1946 
    poly_1952 = poly_1 * poly_885 - poly_1624 
    poly_1953 = poly_10 * poly_183 - poly_1626 
    poly_1954 = poly_5 * poly_426 - poly_1510 
    poly_1955 = poly_1 * poly_886 - poly_1626 - poly_1948 
    poly_1956 = poly_5 * poly_427 - poly_1520 - poly_1521 
    poly_1957 = poly_8 * poly_381 - poly_1622 - poly_1946 
    poly_1958 = poly_5 * poly_429 - poly_1523 
    poly_1959 = poly_53 * poly_56 - poly_1259 - poly_1209 - poly_1472 
    poly_1960 = poly_1 * poly_890 - poly_1649 - poly_1631 - poly_1959 
    poly_1961 = poly_1 * poly_891 - poly_1650 - poly_1632 - poly_1960 
    poly_1962 = poly_5 * poly_430 - poly_1537 - poly_1914 
    poly_1963 = poly_5 * poly_431 - poly_1538 - poly_1915 
    poly_1964 = poly_53 * poly_74 - poly_1534 
    poly_1965 = poly_5 * poly_335 - poly_1264 - poly_1558 
    poly_1966 = poly_2 * poly_825 - poly_1493 - poly_1532 - poly_1492 - poly_1459 - poly_1909 
    poly_1967 = poly_2 * poly_826 - poly_1494 - poly_1532 - poly_1491 - poly_1460 - poly_1913 
    poly_1968 = poly_5 * poly_338 - poly_1267 - poly_1560 
    poly_1969 = poly_1 * poly_896 - poly_1655 - poly_1631 - poly_1966 
    poly_1970 = poly_1 * poly_897 - poly_1656 - poly_1632 - poly_1967 
    poly_1971 = poly_5 * poly_341 - poly_1270 - poly_1558 
    poly_1972 = poly_1 * poly_899 - poly_1658 - poly_1637 - poly_1969 
    poly_1973 = poly_1 * poly_900 - poly_1659 - poly_1637 - poly_1970 
    poly_1974 = poly_5 * poly_432 - poly_1539 - poly_1916 
    poly_1975 = poly_1 * poly_902 - poly_1661 - poly_1662 - poly_1640 
    poly_1976 = poly_38 * poly_74 - poly_1272 
    poly_1977 = poly_9 * poly_311 - poly_1536 
    poly_1978 = poly_1 * poly_905 - poly_1665 - poly_1644 
    poly_1979 = poly_9 * poly_312 - poly_1472 - poly_1541 - poly_1468 
    poly_1980 = poly_2 * poly_837 - poly_1494 - poly_1540 - poly_1492 - poly_1481 - poly_1923 
    poly_1981 = poly_1 * poly_1027 - poly_1968 
    poly_1982 = poly_3 * poly_451 - poly_1659 - poly_1658 - poly_1637 - poly_1973 - poly_1972 
    poly_1983 = poly_1 * poly_907 - poly_1655 - poly_1649 - poly_1979 
    poly_1984 = poly_1 * poly_908 - poly_1656 - poly_1650 - poly_1980 
    poly_1985 = poly_55 * poly_74 - poly_1543 
    poly_1986 = poly_1 * poly_910 - poly_1659 - poly_1658 - poly_1982 
    poly_1987 = poly_1 * poly_911 - poly_1670 - poly_1667 - poly_1983 
    poly_1988 = poly_1 * poly_912 - poly_1671 - poly_1667 - poly_1984 
    poly_1989 = poly_5 * poly_433 - poly_1548 - poly_1930 
    poly_1990 = poly_1 * poly_914 - poly_1671 - poly_1670 - poly_1986 
    poly_1991 = poly_28 * poly_188 - poly_1975 
    poly_1992 = poly_18 * poly_197 - poly_1546 - poly_1536 
    poly_1993 = poly_20 * poly_196 - poly_1476 - poly_1475 
    poly_1994 = poly_5 * poly_357 - poly_1598 
    poly_1995 = poly_1 * poly_919 - poly_1694 - poly_1686 
    poly_1996 = poly_9 * poly_321 - poly_1476 
    poly_1997 = poly_11 * poly_173 - poly_1314 - poly_1303 - poly_1577 
    poly_1998 = poly_1 * poly_921 - poly_1719 - poly_1700 - poly_1997 
    poly_1999 = poly_5 * poly_434 - poly_1600 
    poly_2000 = poly_1 * poly_923 - poly_1721 - poly_1702 - poly_1699 
    poly_2001 = poly_5 * poly_362 - poly_1322 - poly_1704 
    poly_2002 = poly_5 * poly_365 - poly_1325 - poly_1708 
    poly_2003 = poly_2 * poly_852 - poly_1561 - poly_1590 - poly_1554 - poly_1551 - poly_1997 
    poly_2004 = poly_4 * poly_361 - poly_1309 - poly_1296 - poly_1306 - poly_1700 - poly_1592 - poly_1555 
    poly_2005 = poly_5 * poly_363 - poly_1323 - poly_1705 
    poly_2006 = poly_1 * poly_924 - poly_1722 - poly_1700 - poly_2003 
    poly_2007 = poly_1 * poly_925 - poly_1723 - poly_1700 - poly_2004 
    poly_2008 = poly_5 * poly_435 - poly_1601 - poly_2000 
    poly_2009 = poly_1 * poly_927 - poly_1706 - poly_1725 - poly_1703 
    poly_2010 = poly_1 * poly_928 - poly_1707 - poly_1726 - poly_1703 
    poly_2011 = poly_5 * poly_364 - poly_1324 
    poly_2012 = poly_1 * poly_930 - poly_1728 - poly_1707 - poly_1706 
    poly_2013 = poly_5 * poly_389 - poly_1363 - poly_1763 - poly_1363 
    poly_2014 = poly_38 * poly_65 - poly_1364 - poly_1602 - poly_1364 
    poly_2015 = poly_1 * poly_931 - poly_1729 - poly_1712 - poly_1709 
    poly_2016 = poly_1 * poly_932 - poly_1730 - poly_1713 - poly_1709 
    poly_2017 = poly_18 * poly_170 - poly_1690 
    poly_2018 = poly_1 * poly_934 - poly_1732 - poly_1713 - poly_1712 
    poly_2019 = poly_2 * poly_862 - poly_1603 
    poly_2020 = poly_4 * poly_437 - poly_1604 - poly_2019 
    poly_2021 = poly_5 * poly_437 
    poly_2022 = poly_2 * poly_863 - poly_1577 - poly_1606 - poly_1570 
    poly_2023 = poly_3 * poly_454 - poly_1723 - poly_1719 - poly_1700 - poly_2004 - poly_1998 
    poly_2024 = poly_5 * poly_371 - poly_1334 - poly_1727 
    poly_2025 = poly_2 * poly_864 - poly_1578 - poly_1607 - poly_1572 - poly_1571 - poly_2023 
    poly_2026 = poly_1 * poly_1034 - poly_2013 
    poly_2027 = poly_1 * poly_1035 - poly_2020 - poly_2019 
    poly_2028 = poly_1 * poly_936 - poly_1722 - poly_1719 - poly_2022 
    poly_2029 = poly_1 * poly_937 - poly_1723 - poly_1719 - poly_2023 
    poly_2030 = poly_5 * poly_438 - poly_1613 
    poly_2031 = poly_1 * poly_939 - poly_1723 - poly_1722 - poly_2025 
    poly_2032 = poly_1 * poly_940 - poly_1728 - poly_1726 - poly_1725 
    poly_2033 = poly_1 * poly_941 - poly_1732 - poly_1730 - poly_1729 
    poly_2034 = poly_1 * poly_942 - poly_1749 - poly_1737 - poly_1734 
    poly_2035 = poly_1 * poly_943 - poly_1750 - poly_1738 - poly_1734 
    poly_2036 = poly_20 * poly_170 - poly_1690 
    poly_2037 = poly_1 * poly_945 - poly_1752 - poly_1738 - poly_1737 
    poly_2038 = poly_5 * poly_403 - poly_1748 - poly_1787 
    poly_2039 = poly_2 * poly_874 - poly_1605 - poly_2021 - poly_1936 - poly_1605 
    poly_2040 = poly_1 * poly_948 - poly_1752 - poly_1750 - poly_1749 
    poly_2041 = poly_4 * poly_439 - poly_1619 - poly_1615 
    poly_2042 = poly_5 * poly_439 
    poly_2043 = poly_20 * poly_145 - poly_1373 - poly_1323 
    poly_2044 = poly_2 * poly_877 - poly_1617 - poly_1936 - poly_2042 - poly_1617 
    poly_2045 = poly_2 * poly_878 - poly_1619 
    poly_2046 = poly_7 * poly_360 - poly_1299 - poly_1297 - poly_1317 - poly_1315 - poly_1294 - poly_1712 - poly_1709 - poly_1737 - poly_1734 - poly_1706 - poly_1703 
    poly_2047 = poly_7 * poly_361 - poly_1300 - poly_1318 - poly_1295 - poly_1713 - poly_1738 - poly_1707 
    poly_2048 = poly_17 * poly_173 - poly_1322 - poly_1744 - poly_1710 - poly_1735 - poly_1704 
    poly_2049 = poly_7 * poly_363 - poly_1301 - poly_1319 - poly_1716 - poly_1714 - poly_1596 - poly_1741 - poly_1597 - poly_1739 - poly_1708 - poly_1560 - poly_2013 
    poly_2050 = poly_38 * poly_76 - poly_1746 
    poly_2051 = poly_7 * poly_365 - poly_1301 - poly_1319 - poly_1718 - poly_1711 - poly_1743 - poly_1608 - poly_1599 - poly_1736 - poly_1705 - poly_1575 - poly_2026 
    poly_2052 = poly_18 * poly_173 - poly_1298 - poly_1740 
    poly_2053 = poly_18 * poly_174 - poly_1301 - poly_1743 - poly_1741 
    poly_2054 = poly_18 * poly_175 - poly_1742 - poly_1598 - poly_2038 
    poly_2055 = poly_7 * poly_369 - poly_1310 - poly_1328 - poly_1304 - poly_1729 - poly_1749 - poly_1725 
    poly_2056 = poly_3 * poly_484 - poly_2047 - poly_2055 - poly_2046 
    poly_2057 = poly_1 * poly_1040 - poly_2051 - poly_2049 - poly_2048 
    poly_2058 = poly_1 * poly_1041 - poly_2053 - poly_2052 
    poly_2059 = poly_20 * poly_173 - poly_1316 - poly_1715 
    poly_2060 = poly_20 * poly_174 - poly_1319 - poly_1718 - poly_1716 
    poly_2061 = poly_20 * poly_175 - poly_1717 - poly_1598 - poly_2038 
    poly_2062 = poly_18 * poly_178 - poly_1324 - poly_2044 
    poly_2063 = poly_1 * poly_1043 - poly_2060 - poly_2059 
    poly_2064 = poly_1 * poly_949 - poly_1755 
    poly_2065 = poly_5 * poly_378 - poly_1337 
    poly_2066 = poly_2 * poly_881 - poly_1650 - poly_1632 - poly_1649 - poly_1631 - poly_1620 
    poly_2067 = poly_5 * poly_443 - poly_1661 - poly_1662 
    poly_2068 = poly_5 * poly_380 - poly_1344 - poly_1345 
    poly_2069 = poly_2 * poly_884 - poly_1656 - poly_1655 - poly_1623 
    poly_2070 = poly_5 * poly_382 - poly_1347 
    poly_2071 = poly_1 * poly_950 - poly_1756 - poly_2066 - poly_1756 
    poly_2072 = poly_5 * poly_445 - poly_1673 - poly_1674 
    poly_2073 = poly_1 * poly_952 - poly_1756 - poly_2069 - poly_1756 
    poly_2074 = poly_1 * poly_953 - poly_1757 
    poly_2075 = poly_15 * poly_179 - poly_1355 - poly_1336 - poly_1643 
    poly_2076 = poly_1 * poly_954 - poly_1767 - poly_1759 - poly_2075 
    poly_2077 = poly_5 * poly_448 - poly_1688 - poly_2009 
    poly_2078 = poly_5 * poly_449 - poly_1689 - poly_2010 
    poly_2079 = poly_5 * poly_384 - poly_1358 - poly_1704 
    poly_2080 = poly_5 * poly_385 - poly_1359 - poly_1705 
    poly_2081 = poly_2 * poly_896 - poly_1664 - poly_1684 - poly_1662 - poly_1633 - poly_2003 
    poly_2082 = poly_2 * poly_897 - poly_1665 - poly_1684 - poly_1661 - poly_1634 - poly_2004 
    poly_2083 = poly_5 * poly_388 - poly_1362 - poly_1708 
    poly_2084 = poly_1 * poly_957 - poly_1770 - poly_1759 - poly_2081 
    poly_2085 = poly_1 * poly_958 - poly_1771 - poly_1759 - poly_2082 
    poly_2086 = poly_5 * poly_451 - poly_1691 - poly_2012 
    poly_2087 = poly_1 * poly_960 - poly_1773 - poly_1774 - poly_1762 - poly_1762 
    poly_2088 = poly_38 * poly_78 - poly_1364 
    poly_2089 = poly_29 * poly_112 - poly_1687 
    poly_2090 = poly_1 * poly_963 - poly_1778 - poly_1764 
    poly_2091 = poly_5 * poly_393 - poly_1717 
    poly_2092 = poly_29 * poly_113 - poly_1643 - poly_1694 - poly_1639 
    poly_2093 = poly_2 * poly_908 - poly_1665 - poly_1692 - poly_1662 - poly_1652 - poly_2023 
    poly_2094 = poly_5 * poly_396 - poly_1371 - poly_1727 
    poly_2095 = poly_3 * poly_469 - poly_1771 - poly_1770 - poly_1759 - poly_2085 - poly_2084 
    poly_2096 = poly_1 * poly_965 - poly_1770 - poly_1767 - poly_2092 
    poly_2097 = poly_1 * poly_966 - poly_1771 - poly_1767 - poly_2093 
    poly_2098 = poly_5 * poly_452 - poly_1698 - poly_2032 
    poly_2099 = poly_1 * poly_968 - poly_1771 - poly_1770 - poly_2095 
    poly_2100 = poly_8 * poly_470 - poly_2087 
    poly_2101 = poly_8 * poly_471 - poly_2090 - poly_2089 
    poly_2102 = poly_20 * poly_198 - poly_1646 - poly_1645 
    poly_2103 = poly_5 * poly_401 - poly_1742 
    poly_2104 = poly_1 * poly_973 - poly_1792 - poly_1783 
    poly_2105 = poly_29 * poly_120 - poly_1646 
    poly_2106 = poly_56 * poly_76 - poly_1579 - poly_1614 - poly_1574 
    poly_2107 = poly_5 * poly_453 - poly_1744 - poly_2048 
    poly_2108 = poly_5 * poly_454 - poly_1745 - poly_2049 
    poly_2109 = poly_2 * poly_924 - poly_1712 - poly_1737 - poly_1706 - poly_1701 - poly_2046 
    poly_2110 = poly_9 * poly_361 - poly_1580 - poly_1614 - poly_1573 
    poly_2111 = poly_5 * poly_456 - poly_1747 - poly_2051 
    poly_2112 = poly_2 * poly_927 - poly_1715 - poly_1740 - poly_1704 - poly_2048 - poly_2013 
    poly_2113 = poly_2 * poly_928 - poly_1716 - poly_1741 - poly_1705 - poly_2049 - poly_2013 
    poly_2114 = poly_38 * poly_80 - poly_1748 - poly_1787 
    poly_2115 = poly_2 * poly_930 - poly_1718 - poly_1743 - poly_1708 - poly_2051 - poly_2026 
    poly_2116 = poly_9 * poly_366 - poly_1615 - poly_1597 - poly_2019 
    poly_2117 = poly_18 * poly_185 - poly_1359 - poly_1785 - poly_1736 
    poly_2118 = poly_5 * poly_458 - poly_2039 - poly_2054 
    poly_2119 = poly_18 * poly_187 - poly_1371 - poly_1788 - poly_1739 
    poly_2120 = poly_18 * poly_188 - poly_1742 
    poly_2121 = poly_9 * poly_369 - poly_1579 - poly_1618 - poly_1573 
    poly_2122 = poly_2 * poly_937 - poly_1730 - poly_1750 - poly_1726 - poly_1721 - poly_2056 
    poly_2123 = poly_5 * poly_459 - poly_1754 - poly_2057 
    poly_2124 = poly_60 * poly_76 - poly_1580 - poly_1618 - poly_1574 
    poly_2125 = poly_1 * poly_1061 - poly_2115 - poly_2113 - poly_2112 
    poly_2126 = poly_1 * poly_1062 - poly_2119 - poly_2117 - poly_2116 
    poly_2127 = poly_9 * poly_373 - poly_1603 - poly_1596 - poly_2043 
    poly_2128 = poly_20 * poly_185 - poly_1358 - poly_1786 - poly_1711 
    poly_2129 = poly_5 * poly_460 - poly_2044 - poly_2061 
    poly_2130 = poly_20 * poly_187 - poly_1362 - poly_1793 - poly_1714 
    poly_2131 = poly_20 * poly_188 - poly_1717 
    poly_2132 = poly_18 * poly_191 - poly_1364 - poly_2042 - poly_1364 
    poly_2133 = poly_1 * poly_1064 - poly_2130 - poly_2128 - poly_2127 
    poly_2134 = poly_5 * poly_461 - poly_1762 
    poly_2135 = poly_2 * poly_950 - poly_1767 - poly_1759 - poly_1755 
    poly_2136 = poly_5 * poly_463 - poly_1773 - poly_1774 
    poly_2137 = poly_10 * poly_207 - poly_2135 
    poly_2138 = poly_5 * poly_465 - poly_1776 
    poly_2139 = poly_7 * poly_461 - poly_1764 - poly_1781 - poly_1758 
    poly_2140 = poly_5 * poly_466 - poly_1785 - poly_2112 
    poly_2141 = poly_5 * poly_467 - poly_1786 - poly_2113 
    poly_2142 = poly_2 * poly_957 - poly_1777 - poly_1783 - poly_1774 - poly_1760 - poly_2109 
    poly_2143 = poly_36 * poly_82 - poly_1780 - poly_1781 - poly_1768 
    poly_2144 = poly_5 * poly_469 - poly_1788 - poly_2115 
    poly_2145 = poly_2 * poly_960 - poly_1786 - poly_1785 - poly_1763 - poly_2113 - poly_2112 
    poly_2146 = poly_38 * poly_82 - poly_1789 
    poly_2147 = poly_39 * poly_82 - poly_1784 
    poly_2148 = poly_18 * poly_202 - poly_1782 - poly_2147 
    poly_2149 = poly_5 * poly_471 - poly_2120 
    poly_2150 = poly_40 * poly_82 - poly_1764 - poly_1792 - poly_1761 
    poly_2151 = poly_2 * poly_966 - poly_1778 - poly_1790 - poly_1774 - poly_1769 - poly_2122 
    poly_2152 = poly_5 * poly_472 - poly_1793 - poly_2125 
    poly_2153 = poly_7 * poly_465 - poly_1780 - poly_1792 - poly_1772 
    poly_2154 = poly_1 * poly_1074 - poly_2145 
    poly_2155 = poly_1 * poly_1075 - poly_2148 - poly_2147 
    poly_2156 = poly_20 * poly_202 - poly_1766 - poly_1765 
    poly_2157 = poly_5 * poly_473 - poly_2131 
    poly_2158 = poly_2 * poly_973 - poly_1788 - poly_1784 - poly_2130 
    poly_2159 = poly_45 * poly_82 - poly_1787 - poly_2149 - poly_2157 
    poly_2160 = poly_46 * poly_82 - poly_1766 
    poly_2161 = poly_3 * poly_405 - poly_1795 
    poly_2162 = poly_1 * poly_976 - poly_1795 - poly_2161 - poly_2161 
    poly_2163 = poly_1 * poly_977 - poly_1796 - poly_2161 - poly_1796 
    poly_2164 = poly_28 * poly_148 - poly_1795 - poly_2162 
    poly_2165 = poly_10 * poly_206 - poly_1798 
    poly_2166 = poly_1 * poly_980 - poly_1799 
    poly_2167 = poly_28 * poly_150 - poly_1807 
    poly_2168 = poly_11 * poly_206 - poly_1819 
    poly_2169 = poly_3 * poly_477 - poly_1802 - poly_1801 - poly_2168 
    poly_2170 = poly_69 * poly_71 - poly_1806 - poly_1803 - poly_2167 
    poly_2171 = poly_1 * poly_984 - poly_1805 - poly_1804 - poly_2170 
    poly_2172 = poly_1 * poly_985 - poly_1810 - poly_1808 - poly_2171 
    poly_2173 = poly_1 * poly_986 - poly_1812 
    poly_2174 = poly_1 * poly_987 - poly_1816 - poly_1814 - poly_2172 
    poly_2175 = poly_28 * poly_154 - poly_1806 
    poly_2176 = poly_8 * poly_414 - poly_1817 - poly_1814 - poly_2172 
    poly_2177 = poly_5 * poly_493 
    poly_2178 = poly_14 * poly_206 - poly_1802 
    poly_2179 = poly_1 * poly_992 - poly_1833 - poly_1823 
    poly_2180 = poly_1 * poly_993 - poly_1834 - poly_1830 - poly_1829 
    poly_2181 = poly_1 * poly_994 - poly_1842 - poly_1841 
    poly_2182 = poly_2 * poly_980 - poly_1825 - poly_1813 
    poly_2183 = poly_8 * poly_420 - poly_1853 
    poly_2184 = poly_56 * poly_81 - poly_1867 
    poly_2185 = poly_5 * poly_477 - poly_1831 
    poly_2186 = poly_3 * poly_479 - poly_1852 - poly_1850 - poly_2184 
    poly_2187 = poly_1 * poly_998 - poly_1852 - poly_1850 - poly_2186 - poly_1852 
    poly_2188 = poly_1 * poly_999 - poly_1857 - poly_1854 - poly_2187 
    poly_2189 = poly_2 * poly_986 - poly_1838 - poly_1813 
    poly_2190 = poly_1 * poly_1000 - poly_1863 - poly_1860 - poly_2188 
    poly_2191 = poly_74 * poly_81 
    poly_2192 = poly_8 * poly_426 - poly_1858 
    poly_2193 = poly_8 * poly_427 - poly_1865 - poly_1860 - poly_2188 
    poly_2194 = poly_5 * poly_478 - poly_1843 
    poly_2195 = poly_60 * poly_81 - poly_1852 
    poly_2196 = poly_1 * poly_1006 - poly_1890 - poly_1877 - poly_1871 
    poly_2197 = poly_1 * poly_1007 - poly_1891 - poly_1878 - poly_1874 
    poly_2198 = poly_1 * poly_1008 - poly_1892 - poly_1885 - poly_1884 
    poly_2199 = poly_1 * poly_1009 - poly_1903 - poly_1901 - poly_1900 
    poly_2200 = poly_1 * poly_1010 - poly_1922 - poly_1909 
    poly_2201 = poly_1 * poly_1011 - poly_1923 - poly_1913 - poly_1912 
    poly_2202 = poly_1 * poly_1012 - poly_1929 - poly_1928 
    poly_2203 = poly_1 * poly_1013 - poly_1941 
    poly_2204 = poly_28 * poly_179 - poly_1955 
    poly_2205 = poly_5 * poly_479 - poly_1887 
    poly_2206 = poly_5 * poly_422 - poly_1471 
    poly_2207 = poly_3 * poly_481 - poly_1946 - poly_1942 - poly_2204 
    poly_2208 = poly_69 * poly_77 - poly_1953 - poly_1948 - poly_2203 
    poly_2209 = poly_8 * poly_482 
    poly_2210 = poly_1 * poly_1019 - poly_1951 - poly_1948 - poly_2208 
    poly_2211 = poly_5 * poly_428 - poly_1522 
    poly_2212 = poly_1 * poly_1021 - poly_1953 
    poly_2213 = poly_28 * poly_181 - poly_1957 - poly_1942 - poly_2207 
    poly_2214 = poly_5 * poly_480 - poly_1904 
    poly_2215 = poly_28 * poly_183 - poly_1946 
    poly_2216 = poly_1 * poly_1025 - poly_1979 - poly_1966 - poly_1959 
    poly_2217 = poly_1 * poly_1026 - poly_1980 - poly_1967 - poly_1961 
    poly_2218 = poly_1 * poly_1028 - poly_1982 - poly_1973 - poly_1972 
    poly_2219 = poly_1 * poly_1029 - poly_1990 - poly_1988 - poly_1987 
    poly_2220 = poly_1 * poly_1030 - poly_2022 - poly_2003 - poly_1997 
    poly_2221 = poly_1 * poly_1031 - poly_2023 - poly_2004 - poly_1998 
    poly_2222 = poly_5 * poly_436 - poly_1602 - poly_2014 
    poly_2223 = poly_1 * poly_1033 - poly_2025 - poly_2007 - poly_2006 
    poly_2224 = poly_1 * poly_1036 - poly_2031 - poly_2029 - poly_2028 
    poly_2225 = poly_1 * poly_1038 - poly_2055 - poly_2046 
    poly_2226 = poly_1 * poly_1039 - poly_2056 - poly_2047 - poly_2046 - poly_2047 
    poly_2227 = poly_5 * poly_457 - poly_1748 - poly_2114 
    poly_2228 = poly_7 * poly_437 - poly_1605 - poly_2039 
    poly_2229 = poly_1 * poly_1042 - poly_2056 - poly_2055 
    poly_2230 = poly_7 * poly_439 - poly_1617 - poly_2044 
    poly_2231 = poly_2 * poly_1013 - poly_1960 - poly_1938 
    poly_2232 = poly_8 * poly_461 - poly_2071 
    poly_2233 = poly_5 * poly_481 - poly_1975 
    poly_2234 = poly_5 * poly_441 - poly_1641 
    poly_2235 = poly_4 * poly_482 - poly_1968 
    poly_2236 = poly_3 * poly_485 - poly_2069 - poly_2066 - poly_2232 
    poly_2237 = poly_1 * poly_1088 - poly_2235 
    poly_2238 = poly_1 * poly_1047 - poly_2069 - poly_2066 - poly_2236 - poly_2069 
    poly_2239 = poly_5 * poly_446 - poly_1675 
    poly_2240 = poly_2 * poly_1021 - poly_1986 - poly_1954 
    poly_2241 = poly_8 * poly_463 - poly_2073 - poly_2066 - poly_2236 
    poly_2242 = poly_5 * poly_483 - poly_1991 
    poly_2243 = poly_8 * poly_465 - poly_2069 
    poly_2244 = poly_1 * poly_1052 - poly_2092 - poly_2081 - poly_2075 
    poly_2245 = poly_1 * poly_1053 - poly_2093 - poly_2082 - poly_2076 
    poly_2246 = poly_7 * poly_482 - poly_1994 
    poly_2247 = poly_1 * poly_1055 - poly_2095 - poly_2085 - poly_2084 
    poly_2248 = poly_1 * poly_1056 - poly_2099 - poly_2097 - poly_2096 
    poly_2249 = poly_1 * poly_1057 - poly_2121 - poly_2109 - poly_2106 
    poly_2250 = poly_1 * poly_1058 - poly_2122 - poly_2110 - poly_2106 
    poly_2251 = poly_74 * poly_76 - poly_1936 
    poly_2252 = poly_1 * poly_1060 - poly_2124 - poly_2110 - poly_2109 
    poly_2253 = poly_2 * poly_1034 - poly_2038 - poly_2014 - poly_2227 
    poly_2254 = poly_9 * poly_437 - poly_1936 
    poly_2255 = poly_1 * poly_1063 - poly_2124 - poly_2122 - poly_2121 
    poly_2256 = poly_9 * poly_439 - poly_1936 
    poly_2257 = poly_2 * poly_1038 - poly_2052 - poly_2059 - poly_2048 
    poly_2258 = poly_4 * poly_484 - poly_2058 - poly_2052 - poly_2060 - poly_2057 - poly_2049 - poly_2257 
    poly_2259 = poly_5 * poly_484 - poly_2062 
    poly_2260 = poly_2 * poly_1039 - poly_2053 - poly_2060 - poly_2051 - poly_2049 - poly_2258 
    poly_2261 = poly_2 * poly_1040 - poly_2054 - poly_2061 - poly_2050 - poly_2259 - poly_2227 - poly_2050 
    poly_2262 = poly_2 * poly_1041 - poly_2062 - poly_2054 - poly_2228 
    poly_2263 = poly_1 * poly_1090 - poly_2260 - poly_2258 - poly_2257 
    poly_2264 = poly_2 * poly_1043 - poly_2062 - poly_2061 - poly_2230 
    poly_2265 = poly_1 * poly_1065 - poly_2135 
    poly_2266 = poly_5 * poly_485 - poly_2087 
    poly_2267 = poly_5 * poly_462 - poly_1763 
    poly_2268 = poly_3 * poly_489 - poly_2137 - poly_2135 - poly_2265 - poly_2135 
    poly_2269 = poly_1 * poly_1092 - poly_2267 
    poly_2270 = poly_1 * poly_1067 - poly_2137 - poly_2135 - poly_2268 - poly_2137 - poly_2135 
    poly_2271 = poly_5 * poly_487 - poly_2100 
    poly_2272 = poly_1 * poly_1069 - poly_2137 
    poly_2273 = poly_1 * poly_1070 - poly_2150 - poly_2142 - poly_2139 
    poly_2274 = poly_1 * poly_1071 - poly_2151 - poly_2143 - poly_2139 
    poly_2275 = poly_5 * poly_468 - poly_1787 - poly_2114 
    poly_2276 = poly_1 * poly_1073 - poly_2153 - poly_2143 - poly_2142 
    poly_2277 = poly_1 * poly_1076 - poly_2153 - poly_2151 - poly_2150 
    poly_2278 = poly_29 * poly_173 - poly_2019 - poly_2043 - poly_2005 
    poly_2279 = poly_2 * poly_1058 - poly_2117 - poly_2128 - poly_2113 - poly_2108 - poly_2258 
    poly_2280 = poly_5 * poly_488 - poly_2132 - poly_2261 
    poly_2281 = poly_2 * poly_1060 - poly_2119 - poly_2130 - poly_2115 - poly_2111 - poly_2260 
    poly_2282 = poly_7 * poly_470 - poly_1787 - poly_2159 - poly_2120 - poly_2131 - poly_2114 
    poly_2283 = poly_18 * poly_205 - poly_1787 - poly_2159 - poly_2129 
    poly_2284 = poly_1 * poly_1094 - poly_2281 - poly_2279 - poly_2278 
    poly_2285 = poly_20 * poly_205 - poly_1787 - poly_2159 - poly_2118 
    poly_2286 = poly_2 * poly_1065 - poly_2139 - poly_2134 
    poly_2287 = poly_5 * poly_489 - poly_2145 
    poly_2288 = poly_2 * poly_1067 - poly_2151 - poly_2143 - poly_2150 - poly_2142 - poly_2136 
    poly_2289 = poly_1 * poly_1096 - poly_2287 
    poly_2290 = poly_2 * poly_1069 - poly_2153 - poly_2138 
    poly_2291 = poly_15 * poly_207 - poly_2147 - poly_2158 - poly_2144 
    poly_2292 = poly_9 * poly_467 - poly_2117 - poly_2127 - poly_2112 - poly_2080 - poly_2258 
    poly_2293 = poly_5 * poly_492 - poly_2159 - poly_2282 
    poly_2294 = poly_9 * poly_469 - poly_2126 - poly_2130 - poly_2125 - poly_2083 - poly_2260 
    poly_2295 = poly_2 * poly_1074 - poly_2159 - poly_2146 - poly_2282 
    poly_2296 = poly_18 * poly_207 - poly_2157 
    poly_2297 = poly_1 * poly_1098 - poly_2294 - poly_2292 - poly_2291 
    poly_2298 = poly_20 * poly_207 - poly_2149 
    poly_2299 = poly_3 * poly_474 - poly_1796 - poly_2163 
    poly_2300 = poly_8 * poly_474 - poly_2161 
    poly_2301 = poly_69 * poly_81 - poly_2162 
    poly_2302 = poly_3 * poly_493 - poly_2165 
    poly_2303 = poly_1 * poly_1081 - poly_2169 - poly_2168 
    poly_2304 = poly_2 * poly_1099 - poly_2303 
    poly_2305 = poly_1 * poly_1083 - poly_2186 - poly_2184 
    poly_2306 = poly_9 * poly_493 - poly_2305 
    poly_2307 = poly_1 * poly_1085 - poly_2207 - poly_2204 
    poly_2308 = poly_29 * poly_206 - poly_2307 
    poly_2309 = poly_1 * poly_1087 - poly_2236 - poly_2232 
    poly_2310 = poly_5 * poly_482 - poly_1976 
    poly_2311 = poly_81 * poly_82 - poly_2309 
    poly_2312 = poly_7 * poly_484 - poly_2054 - poly_2061 - poly_2050 - poly_2262 - poly_2264 - poly_2261 
    poly_2313 = poly_1 * poly_1091 - poly_2268 - poly_2265 
    poly_2314 = poly_5 * poly_486 - poly_2088 
    poly_2315 = poly_28 * poly_207 - poly_2313 
    poly_2316 = poly_9 * poly_484 - poly_2228 - poly_2230 - poly_2227 
    poly_2317 = poly_1 * poly_1095 - poly_2288 - poly_2286 - poly_2286 
    poly_2318 = poly_5 * poly_490 - poly_2146 
    poly_2319 = poly_8 * poly_494 - poly_2317 
    poly_2320 = poly_76 * poly_82 - poly_2254 - poly_2256 - poly_2251 
    poly_2321 = poly_2 * poly_1095 - poly_2292 - poly_2291 - poly_2287 
    poly_2322 = poly_5 * poly_494 - poly_2295 
    poly_2323 = poly_1 * poly_1100 - poly_2321 
    poly_2324 = poly_7 * poly_494 - poly_2296 - poly_2298 - poly_2293 
    poly_2325 = poly_1 * poly_1099 - poly_2302 
    poly_2326 = poly_2 * poly_1100 - poly_2324 - poly_2322 

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
    poly_1101,    poly_1102,    poly_1103,    poly_1104,    poly_1105, 
    poly_1106,    poly_1107,    poly_1108,    poly_1109,    poly_1110, 
    poly_1111,    poly_1112,    poly_1113,    poly_1114,    poly_1115, 
    poly_1116,    poly_1117,    poly_1118,    poly_1119,    poly_1120, 
    poly_1121,    poly_1122,    poly_1123,    poly_1124,    poly_1125, 
    poly_1126,    poly_1127,    poly_1128,    poly_1129,    poly_1130, 
    poly_1131,    poly_1132,    poly_1133,    poly_1134,    poly_1135, 
    poly_1136,    poly_1137,    poly_1138,    poly_1139,    poly_1140, 
    poly_1141,    poly_1142,    poly_1143,    poly_1144,    poly_1145, 
    poly_1146,    poly_1147,    poly_1148,    poly_1149,    poly_1150, 
    poly_1151,    poly_1152,    poly_1153,    poly_1154,    poly_1155, 
    poly_1156,    poly_1157,    poly_1158,    poly_1159,    poly_1160, 
    poly_1161,    poly_1162,    poly_1163,    poly_1164,    poly_1165, 
    poly_1166,    poly_1167,    poly_1168,    poly_1169,    poly_1170, 
    poly_1171,    poly_1172,    poly_1173,    poly_1174,    poly_1175, 
    poly_1176,    poly_1177,    poly_1178,    poly_1179,    poly_1180, 
    poly_1181,    poly_1182,    poly_1183,    poly_1184,    poly_1185, 
    poly_1186,    poly_1187,    poly_1188,    poly_1189,    poly_1190, 
    poly_1191,    poly_1192,    poly_1193,    poly_1194,    poly_1195, 
    poly_1196,    poly_1197,    poly_1198,    poly_1199,    poly_1200, 
    poly_1201,    poly_1202,    poly_1203,    poly_1204,    poly_1205, 
    poly_1206,    poly_1207,    poly_1208,    poly_1209,    poly_1210, 
    poly_1211,    poly_1212,    poly_1213,    poly_1214,    poly_1215, 
    poly_1216,    poly_1217,    poly_1218,    poly_1219,    poly_1220, 
    poly_1221,    poly_1222,    poly_1223,    poly_1224,    poly_1225, 
    poly_1226,    poly_1227,    poly_1228,    poly_1229,    poly_1230, 
    poly_1231,    poly_1232,    poly_1233,    poly_1234,    poly_1235, 
    poly_1236,    poly_1237,    poly_1238,    poly_1239,    poly_1240, 
    poly_1241,    poly_1242,    poly_1243,    poly_1244,    poly_1245, 
    poly_1246,    poly_1247,    poly_1248,    poly_1249,    poly_1250, 
    poly_1251,    poly_1252,    poly_1253,    poly_1254,    poly_1255, 
    poly_1256,    poly_1257,    poly_1258,    poly_1259,    poly_1260, 
    poly_1261,    poly_1262,    poly_1263,    poly_1264,    poly_1265, 
    poly_1266,    poly_1267,    poly_1268,    poly_1269,    poly_1270, 
    poly_1271,    poly_1272,    poly_1273,    poly_1274,    poly_1275, 
    poly_1276,    poly_1277,    poly_1278,    poly_1279,    poly_1280, 
    poly_1281,    poly_1282,    poly_1283,    poly_1284,    poly_1285, 
    poly_1286,    poly_1287,    poly_1288,    poly_1289,    poly_1290, 
    poly_1291,    poly_1292,    poly_1293,    poly_1294,    poly_1295, 
    poly_1296,    poly_1297,    poly_1298,    poly_1299,    poly_1300, 
    poly_1301,    poly_1302,    poly_1303,    poly_1304,    poly_1305, 
    poly_1306,    poly_1307,    poly_1308,    poly_1309,    poly_1310, 
    poly_1311,    poly_1312,    poly_1313,    poly_1314,    poly_1315, 
    poly_1316,    poly_1317,    poly_1318,    poly_1319,    poly_1320, 
    poly_1321,    poly_1322,    poly_1323,    poly_1324,    poly_1325, 
    poly_1326,    poly_1327,    poly_1328,    poly_1329,    poly_1330, 
    poly_1331,    poly_1332,    poly_1333,    poly_1334,    poly_1335, 
    poly_1336,    poly_1337,    poly_1338,    poly_1339,    poly_1340, 
    poly_1341,    poly_1342,    poly_1343,    poly_1344,    poly_1345, 
    poly_1346,    poly_1347,    poly_1348,    poly_1349,    poly_1350, 
    poly_1351,    poly_1352,    poly_1353,    poly_1354,    poly_1355, 
    poly_1356,    poly_1357,    poly_1358,    poly_1359,    poly_1360, 
    poly_1361,    poly_1362,    poly_1363,    poly_1364,    poly_1365, 
    poly_1366,    poly_1367,    poly_1368,    poly_1369,    poly_1370, 
    poly_1371,    poly_1372,    poly_1373,    poly_1374,    poly_1375, 
    poly_1376,    poly_1377,    poly_1378,    poly_1379,    poly_1380, 
    poly_1381,    poly_1382,    poly_1383,    poly_1384,    poly_1385, 
    poly_1386,    poly_1387,    poly_1388,    poly_1389,    poly_1390, 
    poly_1391,    poly_1392,    poly_1393,    poly_1394,    poly_1395, 
    poly_1396,    poly_1397,    poly_1398,    poly_1399,    poly_1400, 
    poly_1401,    poly_1402,    poly_1403,    poly_1404,    poly_1405, 
    poly_1406,    poly_1407,    poly_1408,    poly_1409,    poly_1410, 
    poly_1411,    poly_1412,    poly_1413,    poly_1414,    poly_1415, 
    poly_1416,    poly_1417,    poly_1418,    poly_1419,    poly_1420, 
    poly_1421,    poly_1422,    poly_1423,    poly_1424,    poly_1425, 
    poly_1426,    poly_1427,    poly_1428,    poly_1429,    poly_1430, 
    poly_1431,    poly_1432,    poly_1433,    poly_1434,    poly_1435, 
    poly_1436,    poly_1437,    poly_1438,    poly_1439,    poly_1440, 
    poly_1441,    poly_1442,    poly_1443,    poly_1444,    poly_1445, 
    poly_1446,    poly_1447,    poly_1448,    poly_1449,    poly_1450, 
    poly_1451,    poly_1452,    poly_1453,    poly_1454,    poly_1455, 
    poly_1456,    poly_1457,    poly_1458,    poly_1459,    poly_1460, 
    poly_1461,    poly_1462,    poly_1463,    poly_1464,    poly_1465, 
    poly_1466,    poly_1467,    poly_1468,    poly_1469,    poly_1470, 
    poly_1471,    poly_1472,    poly_1473,    poly_1474,    poly_1475, 
    poly_1476,    poly_1477,    poly_1478,    poly_1479,    poly_1480, 
    poly_1481,    poly_1482,    poly_1483,    poly_1484,    poly_1485, 
    poly_1486,    poly_1487,    poly_1488,    poly_1489,    poly_1490, 
    poly_1491,    poly_1492,    poly_1493,    poly_1494,    poly_1495, 
    poly_1496,    poly_1497,    poly_1498,    poly_1499,    poly_1500, 
    poly_1501,    poly_1502,    poly_1503,    poly_1504,    poly_1505, 
    poly_1506,    poly_1507,    poly_1508,    poly_1509,    poly_1510, 
    poly_1511,    poly_1512,    poly_1513,    poly_1514,    poly_1515, 
    poly_1516,    poly_1517,    poly_1518,    poly_1519,    poly_1520, 
    poly_1521,    poly_1522,    poly_1523,    poly_1524,    poly_1525, 
    poly_1526,    poly_1527,    poly_1528,    poly_1529,    poly_1530, 
    poly_1531,    poly_1532,    poly_1533,    poly_1534,    poly_1535, 
    poly_1536,    poly_1537,    poly_1538,    poly_1539,    poly_1540, 
    poly_1541,    poly_1542,    poly_1543,    poly_1544,    poly_1545, 
    poly_1546,    poly_1547,    poly_1548,    poly_1549,    poly_1550, 
    poly_1551,    poly_1552,    poly_1553,    poly_1554,    poly_1555, 
    poly_1556,    poly_1557,    poly_1558,    poly_1559,    poly_1560, 
    poly_1561,    poly_1562,    poly_1563,    poly_1564,    poly_1565, 
    poly_1566,    poly_1567,    poly_1568,    poly_1569,    poly_1570, 
    poly_1571,    poly_1572,    poly_1573,    poly_1574,    poly_1575, 
    poly_1576,    poly_1577,    poly_1578,    poly_1579,    poly_1580, 
    poly_1581,    poly_1582,    poly_1583,    poly_1584,    poly_1585, 
    poly_1586,    poly_1587,    poly_1588,    poly_1589,    poly_1590, 
    poly_1591,    poly_1592,    poly_1593,    poly_1594,    poly_1595, 
    poly_1596,    poly_1597,    poly_1598,    poly_1599,    poly_1600, 
    poly_1601,    poly_1602,    poly_1603,    poly_1604,    poly_1605, 
    poly_1606,    poly_1607,    poly_1608,    poly_1609,    poly_1610, 
    poly_1611,    poly_1612,    poly_1613,    poly_1614,    poly_1615, 
    poly_1616,    poly_1617,    poly_1618,    poly_1619,    poly_1620, 
    poly_1621,    poly_1622,    poly_1623,    poly_1624,    poly_1625, 
    poly_1626,    poly_1627,    poly_1628,    poly_1629,    poly_1630, 
    poly_1631,    poly_1632,    poly_1633,    poly_1634,    poly_1635, 
    poly_1636,    poly_1637,    poly_1638,    poly_1639,    poly_1640, 
    poly_1641,    poly_1642,    poly_1643,    poly_1644,    poly_1645, 
    poly_1646,    poly_1647,    poly_1648,    poly_1649,    poly_1650, 
    poly_1651,    poly_1652,    poly_1653,    poly_1654,    poly_1655, 
    poly_1656,    poly_1657,    poly_1658,    poly_1659,    poly_1660, 
    poly_1661,    poly_1662,    poly_1663,    poly_1664,    poly_1665, 
    poly_1666,    poly_1667,    poly_1668,    poly_1669,    poly_1670, 
    poly_1671,    poly_1672,    poly_1673,    poly_1674,    poly_1675, 
    poly_1676,    poly_1677,    poly_1678,    poly_1679,    poly_1680, 
    poly_1681,    poly_1682,    poly_1683,    poly_1684,    poly_1685, 
    poly_1686,    poly_1687,    poly_1688,    poly_1689,    poly_1690, 
    poly_1691,    poly_1692,    poly_1693,    poly_1694,    poly_1695, 
    poly_1696,    poly_1697,    poly_1698,    poly_1699,    poly_1700, 
    poly_1701,    poly_1702,    poly_1703,    poly_1704,    poly_1705, 
    poly_1706,    poly_1707,    poly_1708,    poly_1709,    poly_1710, 
    poly_1711,    poly_1712,    poly_1713,    poly_1714,    poly_1715, 
    poly_1716,    poly_1717,    poly_1718,    poly_1719,    poly_1720, 
    poly_1721,    poly_1722,    poly_1723,    poly_1724,    poly_1725, 
    poly_1726,    poly_1727,    poly_1728,    poly_1729,    poly_1730, 
    poly_1731,    poly_1732,    poly_1733,    poly_1734,    poly_1735, 
    poly_1736,    poly_1737,    poly_1738,    poly_1739,    poly_1740, 
    poly_1741,    poly_1742,    poly_1743,    poly_1744,    poly_1745, 
    poly_1746,    poly_1747,    poly_1748,    poly_1749,    poly_1750, 
    poly_1751,    poly_1752,    poly_1753,    poly_1754,    poly_1755, 
    poly_1756,    poly_1757,    poly_1758,    poly_1759,    poly_1760, 
    poly_1761,    poly_1762,    poly_1763,    poly_1764,    poly_1765, 
    poly_1766,    poly_1767,    poly_1768,    poly_1769,    poly_1770, 
    poly_1771,    poly_1772,    poly_1773,    poly_1774,    poly_1775, 
    poly_1776,    poly_1777,    poly_1778,    poly_1779,    poly_1780, 
    poly_1781,    poly_1782,    poly_1783,    poly_1784,    poly_1785, 
    poly_1786,    poly_1787,    poly_1788,    poly_1789,    poly_1790, 
    poly_1791,    poly_1792,    poly_1793,    poly_1794,    poly_1795, 
    poly_1796,    poly_1797,    poly_1798,    poly_1799,    poly_1800, 
    poly_1801,    poly_1802,    poly_1803,    poly_1804,    poly_1805, 
    poly_1806,    poly_1807,    poly_1808,    poly_1809,    poly_1810, 
    poly_1811,    poly_1812,    poly_1813,    poly_1814,    poly_1815, 
    poly_1816,    poly_1817,    poly_1818,    poly_1819,    poly_1820, 
    poly_1821,    poly_1822,    poly_1823,    poly_1824,    poly_1825, 
    poly_1826,    poly_1827,    poly_1828,    poly_1829,    poly_1830, 
    poly_1831,    poly_1832,    poly_1833,    poly_1834,    poly_1835, 
    poly_1836,    poly_1837,    poly_1838,    poly_1839,    poly_1840, 
    poly_1841,    poly_1842,    poly_1843,    poly_1844,    poly_1845, 
    poly_1846,    poly_1847,    poly_1848,    poly_1849,    poly_1850, 
    poly_1851,    poly_1852,    poly_1853,    poly_1854,    poly_1855, 
    poly_1856,    poly_1857,    poly_1858,    poly_1859,    poly_1860, 
    poly_1861,    poly_1862,    poly_1863,    poly_1864,    poly_1865, 
    poly_1866,    poly_1867,    poly_1868,    poly_1869,    poly_1870, 
    poly_1871,    poly_1872,    poly_1873,    poly_1874,    poly_1875, 
    poly_1876,    poly_1877,    poly_1878,    poly_1879,    poly_1880, 
    poly_1881,    poly_1882,    poly_1883,    poly_1884,    poly_1885, 
    poly_1886,    poly_1887,    poly_1888,    poly_1889,    poly_1890, 
    poly_1891,    poly_1892,    poly_1893,    poly_1894,    poly_1895, 
    poly_1896,    poly_1897,    poly_1898,    poly_1899,    poly_1900, 
    poly_1901,    poly_1902,    poly_1903,    poly_1904,    poly_1905, 
    poly_1906,    poly_1907,    poly_1908,    poly_1909,    poly_1910, 
    poly_1911,    poly_1912,    poly_1913,    poly_1914,    poly_1915, 
    poly_1916,    poly_1917,    poly_1918,    poly_1919,    poly_1920, 
    poly_1921,    poly_1922,    poly_1923,    poly_1924,    poly_1925, 
    poly_1926,    poly_1927,    poly_1928,    poly_1929,    poly_1930, 
    poly_1931,    poly_1932,    poly_1933,    poly_1934,    poly_1935, 
    poly_1936,    poly_1937,    poly_1938,    poly_1939,    poly_1940, 
    poly_1941,    poly_1942,    poly_1943,    poly_1944,    poly_1945, 
    poly_1946,    poly_1947,    poly_1948,    poly_1949,    poly_1950, 
    poly_1951,    poly_1952,    poly_1953,    poly_1954,    poly_1955, 
    poly_1956,    poly_1957,    poly_1958,    poly_1959,    poly_1960, 
    poly_1961,    poly_1962,    poly_1963,    poly_1964,    poly_1965, 
    poly_1966,    poly_1967,    poly_1968,    poly_1969,    poly_1970, 
    poly_1971,    poly_1972,    poly_1973,    poly_1974,    poly_1975, 
    poly_1976,    poly_1977,    poly_1978,    poly_1979,    poly_1980, 
    poly_1981,    poly_1982,    poly_1983,    poly_1984,    poly_1985, 
    poly_1986,    poly_1987,    poly_1988,    poly_1989,    poly_1990, 
    poly_1991,    poly_1992,    poly_1993,    poly_1994,    poly_1995, 
    poly_1996,    poly_1997,    poly_1998,    poly_1999,    poly_2000, 
    poly_2001,    poly_2002,    poly_2003,    poly_2004,    poly_2005, 
    poly_2006,    poly_2007,    poly_2008,    poly_2009,    poly_2010, 
    poly_2011,    poly_2012,    poly_2013,    poly_2014,    poly_2015, 
    poly_2016,    poly_2017,    poly_2018,    poly_2019,    poly_2020, 
    poly_2021,    poly_2022,    poly_2023,    poly_2024,    poly_2025, 
    poly_2026,    poly_2027,    poly_2028,    poly_2029,    poly_2030, 
    poly_2031,    poly_2032,    poly_2033,    poly_2034,    poly_2035, 
    poly_2036,    poly_2037,    poly_2038,    poly_2039,    poly_2040, 
    poly_2041,    poly_2042,    poly_2043,    poly_2044,    poly_2045, 
    poly_2046,    poly_2047,    poly_2048,    poly_2049,    poly_2050, 
    poly_2051,    poly_2052,    poly_2053,    poly_2054,    poly_2055, 
    poly_2056,    poly_2057,    poly_2058,    poly_2059,    poly_2060, 
    poly_2061,    poly_2062,    poly_2063,    poly_2064,    poly_2065, 
    poly_2066,    poly_2067,    poly_2068,    poly_2069,    poly_2070, 
    poly_2071,    poly_2072,    poly_2073,    poly_2074,    poly_2075, 
    poly_2076,    poly_2077,    poly_2078,    poly_2079,    poly_2080, 
    poly_2081,    poly_2082,    poly_2083,    poly_2084,    poly_2085, 
    poly_2086,    poly_2087,    poly_2088,    poly_2089,    poly_2090, 
    poly_2091,    poly_2092,    poly_2093,    poly_2094,    poly_2095, 
    poly_2096,    poly_2097,    poly_2098,    poly_2099,    poly_2100, 
    poly_2101,    poly_2102,    poly_2103,    poly_2104,    poly_2105, 
    poly_2106,    poly_2107,    poly_2108,    poly_2109,    poly_2110, 
    poly_2111,    poly_2112,    poly_2113,    poly_2114,    poly_2115, 
    poly_2116,    poly_2117,    poly_2118,    poly_2119,    poly_2120, 
    poly_2121,    poly_2122,    poly_2123,    poly_2124,    poly_2125, 
    poly_2126,    poly_2127,    poly_2128,    poly_2129,    poly_2130, 
    poly_2131,    poly_2132,    poly_2133,    poly_2134,    poly_2135, 
    poly_2136,    poly_2137,    poly_2138,    poly_2139,    poly_2140, 
    poly_2141,    poly_2142,    poly_2143,    poly_2144,    poly_2145, 
    poly_2146,    poly_2147,    poly_2148,    poly_2149,    poly_2150, 
    poly_2151,    poly_2152,    poly_2153,    poly_2154,    poly_2155, 
    poly_2156,    poly_2157,    poly_2158,    poly_2159,    poly_2160, 
    poly_2161,    poly_2162,    poly_2163,    poly_2164,    poly_2165, 
    poly_2166,    poly_2167,    poly_2168,    poly_2169,    poly_2170, 
    poly_2171,    poly_2172,    poly_2173,    poly_2174,    poly_2175, 
    poly_2176,    poly_2177,    poly_2178,    poly_2179,    poly_2180, 
    poly_2181,    poly_2182,    poly_2183,    poly_2184,    poly_2185, 
    poly_2186,    poly_2187,    poly_2188,    poly_2189,    poly_2190, 
    poly_2191,    poly_2192,    poly_2193,    poly_2194,    poly_2195, 
    poly_2196,    poly_2197,    poly_2198,    poly_2199,    poly_2200, 
    poly_2201,    poly_2202,    poly_2203,    poly_2204,    poly_2205, 
    poly_2206,    poly_2207,    poly_2208,    poly_2209,    poly_2210, 
    poly_2211,    poly_2212,    poly_2213,    poly_2214,    poly_2215, 
    poly_2216,    poly_2217,    poly_2218,    poly_2219,    poly_2220, 
    poly_2221,    poly_2222,    poly_2223,    poly_2224,    poly_2225, 
    poly_2226,    poly_2227,    poly_2228,    poly_2229,    poly_2230, 
    poly_2231,    poly_2232,    poly_2233,    poly_2234,    poly_2235, 
    poly_2236,    poly_2237,    poly_2238,    poly_2239,    poly_2240, 
    poly_2241,    poly_2242,    poly_2243,    poly_2244,    poly_2245, 
    poly_2246,    poly_2247,    poly_2248,    poly_2249,    poly_2250, 
    poly_2251,    poly_2252,    poly_2253,    poly_2254,    poly_2255, 
    poly_2256,    poly_2257,    poly_2258,    poly_2259,    poly_2260, 
    poly_2261,    poly_2262,    poly_2263,    poly_2264,    poly_2265, 
    poly_2266,    poly_2267,    poly_2268,    poly_2269,    poly_2270, 
    poly_2271,    poly_2272,    poly_2273,    poly_2274,    poly_2275, 
    poly_2276,    poly_2277,    poly_2278,    poly_2279,    poly_2280, 
    poly_2281,    poly_2282,    poly_2283,    poly_2284,    poly_2285, 
    poly_2286,    poly_2287,    poly_2288,    poly_2289,    poly_2290, 
    poly_2291,    poly_2292,    poly_2293,    poly_2294,    poly_2295, 
    poly_2296,    poly_2297,    poly_2298,    poly_2299,    poly_2300, 
    poly_2301,    poly_2302,    poly_2303,    poly_2304,    poly_2305, 
    poly_2306,    poly_2307,    poly_2308,    poly_2309,    poly_2310, 
    poly_2311,    poly_2312,    poly_2313,    poly_2314,    poly_2315, 
    poly_2316,    poly_2317,    poly_2318,    poly_2319,    poly_2320, 
    poly_2321,    poly_2322,    poly_2323,    poly_2324,    poly_2325, 
    poly_2326,    ]) 

    return poly 



