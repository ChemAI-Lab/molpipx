import jax 
import jax.numpy as jnp 
from jax import jit

from molpipx.msa_files.molecule_A4B.monomials_MOL_4_1_6 import f_monomials as f_monos 

# File created from ./MOL_4_1_6.POLY 

N_POLYS = 495

# Total number of monomials = 495 

@jit
def f_polynomials(r): 

    mono = f_monos(r.ravel()) 

    poly = jnp.zeros(495) 

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
    poly_491,    poly_492,    poly_493,    poly_494,    ]) 

    return poly 



