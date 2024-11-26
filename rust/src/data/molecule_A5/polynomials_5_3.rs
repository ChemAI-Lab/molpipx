#![allow(unused_variables)]
use super::monomials_5_3::*;


pub const N_POLYS: usize = 12;

// File created from data/molecule_A5/MOL_5_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3] + mono[4] + mono[5] + mono[6] + mono[7] + mono[8] + mono[9] + mono[10];
    poly[2] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16] + mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22] + mono[23] + mono[24] + mono[25];
    poly[3] = mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31] + mono[32] + mono[33] + mono[34] + mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40] + mono[41] + mono[42] + mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54] + mono[55];
    poly[4] = poly[1] * poly[1] - poly[3] - poly[2] - poly[3] - poly[2];
    poly[5] = mono[56] + mono[57] + mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69] + mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85];
    poly[6] = mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93] + mono[94] + mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100] + mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106] + mono[107] + mono[108] + mono[109] + mono[110] + mono[111] + mono[112] + mono[113] + mono[114] + mono[115] + mono[116] + mono[117] + mono[118] + mono[119] + mono[120] + mono[121] + mono[122] + mono[123] + mono[124] + mono[125] + mono[126] + mono[127] + mono[128] + mono[129] + mono[130] + mono[131] + mono[132] + mono[133] + mono[134] + mono[135] + mono[136] + mono[137] + mono[138] + mono[139] + mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145];
    poly[7] = mono[146] + mono[147] + mono[148] + mono[149] + mono[150] + mono[151] + mono[152] + mono[153] + mono[154] + mono[155];
    poly[8] = mono[156] + mono[157] + mono[158] + mono[159] + mono[160] + mono[161] + mono[162] + mono[163] + mono[164] + mono[165] + mono[166] + mono[167] + mono[168] + mono[169] + mono[170] + mono[171] + mono[172] + mono[173] + mono[174] + mono[175];
    poly[9] = poly[1] * poly[2] - poly[6] - poly[5] - poly[5];
    poly[10] = poly[1] * poly[3] - poly[7] - poly[8] - poly[6] - poly[5] - poly[7] - poly[8] - poly[6] - poly[7] - poly[8];
    poly[11] = poly[1] * poly[4] - poly[10] - poly[9];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
