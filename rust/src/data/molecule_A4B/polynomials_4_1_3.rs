#![allow(unused_variables)]
use super::monomials_4_1_3::*;


pub const N_POLYS: usize = 30;

// File created from data/molecule_A4B/MOL_4_1_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3] + mono[4];
    poly[2] = mono[5] + mono[6] + mono[7] + mono[8] + mono[9] + mono[10];
    poly[3] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16];
    poly[4] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22] + mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28];
    poly[5] = mono[29] + mono[30] + mono[31];
    poly[6] = poly[1] * poly[2] - poly[4];
    poly[7] = mono[32] + mono[33] + mono[34] + mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40] + mono[41] + mono[42] + mono[43];
    poly[8] = poly[1] * poly[1] - poly[3] - poly[3];
    poly[9] = poly[2] * poly[2] - poly[7] - poly[5] - poly[7] - poly[5];
    poly[10] = mono[44] + mono[45] + mono[46] + mono[47];
    poly[11] = mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53];
    poly[12] = mono[54] + mono[55] + mono[56] + mono[57] + mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69] + mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77];
    poly[13] = poly[1] * poly[5];
    poly[14] = poly[2] * poly[3] - poly[12] - poly[11];
    poly[15] = mono[78] + mono[79] + mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85] + mono[86] + mono[87] + mono[88] + mono[89];
    poly[16] = mono[90] + mono[91] + mono[92] + mono[93] + mono[94] + mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100] + mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106] + mono[107] + mono[108] + mono[109] + mono[110] + mono[111] + mono[112] + mono[113];
    poly[17] = mono[114] + mono[115] + mono[116] + mono[117] + mono[118] + mono[119] + mono[120] + mono[121] + mono[122] + mono[123] + mono[124] + mono[125];
    poly[18] = mono[126] + mono[127] + mono[128] + mono[129];
    poly[19] = poly[1] * poly[7] - poly[16] - poly[15];
    poly[20] = mono[130] + mono[131] + mono[132] + mono[133];
    poly[21] = poly[1] * poly[3] - poly[10] - poly[10] - poly[10];
    poly[22] = poly[1] * poly[4] - poly[12] - poly[11] - poly[11];
    poly[23] = poly[2] * poly[8] - poly[22];
    poly[24] = poly[2] * poly[4] - poly[16] - poly[15] - poly[13] - poly[15];
    poly[25] = poly[2] * poly[5] - poly[17];
    poly[26] = poly[1] * poly[9] - poly[24];
    poly[27] = poly[2] * poly[7] - poly[18] - poly[20] - poly[17] - poly[18] - poly[20] - poly[17] - poly[18] - poly[20];
    poly[28] = poly[1] * poly[8] - poly[21];
    poly[29] = poly[2] * poly[9] - poly[27] - poly[25];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
