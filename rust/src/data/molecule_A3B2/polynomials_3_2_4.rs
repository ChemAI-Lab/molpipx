#![allow(unused_variables)]
use super::monomials_3_2_4::*;


pub const N_POLYS: usize = 139;

// File created from data/molecule_A3B2/MOL_3_2_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3] + mono[4] + mono[5] + mono[6] + mono[7];
    poly[3] = mono[8] + mono[9] + mono[10];
    poly[4] = poly[1] * poly[2];
    poly[5] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16];
    poly[6] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
    poly[7] = mono[23] + mono[24] + mono[25];
    poly[8] = poly[1] * poly[3];
    poly[9] = mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31];
    poly[10] = poly[2] * poly[3] - poly[9];
    poly[11] = mono[32] + mono[33] + mono[34];
    poly[12] = poly[1] * poly[1];
    poly[13] = poly[2] * poly[2] - poly[7] - poly[6] - poly[5] - poly[7] - poly[6] - poly[5];
    poly[14] = poly[3] * poly[3] - poly[11] - poly[11];
    poly[15] = poly[1] * poly[5];
    poly[16] = poly[1] * poly[6];
    poly[17] = mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40];
    poly[18] = mono[41] + mono[42];
    poly[19] = poly[1] * poly[7];
    poly[20] = mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54];
    poly[21] = poly[1] * poly[9];
    poly[22] = mono[55] + mono[56] + mono[57];
    poly[23] = poly[1] * poly[10];
    poly[24] = mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69];
    poly[25] = mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81];
    poly[26] = poly[3] * poly[5] - poly[24];
    poly[27] = poly[3] * poly[6] - poly[25];
    poly[28] = poly[3] * poly[7] - poly[22];
    poly[29] = poly[1] * poly[11];
    poly[30] = mono[82] + mono[83] + mono[84] + mono[85] + mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93];
    poly[31] = mono[94];
    poly[32] = poly[2] * poly[11] - poly[30];
    poly[33] = poly[1] * poly[4];
    poly[34] = poly[1] * poly[13];
    poly[35] = poly[2] * poly[5] - poly[20] - poly[17] - poly[17];
    poly[36] = poly[2] * poly[6] - poly[20] - poly[18] - poly[17] - poly[18] - poly[18];
    poly[37] = poly[2] * poly[7] - poly[20];
    poly[38] = poly[1] * poly[8];
    poly[39] = poly[2] * poly[9] - poly[25] - poly[24] - poly[22] - poly[22];
    poly[40] = poly[3] * poly[13] - poly[39];
    poly[41] = poly[1] * poly[14];
    poly[42] = poly[3] * poly[9] - poly[30];
    poly[43] = poly[2] * poly[14] - poly[42];
    poly[44] = poly[3] * poly[11] - poly[31] - poly[31] - poly[31];
    poly[45] = poly[1] * poly[12];
    poly[46] = poly[2] * poly[13] - poly[37] - poly[36] - poly[35];
    poly[47] = poly[3] * poly[14] - poly[44];
    poly[48] = poly[1] * poly[17];
    poly[49] = poly[1] * poly[18];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[20];
    poly[51] = mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100];
    poly[52] = mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106];
    poly[53] = mono[107] + mono[108] + mono[109];
    poly[54] = poly[1] * poly[22];
    poly[55] = poly[1] * poly[24];
    poly[56] = poly[1] * poly[25];
    poly[57] = mono[110] + mono[111] + mono[112] + mono[113] + mono[114] + mono[115] + mono[116] + mono[117] + mono[118] + mono[119] + mono[120] + mono[121];
    poly[58] = poly[1] * poly[26];
    poly[59] = mono[122] + mono[123] + mono[124] + mono[125] + mono[126] + mono[127] + mono[128] + mono[129] + mono[130] + mono[131] + mono[132] + mono[133];
    poly[60] = poly[1] * poly[27];
    poly[61] = poly[3] * poly[17] - poly[59];
    poly[62] = poly[3] * poly[18];
    poly[63] = poly[1] * poly[28];
    poly[64] = mono[134] + mono[135] + mono[136] + mono[137] + mono[138] + mono[139] + mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145];
    poly[65] = poly[3] * poly[20] - poly[64] - poly[57];
    poly[66] = poly[1] * poly[30];
    poly[67] = mono[146] + mono[147] + mono[148] + mono[149] + mono[150] + mono[151];
    poly[68] = mono[152] + mono[153] + mono[154] + mono[155] + mono[156] + mono[157];
    poly[69] = mono[158] + mono[159] + mono[160] + mono[161] + mono[162] + mono[163];
    poly[70] = poly[1] * poly[31];
    poly[71] = poly[1] * poly[32];
    poly[72] = poly[5] * poly[11] - poly[67];
    poly[73] = poly[6] * poly[11] - poly[68];
    poly[74] = poly[31] * poly[2];
    poly[75] = poly[7] * poly[11] - poly[69];
    poly[76] = poly[1] * poly[15];
    poly[77] = poly[1] * poly[16];
    poly[78] = poly[1] * poly[19];
    poly[79] = poly[1] * poly[35];
    poly[80] = mono[164] + mono[165] + mono[166] + mono[167] + mono[168] + mono[169];
    poly[81] = poly[1] * poly[36];
    poly[82] = poly[17] * poly[2] - poly[52] - poly[51] - poly[80] - poly[51];
    poly[83] = poly[2] * poly[18] - poly[52];
    poly[84] = poly[5] * poly[6] - poly[52] - poly[82] - poly[52];
    poly[85] = poly[1] * poly[37];
    poly[86] = poly[5] * poly[7] - poly[51];
    poly[87] = poly[6] * poly[7] - poly[52];
    poly[88] = poly[1] * poly[21];
    poly[89] = poly[1] * poly[39];
    poly[90] = poly[2] * poly[22] - poly[57];
    poly[91] = poly[1] * poly[23];
    poly[92] = poly[5] * poly[9] - poly[59] - poly[57];
    poly[93] = poly[6] * poly[9] - poly[62] - poly[61] - poly[57];
    poly[94] = poly[1] * poly[40];
    poly[95] = poly[2] * poly[24] - poly[64] - poly[59] - poly[61] - poly[57] - poly[92] - poly[61];
    poly[96] = poly[2] * poly[25] - poly[64] - poly[62] - poly[59] - poly[57] - poly[93] - poly[62];
    poly[97] = poly[2] * poly[26] - poly[65] - poly[59];
    poly[98] = poly[3] * poly[36] - poly[96] - poly[93];
    poly[99] = poly[3] * poly[37] - poly[90];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[29];
    poly[101] = poly[2] * poly[30] - poly[73] - poly[72] - poly[69] - poly[68] - poly[67] - poly[69] - poly[68] - poly[67];
    poly[102] = poly[11] * poly[13] - poly[101];
    poly[103] = poly[1] * poly[42];
    poly[104] = poly[3] * poly[22] - poly[69];
    poly[105] = poly[1] * poly[43];
    poly[106] = poly[3] * poly[24] - poly[72] - poly[67] - poly[67];
    poly[107] = poly[3] * poly[25] - poly[73] - poly[68] - poly[68];
    poly[108] = poly[3] * poly[26] - poly[72];
    poly[109] = poly[3] * poly[27] - poly[73];
    poly[110] = poly[7] * poly[14] - poly[104];
    poly[111] = poly[1] * poly[44];
    poly[112] = poly[9] * poly[11] - poly[74];
    poly[113] = poly[3] * poly[30] - poly[74] - poly[112] - poly[74];
    poly[114] = poly[31] * poly[3];
    poly[115] = poly[3] * poly[32] - poly[74];
    poly[116] = poly[1] * poly[33];
    poly[117] = poly[1] * poly[34];
    poly[118] = poly[5] * poly[5] - poly[53] - poly[51] - poly[80] - poly[53] - poly[51] - poly[80];
    poly[119] = poly[6] * poly[6] - poly[53] - poly[51] - poly[83] - poly[53] - poly[51] - poly[83];
    poly[120] = poly[7] * poly[7] - poly[53] - poly[53];
    poly[121] = poly[1] * poly[46];
    poly[122] = poly[5] * poly[13] - poly[87] - poly[82];
    poly[123] = poly[6] * poly[13] - poly[86] - poly[83] - poly[80];
    poly[124] = poly[7] * poly[13] - poly[84];
    poly[125] = poly[1] * poly[38];
    poly[126] = poly[2] * poly[39] - poly[93] - poly[92] - poly[90];
    poly[127] = poly[3] * poly[46] - poly[126];
    poly[128] = poly[1] * poly[41];
    poly[129] = poly[3] * poly[39] - poly[101];
    poly[130] = poly[13] * poly[14] - poly[129];
    poly[131] = poly[11] * poly[11] - poly[114] - poly[114];
    poly[132] = poly[1] * poly[47];
    poly[133] = poly[3] * poly[42] - poly[112];
    poly[134] = poly[2] * poly[47] - poly[133];
    poly[135] = poly[11] * poly[14] - poly[114];
    poly[136] = poly[1] * poly[45];
    poly[137] = poly[2] * poly[46] - poly[124] - poly[123] - poly[122];
    poly[138] = poly[3] * poly[47] - poly[135];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);

    return poly;
}
