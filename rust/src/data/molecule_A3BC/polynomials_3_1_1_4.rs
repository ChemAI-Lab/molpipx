#![allow(unused_variables)]
use super::monomials_3_1_1_4::*;


pub const N_POLYS: usize = 231;

// File created from data/molecule_A3BC/MOL_3_1_1_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3] + mono[4];
    poly[3] = mono[5] + mono[6] + mono[7];
    poly[4] = mono[8] + mono[9] + mono[10];
    poly[5] = poly[1] * poly[2];
    poly[6] = mono[11] + mono[12] + mono[13];
    poly[7] = poly[1] * poly[3];
    poly[8] = mono[14] + mono[15] + mono[16] + mono[17] + mono[18] + mono[19];
    poly[9] = mono[20] + mono[21] + mono[22];
    poly[10] = poly[2] * poly[3] - poly[8];
    poly[11] = poly[1] * poly[4];
    poly[12] = mono[23] + mono[24] + mono[25];
    poly[13] = mono[26] + mono[27] + mono[28];
    poly[14] = poly[2] * poly[4] - poly[12];
    poly[15] = poly[3] * poly[4] - poly[13];
    poly[16] = mono[29] + mono[30] + mono[31];
    poly[17] = poly[1] * poly[1];
    poly[18] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[19] = poly[3] * poly[3] - poly[9] - poly[9];
    poly[20] = poly[4] * poly[4] - poly[16] - poly[16];
    poly[21] = poly[1] * poly[6];
    poly[22] = mono[32];
    poly[23] = poly[1] * poly[8];
    poly[24] = mono[33] + mono[34] + mono[35];
    poly[25] = poly[1] * poly[9];
    poly[26] = mono[36] + mono[37] + mono[38];
    poly[27] = mono[39];
    poly[28] = poly[1] * poly[10];
    poly[29] = poly[3] * poly[6] - poly[24];
    poly[30] = poly[2] * poly[9] - poly[26];
    poly[31] = poly[1] * poly[12];
    poly[32] = poly[1] * poly[13];
    poly[33] = mono[40] + mono[41] + mono[42];
    poly[34] = poly[1] * poly[14];
    poly[35] = mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48];
    poly[36] = poly[2] * poly[13] - poly[33];
    poly[37] = poly[4] * poly[6] - poly[35];
    poly[38] = poly[1] * poly[15];
    poly[39] = poly[3] * poly[12] - poly[33];
    poly[40] = mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54];
    poly[41] = poly[4] * poly[8] - poly[39] - poly[36];
    poly[42] = poly[4] * poly[9] - poly[40];
    poly[43] = poly[4] * poly[10] - poly[33];
    poly[44] = poly[1] * poly[16];
    poly[45] = mono[55] + mono[56] + mono[57] + mono[58] + mono[59] + mono[60];
    poly[46] = mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66];
    poly[47] = mono[67];
    poly[48] = poly[2] * poly[16] - poly[45];
    poly[49] = poly[3] * poly[16] - poly[46];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[5];
    poly[51] = poly[1] * poly[18];
    poly[52] = poly[2] * poly[6] - poly[22] - poly[22] - poly[22];
    poly[53] = poly[1] * poly[7];
    poly[54] = poly[2] * poly[8] - poly[29] - poly[24] - poly[24];
    poly[55] = poly[2] * poly[10] - poly[29];
    poly[56] = poly[1] * poly[19];
    poly[57] = poly[3] * poly[8] - poly[30] - poly[26] - poly[26];
    poly[58] = poly[3] * poly[9] - poly[27] - poly[27] - poly[27];
    poly[59] = poly[2] * poly[19] - poly[57];
    poly[60] = poly[1] * poly[11];
    poly[61] = poly[2] * poly[12] - poly[35];
    poly[62] = poly[3] * poly[13] - poly[40];
    poly[63] = poly[4] * poly[18] - poly[61];
    poly[64] = poly[4] * poly[19] - poly[62];
    poly[65] = poly[1] * poly[20];
    poly[66] = poly[4] * poly[12] - poly[45];
    poly[67] = poly[4] * poly[13] - poly[46];
    poly[68] = poly[2] * poly[20] - poly[66];
    poly[69] = poly[3] * poly[20] - poly[67];
    poly[70] = poly[4] * poly[16] - poly[47] - poly[47] - poly[47];
    poly[71] = poly[1] * poly[17];
    poly[72] = poly[2] * poly[18] - poly[52];
    poly[73] = poly[3] * poly[19] - poly[58];
    poly[74] = poly[4] * poly[20] - poly[70];
    poly[75] = poly[1] * poly[22];
    poly[76] = poly[1] * poly[24];
    poly[77] = poly[1] * poly[26];
    poly[78] = poly[1] * poly[27];
    poly[79] = poly[1] * poly[29];
    poly[80] = poly[22] * poly[3];
    poly[81] = poly[1] * poly[30];
    poly[82] = mono[68] + mono[69] + mono[70] + mono[71] + mono[72] + mono[73];
    poly[83] = poly[27] * poly[2];
    poly[84] = poly[6] * poly[9] - poly[82];
    poly[85] = poly[1] * poly[33];
    poly[86] = poly[1] * poly[35];
    poly[87] = poly[1] * poly[36];
    poly[88] = mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79];
    poly[89] = poly[1] * poly[37];
    poly[90] = poly[22] * poly[4];
    poly[91] = poly[6] * poly[13] - poly[88];
    poly[92] = poly[1] * poly[39];
    poly[93] = poly[1] * poly[40];
    poly[94] = mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85];
    poly[95] = poly[1] * poly[41];
    poly[96] = poly[4] * poly[24] - poly[91];
    poly[97] = mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91];
    poly[98] = poly[1] * poly[42];
    poly[99] = poly[4] * poly[26] - poly[97];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[27] * poly[4];
    poly[101] = poly[1] * poly[43];
    poly[102] = poly[3] * poly[35] - poly[96] - poly[88];
    poly[103] = poly[2] * poly[40] - poly[97] - poly[94];
    poly[104] = poly[3] * poly[37] - poly[91];
    poly[105] = poly[2] * poly[42] - poly[99];
    poly[106] = poly[1] * poly[45];
    poly[107] = mono[92] + mono[93] + mono[94];
    poly[108] = poly[1] * poly[46];
    poly[109] = mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100];
    poly[110] = mono[101] + mono[102] + mono[103];
    poly[111] = mono[104] + mono[105] + mono[106] + mono[107] + mono[108] + mono[109];
    poly[112] = poly[1] * poly[47];
    poly[113] = poly[1] * poly[48];
    poly[114] = poly[6] * poly[16] - poly[107];
    poly[115] = poly[2] * poly[46] - poly[111] - poly[109];
    poly[116] = poly[47] * poly[2];
    poly[117] = poly[1] * poly[49];
    poly[118] = poly[3] * poly[45] - poly[111] - poly[109];
    poly[119] = poly[9] * poly[16] - poly[110];
    poly[120] = poly[47] * poly[3];
    poly[121] = poly[2] * poly[49] - poly[118];
    poly[122] = poly[1] * poly[21];
    poly[123] = poly[1] * poly[52];
    poly[124] = poly[22] * poly[2];
    poly[125] = poly[1] * poly[23];
    poly[126] = poly[1] * poly[54];
    poly[127] = poly[2] * poly[24] - poly[80];
    poly[128] = poly[1] * poly[25];
    poly[129] = poly[2] * poly[26] - poly[82];
    poly[130] = poly[1] * poly[28];
    poly[131] = poly[6] * poly[8] - poly[80] - poly[127] - poly[80];
    poly[132] = poly[1] * poly[55];
    poly[133] = poly[6] * poly[10] - poly[80];
    poly[134] = poly[9] * poly[18] - poly[129];
    poly[135] = poly[1] * poly[57];
    poly[136] = poly[3] * poly[24] - poly[82];
    poly[137] = poly[1] * poly[58];
    poly[138] = poly[3] * poly[26] - poly[83];
    poly[139] = poly[27] * poly[3];
    poly[140] = poly[8] * poly[9] - poly[83] - poly[138] - poly[83];
    poly[141] = poly[1] * poly[59];
    poly[142] = poly[6] * poly[19] - poly[136];
    poly[143] = poly[9] * poly[10] - poly[83];
    poly[144] = poly[1] * poly[31];
    poly[145] = poly[1] * poly[61];
    poly[146] = poly[1] * poly[32];
    poly[147] = poly[2] * poly[33] - poly[88];
    poly[148] = poly[1] * poly[62];
    poly[149] = poly[3] * poly[33] - poly[94];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[34];
    poly[151] = poly[6] * poly[12] - poly[90];
    poly[152] = poly[2] * poly[62] - poly[149];
    poly[153] = poly[1] * poly[63];
    poly[154] = poly[2] * poly[35] - poly[90] - poly[151] - poly[90];
    poly[155] = poly[13] * poly[18] - poly[147];
    poly[156] = poly[2] * poly[37] - poly[90];
    poly[157] = poly[1] * poly[38];
    poly[158] = poly[3] * poly[61] - poly[147];
    poly[159] = poly[9] * poly[13] - poly[100];
    poly[160] = poly[2] * poly[41] - poly[104] - poly[96];
    poly[161] = poly[4] * poly[55] - poly[147];
    poly[162] = poly[1] * poly[64];
    poly[163] = poly[12] * poly[19] - poly[149];
    poly[164] = poly[3] * poly[40] - poly[100] - poly[159] - poly[100];
    poly[165] = poly[3] * poly[41] - poly[105] - poly[97];
    poly[166] = poly[3] * poly[42] - poly[100];
    poly[167] = poly[4] * poly[59] - poly[149];
    poly[168] = poly[1] * poly[44];
    poly[169] = poly[2] * poly[45] - poly[114] - poly[107] - poly[107];
    poly[170] = poly[3] * poly[46] - poly[119] - poly[110] - poly[110];
    poly[171] = poly[2] * poly[48] - poly[114];
    poly[172] = poly[3] * poly[49] - poly[119];
    poly[173] = poly[1] * poly[66];
    poly[174] = poly[1] * poly[67];
    poly[175] = poly[4] * poly[33] - poly[111];
    poly[176] = poly[1] * poly[68];
    poly[177] = poly[12] * poly[14] - poly[114] - poly[169];
    poly[178] = poly[2] * poly[67] - poly[175];
    poly[179] = poly[4] * poly[37] - poly[114];
    poly[180] = poly[1] * poly[69];
    poly[181] = poly[3] * poly[66] - poly[175];
    poly[182] = poly[13] * poly[15] - poly[119] - poly[170];
    poly[183] = poly[4] * poly[41] - poly[118] - poly[115];
    poly[184] = poly[4] * poly[42] - poly[119];
    poly[185] = poly[10] * poly[20] - poly[175];
    poly[186] = poly[1] * poly[70];
    poly[187] = poly[12] * poly[16] - poly[116];
    poly[188] = poly[13] * poly[16] - poly[120];
    poly[189] = poly[4] * poly[45] - poly[116] - poly[187] - poly[116];
    poly[190] = poly[4] * poly[46] - poly[120] - poly[188] - poly[120];
    poly[191] = poly[47] * poly[4];
    poly[192] = poly[4] * poly[48] - poly[116];
    poly[193] = poly[4] * poly[49] - poly[120];
    poly[194] = poly[1] * poly[50];
    poly[195] = poly[1] * poly[51];
    poly[196] = poly[6] * poly[6] - poly[124] - poly[124];
    poly[197] = poly[1] * poly[72];
    poly[198] = poly[6] * poly[18] - poly[124];
    poly[199] = poly[1] * poly[53];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[2] * poly[54] - poly[131] - poly[127];
    poly[201] = poly[2] * poly[55] - poly[133];
    poly[202] = poly[1] * poly[56];
    poly[203] = poly[2] * poly[57] - poly[142] - poly[136] - poly[136];
    poly[204] = poly[9] * poly[9] - poly[139] - poly[139];
    poly[205] = poly[2] * poly[59] - poly[142];
    poly[206] = poly[1] * poly[73];
    poly[207] = poly[3] * poly[57] - poly[140] - poly[138];
    poly[208] = poly[9] * poly[19] - poly[139];
    poly[209] = poly[2] * poly[73] - poly[207];
    poly[210] = poly[1] * poly[60];
    poly[211] = poly[2] * poly[61] - poly[151];
    poly[212] = poly[3] * poly[62] - poly[159];
    poly[213] = poly[4] * poly[72] - poly[211];
    poly[214] = poly[4] * poly[73] - poly[212];
    poly[215] = poly[1] * poly[65];
    poly[216] = poly[2] * poly[66] - poly[177];
    poly[217] = poly[3] * poly[67] - poly[182];
    poly[218] = poly[18] * poly[20] - poly[216];
    poly[219] = poly[19] * poly[20] - poly[217];
    poly[220] = poly[16] * poly[16] - poly[191] - poly[191];
    poly[221] = poly[1] * poly[74];
    poly[222] = poly[4] * poly[66] - poly[187];
    poly[223] = poly[4] * poly[67] - poly[188];
    poly[224] = poly[2] * poly[74] - poly[222];
    poly[225] = poly[3] * poly[74] - poly[223];
    poly[226] = poly[16] * poly[20] - poly[191];
    poly[227] = poly[1] * poly[71];
    poly[228] = poly[2] * poly[72] - poly[198];
    poly[229] = poly[3] * poly[73] - poly[208];
    poly[230] = poly[4] * poly[74] - poly[226];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);
    f_polynomials4(&mut poly, &mono);

    return poly;
}
