#![allow(unused_variables)]
use super::monomials_4_1_4::*;


pub const N_POLYS: usize = 83;

// File created from data/molecule_A4B/MOL_4_1_4.POLY 


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
    poly[30] = mono[134];
    poly[31] = mono[135] + mono[136] + mono[137] + mono[138] + mono[139] + mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145] + mono[146];
    poly[32] = mono[147] + mono[148] + mono[149] + mono[150] + mono[151] + mono[152] + mono[153] + mono[154] + mono[155] + mono[156] + mono[157] + mono[158];
    poly[33] = poly[2] * poly[10] - poly[31];
    poly[34] = poly[3] * poly[5] - poly[32];
    poly[35] = mono[159] + mono[160] + mono[161] + mono[162] + mono[163] + mono[164] + mono[165] + mono[166] + mono[167] + mono[168] + mono[169] + mono[170] + mono[171] + mono[172] + mono[173] + mono[174] + mono[175] + mono[176] + mono[177] + mono[178] + mono[179] + mono[180] + mono[181] + mono[182];
    poly[36] = mono[183] + mono[184] + mono[185] + mono[186] + mono[187] + mono[188] + mono[189] + mono[190] + mono[191] + mono[192] + mono[193] + mono[194];
    poly[37] = mono[195] + mono[196] + mono[197] + mono[198] + mono[199] + mono[200] + mono[201] + mono[202] + mono[203] + mono[204] + mono[205] + mono[206] + mono[207] + mono[208] + mono[209] + mono[210] + mono[211] + mono[212] + mono[213] + mono[214] + mono[215] + mono[216] + mono[217] + mono[218];
    poly[38] = mono[219] + mono[220] + mono[221];
    poly[39] = mono[222] + mono[223] + mono[224] + mono[225];
    poly[40] = mono[226] + mono[227] + mono[228] + mono[229] + mono[230] + mono[231] + mono[232] + mono[233] + mono[234] + mono[235] + mono[236] + mono[237];
    poly[41] = poly[3] * poly[7] - poly[36] - poly[40] - poly[35];
    poly[42] = poly[1] * poly[17] - poly[37];
    poly[43] = poly[1] * poly[18] - poly[39];
    poly[44] = mono[238] + mono[239] + mono[240] + mono[241] + mono[242] + mono[243] + mono[244] + mono[245] + mono[246] + mono[247] + mono[248] + mono[249];
    poly[45] = mono[250] + mono[251] + mono[252] + mono[253] + mono[254] + mono[255] + mono[256] + mono[257] + mono[258] + mono[259] + mono[260] + mono[261];
    poly[46] = poly[1] * poly[20] - poly[44];
    poly[47] = poly[1] * poly[10] - poly[30] - poly[30] - poly[30] - poly[30];
    poly[48] = poly[1] * poly[11] - poly[31];
    poly[49] = poly[3] * poly[4] - poly[33] - poly[31] - poly[48] - poly[31];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[12] - poly[33] - poly[31] - poly[49] - poly[33] - poly[31];
    poly[51] = poly[5] * poly[8];
    poly[52] = poly[1] * poly[14] - poly[33];
    poly[53] = poly[1] * poly[15] - poly[40] - poly[35];
    poly[54] = poly[1] * poly[16] - poly[41] - poly[36] - poly[35] - poly[36];
    poly[55] = poly[1] * poly[19] - poly[41] - poly[40];
    poly[56] = poly[2] * poly[11] - poly[35] - poly[34];
    poly[57] = poly[4] * poly[5] - poly[37];
    poly[58] = poly[2] * poly[12] - poly[41] - poly[36] - poly[40] - poly[35] - poly[32] - poly[36] - poly[40] - poly[32];
    poly[59] = poly[1] * poly[25] - poly[57];
    poly[60] = poly[2] * poly[14] - poly[41] - poly[34];
    poly[61] = poly[2] * poly[15] - poly[39] - poly[44] - poly[37] - poly[39] - poly[39];
    poly[62] = poly[4] * poly[7] - poly[43] - poly[39] - poly[44] - poly[42] - poly[37] - poly[61] - poly[39] - poly[44] - poly[39];
    poly[63] = poly[5] * poly[7] - poly[45];
    poly[64] = poly[2] * poly[16] - poly[43] - poly[44] - poly[42] - poly[37] - poly[62] - poly[43] - poly[44];
    poly[65] = poly[2] * poly[17] - poly[45] - poly[38] - poly[63] - poly[45] - poly[38] - poly[38] - poly[38];
    poly[66] = poly[2] * poly[18] - poly[45];
    poly[67] = poly[1] * poly[27] - poly[64] - poly[62] - poly[61];
    poly[68] = poly[2] * poly[20] - poly[45];
    poly[69] = poly[3] * poly[3] - poly[30] - poly[47] - poly[30] - poly[47] - poly[30] - poly[30] - poly[30] - poly[30];
    poly[70] = poly[3] * poly[8] - poly[47];
    poly[71] = poly[1] * poly[22] - poly[49] - poly[48];
    poly[72] = poly[2] * poly[28] - poly[71];
    poly[73] = poly[1] * poly[24] - poly[58] - poly[56] - poly[56];
    poly[74] = poly[5] * poly[5] - poly[38] - poly[38];
    poly[75] = poly[8] * poly[9] - poly[73];
    poly[76] = poly[7] * poly[7] - poly[45] - poly[38] - poly[66] - poly[68] - poly[65] - poly[45] - poly[38] - poly[66] - poly[68] - poly[65] - poly[45] - poly[38] - poly[45] - poly[38];
    poly[77] = poly[2] * poly[24] - poly[62] - poly[61] - poly[57];
    poly[78] = poly[5] * poly[9] - poly[65];
    poly[79] = poly[1] * poly[29] - poly[77];
    poly[80] = poly[7] * poly[9] - poly[66] - poly[68] - poly[63];
    poly[81] = poly[1] * poly[28] - poly[70];
    poly[82] = poly[2] * poly[29] - poly[80] - poly[78];
}

// Total number of monomials = 83 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
