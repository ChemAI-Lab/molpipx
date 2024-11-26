#![allow(unused_variables)]
use super::monomials_2_2_1_4::*;


pub const N_POLYS: usize = 323;

// File created from data/molecule_A2B2C/MOL_2_2_1_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2];
    poly[2] = mono[3];
    poly[3] = mono[4] + mono[5];
    poly[4] = mono[6] + mono[7] + mono[8] + mono[9];
    poly[5] = mono[10];
    poly[6] = mono[11];
    poly[7] = poly[2] * poly[1];
    poly[8] = poly[1] * poly[3];
    poly[9] = poly[2] * poly[3];
    poly[10] = mono[12];
    poly[11] = mono[13] + mono[14] + mono[15] + mono[16];
    poly[12] = poly[1] * poly[4] - poly[11];
    poly[13] = poly[2] * poly[4];
    poly[14] = mono[17] + mono[18] + mono[19] + mono[20];
    poly[15] = mono[21] + mono[22];
    poly[16] = mono[23] + mono[24];
    poly[17] = poly[3] * poly[4] - poly[14];
    poly[18] = mono[25] + mono[26];
    poly[19] = poly[5] * poly[1];
    poly[20] = poly[2] * poly[5];
    poly[21] = poly[5] * poly[3];
    poly[22] = poly[5] * poly[4];
    poly[23] = poly[1] * poly[1] - poly[6] - poly[6];
    poly[24] = poly[2] * poly[2];
    poly[25] = poly[3] * poly[3] - poly[10] - poly[10];
    poly[26] = poly[4] * poly[4] - poly[18] - poly[16] - poly[15] - poly[18] - poly[16] - poly[15];
    poly[27] = poly[5] * poly[5];
    poly[28] = poly[2] * poly[6];
    poly[29] = poly[6] * poly[3];
    poly[30] = poly[2] * poly[8];
    poly[31] = poly[10] * poly[1];
    poly[32] = poly[2] * poly[10];
    poly[33] = poly[6] * poly[4];
    poly[34] = poly[2] * poly[11];
    poly[35] = poly[2] * poly[12];
    poly[36] = mono[27] + mono[28] + mono[29] + mono[30];
    poly[37] = poly[1] * poly[14] - poly[36];
    poly[38] = poly[2] * poly[14];
    poly[39] = poly[1] * poly[15];
    poly[40] = poly[2] * poly[15];
    poly[41] = mono[31] + mono[32];
    poly[42] = poly[1] * poly[16] - poly[41];
    poly[43] = poly[2] * poly[16];
    poly[44] = poly[3] * poly[11] - poly[36];
    poly[45] = poly[1] * poly[17] - poly[44];
    poly[46] = poly[2] * poly[17];
    poly[47] = poly[10] * poly[4];
    poly[48] = poly[3] * poly[15];
    poly[49] = poly[3] * poly[16];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[18];
    poly[51] = poly[2] * poly[18];
    poly[52] = mono[33] + mono[34];
    poly[53] = mono[35] + mono[36] + mono[37] + mono[38];
    poly[54] = poly[3] * poly[18] - poly[52];
    poly[55] = poly[5] * poly[6];
    poly[56] = poly[2] * poly[19];
    poly[57] = poly[5] * poly[8];
    poly[58] = poly[2] * poly[21];
    poly[59] = poly[5] * poly[10];
    poly[60] = poly[5] * poly[11];
    poly[61] = poly[5] * poly[12];
    poly[62] = poly[2] * poly[22];
    poly[63] = poly[5] * poly[14];
    poly[64] = poly[5] * poly[15];
    poly[65] = poly[5] * poly[16];
    poly[66] = poly[5] * poly[17];
    poly[67] = poly[5] * poly[18];
    poly[68] = poly[6] * poly[1];
    poly[69] = poly[2] * poly[23];
    poly[70] = poly[2] * poly[7];
    poly[71] = poly[3] * poly[23];
    poly[72] = poly[2] * poly[9];
    poly[73] = poly[1] * poly[25];
    poly[74] = poly[2] * poly[25];
    poly[75] = poly[10] * poly[3];
    poly[76] = poly[1] * poly[11] - poly[33];
    poly[77] = poly[1] * poly[12] - poly[33];
    poly[78] = poly[2] * poly[13];
    poly[79] = poly[3] * poly[14] - poly[47];
    poly[80] = poly[3] * poly[17] - poly[47];
    poly[81] = poly[4] * poly[11] - poly[50] - poly[41] - poly[39] - poly[41];
    poly[82] = poly[1] * poly[26] - poly[81];
    poly[83] = poly[2] * poly[26];
    poly[84] = poly[4] * poly[14] - poly[52] - poly[49] - poly[48] - poly[52];
    poly[85] = poly[4] * poly[15] - poly[53];
    poly[86] = poly[4] * poly[16] - poly[53];
    poly[87] = poly[3] * poly[26] - poly[84];
    poly[88] = poly[4] * poly[18] - poly[53];
    poly[89] = poly[5] * poly[23];
    poly[90] = poly[2] * poly[20];
    poly[91] = poly[5] * poly[25];
    poly[92] = poly[5] * poly[26];
    poly[93] = poly[5] * poly[19];
    poly[94] = poly[2] * poly[27];
    poly[95] = poly[5] * poly[21];
    poly[96] = poly[5] * poly[22];
    poly[97] = poly[1] * poly[23] - poly[68];
    poly[98] = poly[2] * poly[24];
    poly[99] = poly[3] * poly[25] - poly[75];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[4] * poly[26] - poly[88] - poly[86] - poly[85];
    poly[101] = poly[5] * poly[27];
    poly[102] = poly[2] * poly[29];
    poly[103] = poly[6] * poly[10];
    poly[104] = poly[2] * poly[31];
    poly[105] = poly[2] * poly[33];
    poly[106] = poly[6] * poly[14];
    poly[107] = poly[2] * poly[36];
    poly[108] = poly[2] * poly[37];
    poly[109] = poly[6] * poly[15];
    poly[110] = poly[2] * poly[39];
    poly[111] = poly[6] * poly[16];
    poly[112] = poly[2] * poly[41];
    poly[113] = poly[2] * poly[42];
    poly[114] = poly[6] * poly[17];
    poly[115] = poly[2] * poly[44];
    poly[116] = poly[2] * poly[45];
    poly[117] = poly[10] * poly[11];
    poly[118] = poly[10] * poly[12];
    poly[119] = poly[2] * poly[47];
    poly[120] = mono[39] + mono[40] + mono[41] + mono[42];
    poly[121] = poly[1] * poly[48] - poly[120];
    poly[122] = poly[2] * poly[48];
    poly[123] = poly[10] * poly[15];
    poly[124] = poly[3] * poly[41];
    poly[125] = poly[3] * poly[42];
    poly[126] = poly[2] * poly[49];
    poly[127] = poly[10] * poly[16];
    poly[128] = poly[6] * poly[18];
    poly[129] = poly[2] * poly[50];
    poly[130] = poly[1] * poly[52];
    poly[131] = poly[2] * poly[52];
    poly[132] = mono[43] + mono[44] + mono[45] + mono[46];
    poly[133] = poly[1] * poly[53] - poly[132];
    poly[134] = poly[2] * poly[53];
    poly[135] = mono[47] + mono[48] + mono[49] + mono[50];
    poly[136] = mono[51];
    poly[137] = poly[1] * poly[54];
    poly[138] = poly[2] * poly[54];
    poly[139] = poly[10] * poly[18];
    poly[140] = poly[3] * poly[53] - poly[135];
    poly[141] = poly[2] * poly[55];
    poly[142] = poly[5] * poly[29];
    poly[143] = poly[2] * poly[57];
    poly[144] = poly[5] * poly[31];
    poly[145] = poly[2] * poly[59];
    poly[146] = poly[5] * poly[33];
    poly[147] = poly[2] * poly[60];
    poly[148] = poly[2] * poly[61];
    poly[149] = poly[5] * poly[36];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[5] * poly[37];
    poly[151] = poly[2] * poly[63];
    poly[152] = poly[5] * poly[39];
    poly[153] = poly[2] * poly[64];
    poly[154] = poly[5] * poly[41];
    poly[155] = poly[5] * poly[42];
    poly[156] = poly[2] * poly[65];
    poly[157] = poly[5] * poly[44];
    poly[158] = poly[5] * poly[45];
    poly[159] = poly[2] * poly[66];
    poly[160] = poly[5] * poly[47];
    poly[161] = poly[5] * poly[48];
    poly[162] = poly[5] * poly[49];
    poly[163] = poly[5] * poly[50];
    poly[164] = poly[2] * poly[67];
    poly[165] = poly[5] * poly[52];
    poly[166] = poly[5] * poly[53];
    poly[167] = poly[5] * poly[54];
    poly[168] = poly[2] * poly[68];
    poly[169] = poly[2] * poly[28];
    poly[170] = poly[6] * poly[8];
    poly[171] = poly[2] * poly[71];
    poly[172] = poly[2] * poly[30];
    poly[173] = poly[10] * poly[23];
    poly[174] = poly[2] * poly[32];
    poly[175] = poly[6] * poly[25];
    poly[176] = poly[2] * poly[73];
    poly[177] = poly[10] * poly[8];
    poly[178] = poly[2] * poly[75];
    poly[179] = poly[6] * poly[11];
    poly[180] = poly[6] * poly[12];
    poly[181] = poly[2] * poly[76];
    poly[182] = poly[2] * poly[77];
    poly[183] = poly[2] * poly[34];
    poly[184] = poly[2] * poly[35];
    poly[185] = poly[1] * poly[36] - poly[106];
    poly[186] = poly[1] * poly[37] - poly[106];
    poly[187] = poly[2] * poly[38];
    poly[188] = poly[3] * poly[36] - poly[117];
    poly[189] = poly[1] * poly[79] - poly[188];
    poly[190] = poly[2] * poly[79];
    poly[191] = poly[15] * poly[23];
    poly[192] = poly[2] * poly[40];
    poly[193] = poly[1] * poly[41] - poly[111];
    poly[194] = poly[1] * poly[42] - poly[111];
    poly[195] = poly[2] * poly[43];
    poly[196] = poly[1] * poly[44] - poly[114];
    poly[197] = poly[1] * poly[45] - poly[114];
    poly[198] = poly[2] * poly[46];
    poly[199] = poly[10] * poly[14];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[3] * poly[44] - poly[117];
    poly[201] = poly[1] * poly[80] - poly[200];
    poly[202] = poly[2] * poly[80];
    poly[203] = poly[10] * poly[17];
    poly[204] = poly[15] * poly[25];
    poly[205] = poly[16] * poly[25];
    poly[206] = poly[18] * poly[23];
    poly[207] = poly[2] * poly[51];
    poly[208] = poly[3] * poly[52] - poly[139];
    poly[209] = poly[3] * poly[54] - poly[139];
    poly[210] = poly[6] * poly[26];
    poly[211] = poly[2] * poly[81];
    poly[212] = poly[2] * poly[82];
    poly[213] = poly[4] * poly[36] - poly[130] - poly[124] - poly[121];
    poly[214] = poly[1] * poly[84] - poly[213];
    poly[215] = poly[2] * poly[84];
    poly[216] = poly[11] * poly[15] - poly[132];
    poly[217] = poly[1] * poly[85] - poly[216];
    poly[218] = poly[2] * poly[85];
    poly[219] = poly[14] * poly[15] - poly[135];
    poly[220] = poly[4] * poly[41] - poly[132];
    poly[221] = poly[1] * poly[86] - poly[220];
    poly[222] = poly[2] * poly[86];
    poly[223] = poly[14] * poly[16] - poly[135];
    poly[224] = poly[15] * poly[16];
    poly[225] = poly[3] * poly[81] - poly[213];
    poly[226] = poly[1] * poly[87] - poly[225];
    poly[227] = poly[2] * poly[87];
    poly[228] = poly[10] * poly[26];
    poly[229] = poly[3] * poly[85] - poly[219];
    poly[230] = poly[3] * poly[86] - poly[223];
    poly[231] = poly[11] * poly[18] - poly[132];
    poly[232] = poly[1] * poly[88] - poly[231];
    poly[233] = poly[2] * poly[88];
    poly[234] = poly[4] * poly[52] - poly[135];
    poly[235] = poly[15] * poly[18];
    poly[236] = poly[16] * poly[18];
    poly[237] = poly[3] * poly[88] - poly[234];
    poly[238] = poly[5] * poly[68];
    poly[239] = poly[2] * poly[89];
    poly[240] = poly[2] * poly[56];
    poly[241] = poly[5] * poly[71];
    poly[242] = poly[2] * poly[58];
    poly[243] = poly[5] * poly[73];
    poly[244] = poly[2] * poly[91];
    poly[245] = poly[5] * poly[75];
    poly[246] = poly[5] * poly[76];
    poly[247] = poly[5] * poly[77];
    poly[248] = poly[2] * poly[62];
    poly[249] = poly[5] * poly[79];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[5] * poly[80];
    poly[251] = poly[5] * poly[81];
    poly[252] = poly[5] * poly[82];
    poly[253] = poly[2] * poly[92];
    poly[254] = poly[5] * poly[84];
    poly[255] = poly[5] * poly[85];
    poly[256] = poly[5] * poly[86];
    poly[257] = poly[5] * poly[87];
    poly[258] = poly[5] * poly[88];
    poly[259] = poly[5] * poly[55];
    poly[260] = poly[2] * poly[93];
    poly[261] = poly[5] * poly[57];
    poly[262] = poly[2] * poly[95];
    poly[263] = poly[5] * poly[59];
    poly[264] = poly[5] * poly[60];
    poly[265] = poly[5] * poly[61];
    poly[266] = poly[2] * poly[96];
    poly[267] = poly[5] * poly[63];
    poly[268] = poly[5] * poly[64];
    poly[269] = poly[5] * poly[65];
    poly[270] = poly[5] * poly[66];
    poly[271] = poly[5] * poly[67];
    poly[272] = poly[6] * poly[6];
    poly[273] = poly[6] * poly[23];
    poly[274] = poly[2] * poly[97];
    poly[275] = poly[2] * poly[69];
    poly[276] = poly[2] * poly[70];
    poly[277] = poly[3] * poly[97];
    poly[278] = poly[2] * poly[72];
    poly[279] = poly[23] * poly[25];
    poly[280] = poly[2] * poly[74];
    poly[281] = poly[10] * poly[10];
    poly[282] = poly[1] * poly[99];
    poly[283] = poly[2] * poly[99];
    poly[284] = poly[10] * poly[25];
    poly[285] = poly[1] * poly[76] - poly[179];
    poly[286] = poly[1] * poly[77] - poly[180];
    poly[287] = poly[2] * poly[78];
    poly[288] = poly[3] * poly[79] - poly[199];
    poly[289] = poly[3] * poly[80] - poly[203];
    poly[290] = poly[1] * poly[81] - poly[210];
    poly[291] = poly[1] * poly[82] - poly[210];
    poly[292] = poly[2] * poly[83];
    poly[293] = poly[3] * poly[84] - poly[228];
    poly[294] = poly[15] * poly[15] - poly[136] - poly[136];
    poly[295] = poly[16] * poly[16] - poly[136] - poly[136];
    poly[296] = poly[3] * poly[87] - poly[228];
    poly[297] = poly[18] * poly[18] - poly[136] - poly[136];
    poly[298] = poly[4] * poly[81] - poly[231] - poly[220] - poly[216];
    poly[299] = poly[1] * poly[100] - poly[298];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[2] * poly[100];
    poly[301] = poly[4] * poly[84] - poly[234] - poly[223] - poly[219];
    poly[302] = poly[15] * poly[26] - poly[236];
    poly[303] = poly[16] * poly[26] - poly[235];
    poly[304] = poly[3] * poly[100] - poly[301];
    poly[305] = poly[18] * poly[26] - poly[224];
    poly[306] = poly[5] * poly[97];
    poly[307] = poly[2] * poly[90];
    poly[308] = poly[5] * poly[99];
    poly[309] = poly[5] * poly[100];
    poly[310] = poly[5] * poly[89];
    poly[311] = poly[2] * poly[94];
    poly[312] = poly[5] * poly[91];
    poly[313] = poly[5] * poly[92];
    poly[314] = poly[5] * poly[93];
    poly[315] = poly[2] * poly[101];
    poly[316] = poly[5] * poly[95];
    poly[317] = poly[5] * poly[96];
    poly[318] = poly[1] * poly[97] - poly[273];
    poly[319] = poly[2] * poly[98];
    poly[320] = poly[3] * poly[99] - poly[284];
    poly[321] = poly[4] * poly[100] - poly[305] - poly[303] - poly[302];
    poly[322] = poly[5] * poly[101];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);
    f_polynomials4(&mut poly, &mono);
    f_polynomials5(&mut poly, &mono);
    f_polynomials6(&mut poly, &mono);

    return poly;
}
