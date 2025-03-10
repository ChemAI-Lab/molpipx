#![allow(unused_variables)]
use super::monomials_3_1_7::*;


pub const N_POLYS: usize = 348;

// File created from data/molecule_A3B/MOL_3_1_7.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3];
    poly[2] = mono[4] + mono[5] + mono[6];
    poly[3] = mono[7] + mono[8] + mono[9];
    poly[4] = mono[10] + mono[11] + mono[12];
    poly[5] = poly[1] * poly[2] - poly[4];
    poly[6] = mono[13] + mono[14] + mono[15];
    poly[7] = poly[1] * poly[1] - poly[3] - poly[3];
    poly[8] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[9] = mono[16];
    poly[10] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
    poly[11] = poly[2] * poly[3] - poly[10];
    poly[12] = mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28];
    poly[13] = mono[29];
    poly[14] = poly[1] * poly[6] - poly[12];
    poly[15] = poly[1] * poly[3] - poly[9] - poly[9] - poly[9];
    poly[16] = poly[1] * poly[4] - poly[10];
    poly[17] = poly[2] * poly[7] - poly[16];
    poly[18] = poly[2] * poly[4] - poly[12];
    poly[19] = poly[1] * poly[8] - poly[18];
    poly[20] = poly[2] * poly[6] - poly[13] - poly[13] - poly[13];
    poly[21] = poly[1] * poly[7] - poly[15];
    poly[22] = poly[2] * poly[8] - poly[20];
    poly[23] = poly[9] * poly[2];
    poly[24] = mono[30] + mono[31] + mono[32];
    poly[25] = poly[3] * poly[6] - poly[24];
    poly[26] = poly[13] * poly[1];
    poly[27] = poly[9] * poly[1];
    poly[28] = poly[3] * poly[4] - poly[23];
    poly[29] = poly[1] * poly[10] - poly[23] - poly[28] - poly[23];
    poly[30] = poly[1] * poly[11] - poly[23];
    poly[31] = poly[1] * poly[12] - poly[25] - poly[24] - poly[24];
    poly[32] = poly[1] * poly[14] - poly[25];
    poly[33] = poly[4] * poly[5] - poly[25] - poly[31];
    poly[34] = poly[2] * poly[11] - poly[25];
    poly[35] = poly[4] * poly[6] - poly[26];
    poly[36] = poly[2] * poly[12] - poly[26] - poly[35] - poly[26];
    poly[37] = poly[13] * poly[2];
    poly[38] = poly[2] * poly[14] - poly[26];
    poly[39] = poly[3] * poly[3] - poly[27] - poly[27];
    poly[40] = poly[3] * poly[7] - poly[27];
    poly[41] = poly[1] * poly[16] - poly[28];
    poly[42] = poly[2] * poly[21] - poly[41];
    poly[43] = poly[1] * poly[18] - poly[33];
    poly[44] = poly[7] * poly[8] - poly[43];
    poly[45] = poly[6] * poly[6] - poly[37] - poly[37];
    poly[46] = poly[2] * poly[18] - poly[35];
    poly[47] = poly[1] * poly[22] - poly[46];
    poly[48] = poly[6] * poly[8] - poly[37];
    poly[49] = poly[1] * poly[21] - poly[40];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[2] * poly[22] - poly[48];
    poly[51] = poly[9] * poly[6];
    poly[52] = poly[13] * poly[3];
    poly[53] = poly[9] * poly[4];
    poly[54] = poly[9] * poly[5];
    poly[55] = poly[1] * poly[24] - poly[51];
    poly[56] = poly[3] * poly[12] - poly[51] - poly[55] - poly[51];
    poly[57] = poly[3] * poly[14] - poly[51];
    poly[58] = poly[13] * poly[7];
    poly[59] = poly[9] * poly[8];
    poly[60] = poly[2] * poly[24] - poly[52];
    poly[61] = poly[13] * poly[4];
    poly[62] = poly[4] * poly[14] - poly[58];
    poly[63] = poly[6] * poly[11] - poly[52];
    poly[64] = poly[13] * poly[5];
    poly[65] = poly[9] * poly[3];
    poly[66] = poly[9] * poly[7];
    poly[67] = poly[3] * poly[16] - poly[53];
    poly[68] = poly[4] * poly[15] - poly[54] - poly[67];
    poly[69] = poly[2] * poly[39] - poly[68];
    poly[70] = poly[1] * poly[29] - poly[54] - poly[68];
    poly[71] = poly[7] * poly[11] - poly[53];
    poly[72] = poly[1] * poly[31] - poly[56] - poly[55];
    poly[73] = poly[1] * poly[32] - poly[57];
    poly[74] = poly[3] * poly[18] - poly[59];
    poly[75] = poly[2] * poly[29] - poly[57] - poly[55];
    poly[76] = poly[1] * poly[34] - poly[59];
    poly[77] = poly[6] * poly[16] - poly[58];
    poly[78] = poly[1] * poly[36] - poly[63] - poly[60];
    poly[79] = poly[2] * poly[32] - poly[58];
    poly[80] = poly[6] * poly[12] - poly[64] - poly[61] - poly[61];
    poly[81] = poly[13] * poly[6];
    poly[82] = poly[1] * poly[45] - poly[80];
    poly[83] = poly[2] * poly[33] - poly[62] - poly[60];
    poly[84] = poly[2] * poly[34] - poly[63];
    poly[85] = poly[6] * poly[18] - poly[61];
    poly[86] = poly[2] * poly[36] - poly[64] - poly[80];
    poly[87] = poly[13] * poly[8];
    poly[88] = poly[8] * poly[14] - poly[61];
    poly[89] = poly[1] * poly[39] - poly[65];
    poly[90] = poly[3] * poly[21] - poly[66];
    poly[91] = poly[1] * poly[41] - poly[67];
    poly[92] = poly[2] * poly[49] - poly[91];
    poly[93] = poly[1] * poly[43] - poly[74];
    poly[94] = poly[8] * poly[21] - poly[93];
    poly[95] = poly[1] * poly[46] - poly[83];
    poly[96] = poly[7] * poly[22] - poly[95];
    poly[97] = poly[2] * poly[45] - poly[81];
    poly[98] = poly[2] * poly[46] - poly[85];
    poly[99] = poly[1] * poly[50] - poly[98];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[6] * poly[22] - poly[87];
    poly[101] = poly[1] * poly[49] - poly[90];
    poly[102] = poly[2] * poly[50] - poly[100];
    poly[103] = poly[9] * poly[13];
    poly[104] = poly[9] * poly[12];
    poly[105] = poly[9] * poly[14];
    poly[106] = poly[13] * poly[15];
    poly[107] = poly[9] * poly[20];
    poly[108] = poly[13] * poly[10];
    poly[109] = poly[13] * poly[11];
    poly[110] = poly[9] * poly[16];
    poly[111] = poly[9] * poly[10];
    poly[112] = poly[9] * poly[11];
    poly[113] = poly[9] * poly[17];
    poly[114] = poly[3] * poly[24] - poly[104];
    poly[115] = poly[7] * poly[24] - poly[105];
    poly[116] = poly[3] * poly[31] - poly[104] - poly[115];
    poly[117] = poly[6] * poly[39] - poly[114];
    poly[118] = poly[3] * poly[32] - poly[105];
    poly[119] = poly[13] * poly[21];
    poly[120] = poly[9] * poly[18];
    poly[121] = poly[9] * poly[19];
    poly[122] = poly[4] * poly[24] - poly[103] - poly[103] - poly[103];
    poly[123] = poly[1] * poly[60] - poly[107] - poly[122];
    poly[124] = poly[13] * poly[16];
    poly[125] = poly[14] * poly[16] - poly[119];
    poly[126] = poly[2] * poly[56] - poly[106] - poly[125];
    poly[127] = poly[4] * poly[32] - poly[119];
    poly[128] = poly[1] * poly[63] - poly[107] - poly[126];
    poly[129] = poly[13] * poly[17];
    poly[130] = poly[6] * poly[24] - poly[108];
    poly[131] = poly[13] * poly[12];
    poly[132] = poly[3] * poly[45] - poly[130];
    poly[133] = poly[13] * poly[14];
    poly[134] = poly[9] * poly[22];
    poly[135] = poly[8] * poly[24] - poly[109];
    poly[136] = poly[13] * poly[18];
    poly[137] = poly[14] * poly[18] - poly[124];
    poly[138] = poly[6] * poly[34] - poly[109];
    poly[139] = poly[13] * poly[19];
    poly[140] = poly[9] * poly[9];
    poly[141] = poly[9] * poly[15];
    poly[142] = poly[9] * poly[21];
    poly[143] = poly[3] * poly[41] - poly[110];
    poly[144] = poly[4] * poly[39] - poly[112];
    poly[145] = poly[1] * poly[68] - poly[111] - poly[144];
    poly[146] = poly[1] * poly[69] - poly[112];
    poly[147] = poly[1] * poly[70] - poly[113] - poly[145];
    poly[148] = poly[11] * poly[21] - poly[110];
    poly[149] = poly[1] * poly[72] - poly[116] - poly[115];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[73] - poly[118];
    poly[151] = poly[3] * poly[43] - poly[120];
    poly[152] = poly[15] * poly[18] - poly[121] - poly[151];
    poly[153] = poly[2] * poly[69] - poly[117];
    poly[154] = poly[1] * poly[75] - poly[121] - poly[152];
    poly[155] = poly[7] * poly[34] - poly[120];
    poly[156] = poly[6] * poly[41] - poly[119];
    poly[157] = poly[1] * poly[78] - poly[126] - poly[123];
    poly[158] = poly[2] * poly[73] - poly[119];
    poly[159] = poly[1] * poly[80] - poly[132] - poly[130] - poly[130];
    poly[160] = poly[13] * poly[13];
    poly[161] = poly[1] * poly[82] - poly[132];
    poly[162] = poly[3] * poly[46] - poly[134];
    poly[163] = poly[2] * poly[75] - poly[127] - poly[123];
    poly[164] = poly[1] * poly[84] - poly[134];
    poly[165] = poly[6] * poly[43] - poly[124];
    poly[166] = poly[1] * poly[86] - poly[138] - poly[135];
    poly[167] = poly[8] * poly[32] - poly[124];
    poly[168] = poly[4] * poly[45] - poly[133];
    poly[169] = poly[2] * poly[80] - poly[131] - poly[168];
    poly[170] = poly[13] * poly[20];
    poly[171] = poly[2] * poly[82] - poly[133];
    poly[172] = poly[2] * poly[83] - poly[137] - poly[135];
    poly[173] = poly[2] * poly[84] - poly[138];
    poly[174] = poly[6] * poly[46] - poly[136];
    poly[175] = poly[2] * poly[86] - poly[139] - poly[169];
    poly[176] = poly[13] * poly[22];
    poly[177] = poly[14] * poly[22] - poly[136];
    poly[178] = poly[3] * poly[39] - poly[141];
    poly[179] = poly[1] * poly[89] - poly[141] - poly[178] - poly[178];
    poly[180] = poly[3] * poly[49] - poly[142];
    poly[181] = poly[1] * poly[91] - poly[143];
    poly[182] = poly[2] * poly[101] - poly[181];
    poly[183] = poly[1] * poly[93] - poly[151];
    poly[184] = poly[8] * poly[49] - poly[183];
    poly[185] = poly[1] * poly[95] - poly[162];
    poly[186] = poly[21] * poly[22] - poly[185];
    poly[187] = poly[6] * poly[45] - poly[170];
    poly[188] = poly[1] * poly[98] - poly[172];
    poly[189] = poly[7] * poly[50] - poly[188];
    poly[190] = poly[2] * poly[97] - poly[170] - poly[187] - poly[187];
    poly[191] = poly[2] * poly[98] - poly[174];
    poly[192] = poly[1] * poly[102] - poly[191];
    poly[193] = poly[6] * poly[50] - poly[176];
    poly[194] = poly[1] * poly[101] - poly[180];
    poly[195] = poly[2] * poly[102] - poly[193];
    poly[196] = poly[9] * poly[26];
    poly[197] = poly[9] * poly[37];
    poly[198] = poly[9] * poly[24];
    poly[199] = poly[9] * poly[31];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[9] * poly[25];
    poly[201] = poly[13] * poly[39];
    poly[202] = poly[9] * poly[32];
    poly[203] = poly[13] * poly[40];
    poly[204] = poly[9] * poly[35];
    poly[205] = poly[9] * poly[36];
    poly[206] = poly[13] * poly[28];
    poly[207] = poly[9] * poly[38];
    poly[208] = poly[13] * poly[29];
    poly[209] = poly[13] * poly[30];
    poly[210] = poly[13] * poly[24];
    poly[211] = poly[9] * poly[45];
    poly[212] = poly[13] * poly[25];
    poly[213] = poly[9] * poly[48];
    poly[214] = poly[13] * poly[33];
    poly[215] = poly[13] * poly[34];
    poly[216] = poly[9] * poly[41];
    poly[217] = poly[9] * poly[28];
    poly[218] = poly[9] * poly[23];
    poly[219] = poly[9] * poly[29];
    poly[220] = poly[9] * poly[30];
    poly[221] = poly[9] * poly[42];
    poly[222] = poly[1] * poly[114] - poly[198];
    poly[223] = poly[21] * poly[24] - poly[202];
    poly[224] = poly[3] * poly[72] - poly[199] - poly[223];
    poly[225] = poly[1] * poly[116] - poly[199] - poly[224];
    poly[226] = poly[14] * poly[39] - poly[198];
    poly[227] = poly[3] * poly[73] - poly[202];
    poly[228] = poly[13] * poly[49];
    poly[229] = poly[9] * poly[43];
    poly[230] = poly[9] * poly[33];
    poly[231] = poly[9] * poly[34];
    poly[232] = poly[9] * poly[44];
    poly[233] = poly[16] * poly[24] - poly[196];
    poly[234] = poly[2] * poly[114] - poly[201];
    poly[235] = poly[1] * poly[123] - poly[205] - poly[234];
    poly[236] = poly[13] * poly[41];
    poly[237] = poly[14] * poly[41] - poly[228];
    poly[238] = poly[2] * poly[116] - poly[203] - poly[237];
    poly[239] = poly[16] * poly[32] - poly[228];
    poly[240] = poly[6] * poly[69] - poly[201];
    poly[241] = poly[4] * poly[73] - poly[228];
    poly[242] = poly[11] * poly[32] - poly[196];
    poly[243] = poly[13] * poly[42];
    poly[244] = poly[1] * poly[130] - poly[211];
    poly[245] = poly[13] * poly[31];
    poly[246] = poly[4] * poly[63] - poly[213] - poly[209];
    poly[247] = poly[13] * poly[26];
    poly[248] = poly[3] * poly[82] - poly[211];
    poly[249] = poly[13] * poly[32];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[9] * poly[46];
    poly[251] = poly[9] * poly[47];
    poly[252] = poly[18] * poly[24] - poly[197];
    poly[253] = poly[1] * poly[135] - poly[213] - poly[252];
    poly[254] = poly[13] * poly[43];
    poly[255] = poly[14] * poly[43] - poly[236];
    poly[256] = poly[2] * poly[126] - poly[209] - poly[246];
    poly[257] = poly[18] * poly[32] - poly[236];
    poly[258] = poly[14] * poly[34] - poly[197];
    poly[259] = poly[13] * poly[44];
    poly[260] = poly[2] * poly[130] - poly[210];
    poly[261] = poly[13] * poly[35];
    poly[262] = poly[13] * poly[36];
    poly[263] = poly[4] * poly[82] - poly[249];
    poly[264] = poly[11] * poly[45] - poly[210];
    poly[265] = poly[13] * poly[38];
    poly[266] = poly[9] * poly[50];
    poly[267] = poly[22] * poly[24] - poly[215];
    poly[268] = poly[13] * poly[46];
    poly[269] = poly[14] * poly[46] - poly[254];
    poly[270] = poly[6] * poly[84] - poly[215];
    poly[271] = poly[13] * poly[47];
    poly[272] = poly[9] * poly[27];
    poly[273] = poly[9] * poly[39];
    poly[274] = poly[9] * poly[40];
    poly[275] = poly[9] * poly[49];
    poly[276] = poly[3] * poly[91] - poly[216];
    poly[277] = poly[16] * poly[39] - poly[218];
    poly[278] = poly[1] * poly[144] - poly[217] - poly[277];
    poly[279] = poly[2] * poly[178] - poly[278];
    poly[280] = poly[1] * poly[145] - poly[219] - poly[278];
    poly[281] = poly[7] * poly[69] - poly[218];
    poly[282] = poly[1] * poly[147] - poly[221] - poly[280];
    poly[283] = poly[11] * poly[49] - poly[216];
    poly[284] = poly[1] * poly[149] - poly[224] - poly[223];
    poly[285] = poly[1] * poly[150] - poly[227];
    poly[286] = poly[3] * poly[93] - poly[229];
    poly[287] = poly[18] * poly[39] - poly[231];
    poly[288] = poly[1] * poly[152] - poly[230] - poly[287];
    poly[289] = poly[1] * poly[153] - poly[231];
    poly[290] = poly[1] * poly[154] - poly[232] - poly[288];
    poly[291] = poly[21] * poly[34] - poly[229];
    poly[292] = poly[6] * poly[91] - poly[228];
    poly[293] = poly[1] * poly[157] - poly[238] - poly[235];
    poly[294] = poly[2] * poly[150] - poly[228];
    poly[295] = poly[1] * poly[159] - poly[246] - poly[244];
    poly[296] = poly[1] * poly[161] - poly[248];
    poly[297] = poly[3] * poly[95] - poly[250];
    poly[298] = poly[2] * poly[152] - poly[239] - poly[234];
    poly[299] = poly[2] * poly[153] - poly[240];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[1] * poly[163] - poly[251] - poly[298];
    poly[301] = poly[7] * poly[84] - poly[250];
    poly[302] = poly[6] * poly[93] - poly[236];
    poly[303] = poly[1] * poly[166] - poly[256] - poly[253];
    poly[304] = poly[8] * poly[73] - poly[236];
    poly[305] = poly[16] * poly[45] - poly[249];
    poly[306] = poly[1] * poly[169] - poly[264] - poly[260];
    poly[307] = poly[13] * poly[37];
    poly[308] = poly[2] * poly[161] - poly[249];
    poly[309] = poly[6] * poly[80] - poly[262] - poly[261];
    poly[310] = poly[13] * poly[45];
    poly[311] = poly[1] * poly[187] - poly[309];
    poly[312] = poly[3] * poly[98] - poly[266];
    poly[313] = poly[2] * poly[163] - poly[257] - poly[253];
    poly[314] = poly[1] * poly[173] - poly[266];
    poly[315] = poly[6] * poly[95] - poly[254];
    poly[316] = poly[1] * poly[175] - poly[270] - poly[267];
    poly[317] = poly[22] * poly[32] - poly[254];
    poly[318] = poly[18] * poly[45] - poly[247];
    poly[319] = poly[2] * poly[169] - poly[262] - poly[309];
    poly[320] = poly[13] * poly[48];
    poly[321] = poly[8] * poly[82] - poly[247];
    poly[322] = poly[2] * poly[172] - poly[269] - poly[267];
    poly[323] = poly[2] * poly[173] - poly[270];
    poly[324] = poly[6] * poly[98] - poly[268];
    poly[325] = poly[2] * poly[175] - poly[271] - poly[319];
    poly[326] = poly[13] * poly[50];
    poly[327] = poly[14] * poly[50] - poly[268];
    poly[328] = poly[1] * poly[178] - poly[273];
    poly[329] = poly[21] * poly[39] - poly[272];
    poly[330] = poly[3] * poly[101] - poly[275];
    poly[331] = poly[1] * poly[181] - poly[276];
    poly[332] = poly[2] * poly[194] - poly[331];
    poly[333] = poly[1] * poly[183] - poly[286];
    poly[334] = poly[8] * poly[101] - poly[333];
    poly[335] = poly[1] * poly[185] - poly[297];
    poly[336] = poly[22] * poly[49] - poly[335];
    poly[337] = poly[1] * poly[188] - poly[312];
    poly[338] = poly[21] * poly[50] - poly[337];
    poly[339] = poly[2] * poly[187] - poly[310];
    poly[340] = poly[1] * poly[191] - poly[322];
    poly[341] = poly[7] * poly[102] - poly[340];
    poly[342] = poly[22] * poly[45] - poly[307];
    poly[343] = poly[2] * poly[191] - poly[324];
    poly[344] = poly[1] * poly[195] - poly[343];
    poly[345] = poly[6] * poly[102] - poly[326];
    poly[346] = poly[1] * poly[194] - poly[330];
    poly[347] = poly[2] * poly[195] - poly[345];
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
