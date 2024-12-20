#![allow(unused_variables)]
use super::monomials_2_1_1_1_4::*;


pub const N_POLYS: usize = 561;

// File created from data/molecule_A2BCD/MOL_2_1_1_1_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4] + mono[5];
    poly[5] = mono[6] + mono[7];
    poly[6] = mono[8] + mono[9];
    poly[7] = mono[10];
    poly[8] = poly[1] * poly[2];
    poly[9] = poly[1] * poly[3];
    poly[10] = poly[2] * poly[3];
    poly[11] = poly[1] * poly[4];
    poly[12] = poly[2] * poly[4];
    poly[13] = poly[3] * poly[4];
    poly[14] = mono[11];
    poly[15] = poly[1] * poly[5];
    poly[16] = poly[2] * poly[5];
    poly[17] = poly[3] * poly[5];
    poly[18] = mono[12] + mono[13];
    poly[19] = mono[14];
    poly[20] = poly[4] * poly[5] - poly[18];
    poly[21] = poly[1] * poly[6];
    poly[22] = poly[2] * poly[6];
    poly[23] = poly[3] * poly[6];
    poly[24] = mono[15] + mono[16];
    poly[25] = mono[17] + mono[18];
    poly[26] = mono[19];
    poly[27] = poly[4] * poly[6] - poly[24];
    poly[28] = poly[5] * poly[6] - poly[25];
    poly[29] = poly[1] * poly[7];
    poly[30] = poly[2] * poly[7];
    poly[31] = poly[3] * poly[7];
    poly[32] = poly[7] * poly[4];
    poly[33] = poly[7] * poly[5];
    poly[34] = poly[7] * poly[6];
    poly[35] = poly[1] * poly[1];
    poly[36] = poly[2] * poly[2];
    poly[37] = poly[3] * poly[3];
    poly[38] = poly[4] * poly[4] - poly[14] - poly[14];
    poly[39] = poly[5] * poly[5] - poly[19] - poly[19];
    poly[40] = poly[6] * poly[6] - poly[26] - poly[26];
    poly[41] = poly[7] * poly[7];
    poly[42] = poly[1] * poly[10];
    poly[43] = poly[1] * poly[12];
    poly[44] = poly[1] * poly[13];
    poly[45] = poly[2] * poly[13];
    poly[46] = poly[1] * poly[14];
    poly[47] = poly[2] * poly[14];
    poly[48] = poly[3] * poly[14];
    poly[49] = poly[1] * poly[16];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[17];
    poly[51] = poly[2] * poly[17];
    poly[52] = poly[1] * poly[18];
    poly[53] = poly[2] * poly[18];
    poly[54] = poly[3] * poly[18];
    poly[55] = poly[1] * poly[19];
    poly[56] = poly[2] * poly[19];
    poly[57] = poly[3] * poly[19];
    poly[58] = poly[1] * poly[20];
    poly[59] = poly[2] * poly[20];
    poly[60] = poly[3] * poly[20];
    poly[61] = poly[14] * poly[5];
    poly[62] = poly[19] * poly[4];
    poly[63] = poly[1] * poly[22];
    poly[64] = poly[1] * poly[23];
    poly[65] = poly[2] * poly[23];
    poly[66] = poly[1] * poly[24];
    poly[67] = poly[2] * poly[24];
    poly[68] = poly[3] * poly[24];
    poly[69] = poly[1] * poly[25];
    poly[70] = poly[2] * poly[25];
    poly[71] = poly[3] * poly[25];
    poly[72] = mono[20] + mono[21];
    poly[73] = poly[1] * poly[26];
    poly[74] = poly[2] * poly[26];
    poly[75] = poly[3] * poly[26];
    poly[76] = poly[1] * poly[27];
    poly[77] = poly[2] * poly[27];
    poly[78] = poly[3] * poly[27];
    poly[79] = poly[14] * poly[6];
    poly[80] = poly[4] * poly[25] - poly[72];
    poly[81] = poly[26] * poly[4];
    poly[82] = poly[1] * poly[28];
    poly[83] = poly[2] * poly[28];
    poly[84] = poly[3] * poly[28];
    poly[85] = poly[5] * poly[24] - poly[72];
    poly[86] = poly[19] * poly[6];
    poly[87] = poly[26] * poly[5];
    poly[88] = poly[4] * poly[28] - poly[85];
    poly[89] = poly[1] * poly[30];
    poly[90] = poly[1] * poly[31];
    poly[91] = poly[2] * poly[31];
    poly[92] = poly[1] * poly[32];
    poly[93] = poly[2] * poly[32];
    poly[94] = poly[3] * poly[32];
    poly[95] = poly[7] * poly[14];
    poly[96] = poly[1] * poly[33];
    poly[97] = poly[2] * poly[33];
    poly[98] = poly[3] * poly[33];
    poly[99] = poly[7] * poly[18];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[7] * poly[19];
    poly[101] = poly[7] * poly[20];
    poly[102] = poly[1] * poly[34];
    poly[103] = poly[2] * poly[34];
    poly[104] = poly[3] * poly[34];
    poly[105] = poly[7] * poly[24];
    poly[106] = poly[7] * poly[25];
    poly[107] = poly[7] * poly[26];
    poly[108] = poly[7] * poly[27];
    poly[109] = poly[7] * poly[28];
    poly[110] = poly[1] * poly[8];
    poly[111] = poly[1] * poly[36];
    poly[112] = poly[1] * poly[9];
    poly[113] = poly[2] * poly[10];
    poly[114] = poly[1] * poly[37];
    poly[115] = poly[2] * poly[37];
    poly[116] = poly[1] * poly[11];
    poly[117] = poly[2] * poly[12];
    poly[118] = poly[3] * poly[13];
    poly[119] = poly[1] * poly[38];
    poly[120] = poly[2] * poly[38];
    poly[121] = poly[3] * poly[38];
    poly[122] = poly[14] * poly[4];
    poly[123] = poly[1] * poly[15];
    poly[124] = poly[2] * poly[16];
    poly[125] = poly[3] * poly[17];
    poly[126] = poly[4] * poly[18] - poly[61];
    poly[127] = poly[4] * poly[20] - poly[61];
    poly[128] = poly[1] * poly[39];
    poly[129] = poly[2] * poly[39];
    poly[130] = poly[3] * poly[39];
    poly[131] = poly[5] * poly[18] - poly[62];
    poly[132] = poly[19] * poly[5];
    poly[133] = poly[4] * poly[39] - poly[131];
    poly[134] = poly[1] * poly[21];
    poly[135] = poly[2] * poly[22];
    poly[136] = poly[3] * poly[23];
    poly[137] = poly[4] * poly[24] - poly[79];
    poly[138] = poly[5] * poly[25] - poly[86];
    poly[139] = poly[4] * poly[27] - poly[79];
    poly[140] = poly[5] * poly[28] - poly[86];
    poly[141] = poly[1] * poly[40];
    poly[142] = poly[2] * poly[40];
    poly[143] = poly[3] * poly[40];
    poly[144] = poly[6] * poly[24] - poly[81];
    poly[145] = poly[6] * poly[25] - poly[87];
    poly[146] = poly[26] * poly[6];
    poly[147] = poly[4] * poly[40] - poly[144];
    poly[148] = poly[5] * poly[40] - poly[145];
    poly[149] = poly[1] * poly[29];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[2] * poly[30];
    poly[151] = poly[3] * poly[31];
    poly[152] = poly[7] * poly[38];
    poly[153] = poly[7] * poly[39];
    poly[154] = poly[7] * poly[40];
    poly[155] = poly[1] * poly[41];
    poly[156] = poly[2] * poly[41];
    poly[157] = poly[3] * poly[41];
    poly[158] = poly[7] * poly[32];
    poly[159] = poly[7] * poly[33];
    poly[160] = poly[7] * poly[34];
    poly[161] = poly[1] * poly[35];
    poly[162] = poly[2] * poly[36];
    poly[163] = poly[3] * poly[37];
    poly[164] = poly[4] * poly[38] - poly[122];
    poly[165] = poly[5] * poly[39] - poly[132];
    poly[166] = poly[6] * poly[40] - poly[146];
    poly[167] = poly[7] * poly[41];
    poly[168] = poly[1] * poly[45];
    poly[169] = poly[1] * poly[47];
    poly[170] = poly[1] * poly[48];
    poly[171] = poly[2] * poly[48];
    poly[172] = poly[1] * poly[51];
    poly[173] = poly[1] * poly[53];
    poly[174] = poly[1] * poly[54];
    poly[175] = poly[2] * poly[54];
    poly[176] = poly[1] * poly[56];
    poly[177] = poly[1] * poly[57];
    poly[178] = poly[2] * poly[57];
    poly[179] = poly[1] * poly[59];
    poly[180] = poly[1] * poly[60];
    poly[181] = poly[2] * poly[60];
    poly[182] = poly[1] * poly[61];
    poly[183] = poly[2] * poly[61];
    poly[184] = poly[3] * poly[61];
    poly[185] = poly[1] * poly[62];
    poly[186] = poly[2] * poly[62];
    poly[187] = poly[3] * poly[62];
    poly[188] = poly[14] * poly[19];
    poly[189] = poly[1] * poly[65];
    poly[190] = poly[1] * poly[67];
    poly[191] = poly[1] * poly[68];
    poly[192] = poly[2] * poly[68];
    poly[193] = poly[1] * poly[70];
    poly[194] = poly[1] * poly[71];
    poly[195] = poly[2] * poly[71];
    poly[196] = poly[1] * poly[72];
    poly[197] = poly[2] * poly[72];
    poly[198] = poly[3] * poly[72];
    poly[199] = poly[1] * poly[74];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[1] * poly[75];
    poly[201] = poly[2] * poly[75];
    poly[202] = poly[1] * poly[77];
    poly[203] = poly[1] * poly[78];
    poly[204] = poly[2] * poly[78];
    poly[205] = poly[1] * poly[79];
    poly[206] = poly[2] * poly[79];
    poly[207] = poly[3] * poly[79];
    poly[208] = poly[1] * poly[80];
    poly[209] = poly[2] * poly[80];
    poly[210] = poly[3] * poly[80];
    poly[211] = poly[14] * poly[25];
    poly[212] = poly[1] * poly[81];
    poly[213] = poly[2] * poly[81];
    poly[214] = poly[3] * poly[81];
    poly[215] = poly[14] * poly[26];
    poly[216] = poly[1] * poly[83];
    poly[217] = poly[1] * poly[84];
    poly[218] = poly[2] * poly[84];
    poly[219] = poly[1] * poly[85];
    poly[220] = poly[2] * poly[85];
    poly[221] = poly[3] * poly[85];
    poly[222] = poly[1] * poly[86];
    poly[223] = poly[2] * poly[86];
    poly[224] = poly[3] * poly[86];
    poly[225] = poly[19] * poly[24];
    poly[226] = poly[1] * poly[87];
    poly[227] = poly[2] * poly[87];
    poly[228] = poly[3] * poly[87];
    poly[229] = poly[26] * poly[18];
    poly[230] = poly[19] * poly[26];
    poly[231] = poly[1] * poly[88];
    poly[232] = poly[2] * poly[88];
    poly[233] = poly[3] * poly[88];
    poly[234] = poly[14] * poly[28];
    poly[235] = poly[19] * poly[27];
    poly[236] = poly[26] * poly[20];
    poly[237] = poly[1] * poly[91];
    poly[238] = poly[1] * poly[93];
    poly[239] = poly[1] * poly[94];
    poly[240] = poly[2] * poly[94];
    poly[241] = poly[1] * poly[95];
    poly[242] = poly[2] * poly[95];
    poly[243] = poly[3] * poly[95];
    poly[244] = poly[1] * poly[97];
    poly[245] = poly[1] * poly[98];
    poly[246] = poly[2] * poly[98];
    poly[247] = poly[1] * poly[99];
    poly[248] = poly[2] * poly[99];
    poly[249] = poly[3] * poly[99];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[1] * poly[100];
    poly[251] = poly[2] * poly[100];
    poly[252] = poly[3] * poly[100];
    poly[253] = poly[1] * poly[101];
    poly[254] = poly[2] * poly[101];
    poly[255] = poly[3] * poly[101];
    poly[256] = poly[7] * poly[61];
    poly[257] = poly[7] * poly[62];
    poly[258] = poly[1] * poly[103];
    poly[259] = poly[1] * poly[104];
    poly[260] = poly[2] * poly[104];
    poly[261] = poly[1] * poly[105];
    poly[262] = poly[2] * poly[105];
    poly[263] = poly[3] * poly[105];
    poly[264] = poly[1] * poly[106];
    poly[265] = poly[2] * poly[106];
    poly[266] = poly[3] * poly[106];
    poly[267] = poly[7] * poly[72];
    poly[268] = poly[1] * poly[107];
    poly[269] = poly[2] * poly[107];
    poly[270] = poly[3] * poly[107];
    poly[271] = poly[1] * poly[108];
    poly[272] = poly[2] * poly[108];
    poly[273] = poly[3] * poly[108];
    poly[274] = poly[7] * poly[79];
    poly[275] = poly[7] * poly[80];
    poly[276] = poly[7] * poly[81];
    poly[277] = poly[1] * poly[109];
    poly[278] = poly[2] * poly[109];
    poly[279] = poly[3] * poly[109];
    poly[280] = poly[7] * poly[85];
    poly[281] = poly[7] * poly[86];
    poly[282] = poly[7] * poly[87];
    poly[283] = poly[7] * poly[88];
    poly[284] = poly[1] * poly[42];
    poly[285] = poly[1] * poly[113];
    poly[286] = poly[1] * poly[115];
    poly[287] = poly[1] * poly[43];
    poly[288] = poly[1] * poly[117];
    poly[289] = poly[1] * poly[44];
    poly[290] = poly[2] * poly[45];
    poly[291] = poly[1] * poly[118];
    poly[292] = poly[2] * poly[118];
    poly[293] = poly[1] * poly[46];
    poly[294] = poly[2] * poly[47];
    poly[295] = poly[3] * poly[48];
    poly[296] = poly[1] * poly[120];
    poly[297] = poly[1] * poly[121];
    poly[298] = poly[2] * poly[121];
    poly[299] = poly[1] * poly[122];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[2] * poly[122];
    poly[301] = poly[3] * poly[122];
    poly[302] = poly[1] * poly[49];
    poly[303] = poly[1] * poly[124];
    poly[304] = poly[1] * poly[50];
    poly[305] = poly[2] * poly[51];
    poly[306] = poly[1] * poly[125];
    poly[307] = poly[2] * poly[125];
    poly[308] = poly[1] * poly[52];
    poly[309] = poly[2] * poly[53];
    poly[310] = poly[3] * poly[54];
    poly[311] = poly[1] * poly[126];
    poly[312] = poly[2] * poly[126];
    poly[313] = poly[3] * poly[126];
    poly[314] = poly[1] * poly[55];
    poly[315] = poly[2] * poly[56];
    poly[316] = poly[3] * poly[57];
    poly[317] = poly[1] * poly[58];
    poly[318] = poly[2] * poly[59];
    poly[319] = poly[3] * poly[60];
    poly[320] = poly[14] * poly[18];
    poly[321] = poly[1] * poly[127];
    poly[322] = poly[2] * poly[127];
    poly[323] = poly[3] * poly[127];
    poly[324] = poly[14] * poly[20];
    poly[325] = poly[19] * poly[38];
    poly[326] = poly[1] * poly[129];
    poly[327] = poly[1] * poly[130];
    poly[328] = poly[2] * poly[130];
    poly[329] = poly[1] * poly[131];
    poly[330] = poly[2] * poly[131];
    poly[331] = poly[3] * poly[131];
    poly[332] = poly[1] * poly[132];
    poly[333] = poly[2] * poly[132];
    poly[334] = poly[3] * poly[132];
    poly[335] = poly[19] * poly[18];
    poly[336] = poly[1] * poly[133];
    poly[337] = poly[2] * poly[133];
    poly[338] = poly[3] * poly[133];
    poly[339] = poly[14] * poly[39];
    poly[340] = poly[19] * poly[20];
    poly[341] = poly[1] * poly[63];
    poly[342] = poly[1] * poly[135];
    poly[343] = poly[1] * poly[64];
    poly[344] = poly[2] * poly[65];
    poly[345] = poly[1] * poly[136];
    poly[346] = poly[2] * poly[136];
    poly[347] = poly[1] * poly[66];
    poly[348] = poly[2] * poly[67];
    poly[349] = poly[3] * poly[68];
}

#[inline(never)]
fn f_polynomials7(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[350] = poly[1] * poly[137];
    poly[351] = poly[2] * poly[137];
    poly[352] = poly[3] * poly[137];
    poly[353] = poly[1] * poly[69];
    poly[354] = poly[2] * poly[70];
    poly[355] = poly[3] * poly[71];
    poly[356] = poly[4] * poly[72] - poly[211];
    poly[357] = poly[1] * poly[138];
    poly[358] = poly[2] * poly[138];
    poly[359] = poly[3] * poly[138];
    poly[360] = poly[5] * poly[72] - poly[225];
    poly[361] = poly[1] * poly[73];
    poly[362] = poly[2] * poly[74];
    poly[363] = poly[3] * poly[75];
    poly[364] = poly[1] * poly[76];
    poly[365] = poly[2] * poly[77];
    poly[366] = poly[3] * poly[78];
    poly[367] = poly[14] * poly[24];
    poly[368] = poly[4] * poly[138] - poly[360];
    poly[369] = poly[1] * poly[139];
    poly[370] = poly[2] * poly[139];
    poly[371] = poly[3] * poly[139];
    poly[372] = poly[14] * poly[27];
    poly[373] = poly[4] * poly[80] - poly[211];
    poly[374] = poly[26] * poly[38];
    poly[375] = poly[1] * poly[82];
    poly[376] = poly[2] * poly[83];
    poly[377] = poly[3] * poly[84];
    poly[378] = poly[4] * poly[85] - poly[234];
    poly[379] = poly[19] * poly[25];
    poly[380] = poly[4] * poly[88] - poly[234];
    poly[381] = poly[1] * poly[140];
    poly[382] = poly[2] * poly[140];
    poly[383] = poly[3] * poly[140];
    poly[384] = poly[5] * poly[85] - poly[225];
    poly[385] = poly[19] * poly[28];
    poly[386] = poly[26] * poly[39];
    poly[387] = poly[4] * poly[140] - poly[384];
    poly[388] = poly[1] * poly[142];
    poly[389] = poly[1] * poly[143];
    poly[390] = poly[2] * poly[143];
    poly[391] = poly[1] * poly[144];
    poly[392] = poly[2] * poly[144];
    poly[393] = poly[3] * poly[144];
    poly[394] = poly[1] * poly[145];
    poly[395] = poly[2] * poly[145];
    poly[396] = poly[3] * poly[145];
    poly[397] = poly[6] * poly[72] - poly[236];
    poly[398] = poly[1] * poly[146];
    poly[399] = poly[2] * poly[146];
}

#[inline(never)]
fn f_polynomials8(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[400] = poly[3] * poly[146];
    poly[401] = poly[26] * poly[24];
    poly[402] = poly[26] * poly[25];
    poly[403] = poly[1] * poly[147];
    poly[404] = poly[2] * poly[147];
    poly[405] = poly[3] * poly[147];
    poly[406] = poly[14] * poly[40];
    poly[407] = poly[4] * poly[145] - poly[397];
    poly[408] = poly[26] * poly[27];
    poly[409] = poly[1] * poly[148];
    poly[410] = poly[2] * poly[148];
    poly[411] = poly[3] * poly[148];
    poly[412] = poly[5] * poly[144] - poly[397];
    poly[413] = poly[19] * poly[40];
    poly[414] = poly[26] * poly[28];
    poly[415] = poly[4] * poly[148] - poly[412];
    poly[416] = poly[1] * poly[89];
    poly[417] = poly[1] * poly[150];
    poly[418] = poly[1] * poly[90];
    poly[419] = poly[2] * poly[91];
    poly[420] = poly[1] * poly[151];
    poly[421] = poly[2] * poly[151];
    poly[422] = poly[1] * poly[92];
    poly[423] = poly[2] * poly[93];
    poly[424] = poly[3] * poly[94];
    poly[425] = poly[1] * poly[152];
    poly[426] = poly[2] * poly[152];
    poly[427] = poly[3] * poly[152];
    poly[428] = poly[7] * poly[122];
    poly[429] = poly[1] * poly[96];
    poly[430] = poly[2] * poly[97];
    poly[431] = poly[3] * poly[98];
    poly[432] = poly[7] * poly[126];
    poly[433] = poly[7] * poly[127];
    poly[434] = poly[1] * poly[153];
    poly[435] = poly[2] * poly[153];
    poly[436] = poly[3] * poly[153];
    poly[437] = poly[7] * poly[131];
    poly[438] = poly[7] * poly[132];
    poly[439] = poly[7] * poly[133];
    poly[440] = poly[1] * poly[102];
    poly[441] = poly[2] * poly[103];
    poly[442] = poly[3] * poly[104];
    poly[443] = poly[7] * poly[137];
    poly[444] = poly[7] * poly[138];
    poly[445] = poly[7] * poly[139];
    poly[446] = poly[7] * poly[140];
    poly[447] = poly[1] * poly[154];
    poly[448] = poly[2] * poly[154];
    poly[449] = poly[3] * poly[154];
}

#[inline(never)]
fn f_polynomials9(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[450] = poly[7] * poly[144];
    poly[451] = poly[7] * poly[145];
    poly[452] = poly[7] * poly[146];
    poly[453] = poly[7] * poly[147];
    poly[454] = poly[7] * poly[148];
    poly[455] = poly[1] * poly[156];
    poly[456] = poly[1] * poly[157];
    poly[457] = poly[2] * poly[157];
    poly[458] = poly[1] * poly[158];
    poly[459] = poly[2] * poly[158];
    poly[460] = poly[3] * poly[158];
    poly[461] = poly[7] * poly[95];
    poly[462] = poly[1] * poly[159];
    poly[463] = poly[2] * poly[159];
    poly[464] = poly[3] * poly[159];
    poly[465] = poly[7] * poly[99];
    poly[466] = poly[7] * poly[100];
    poly[467] = poly[7] * poly[101];
    poly[468] = poly[1] * poly[160];
    poly[469] = poly[2] * poly[160];
    poly[470] = poly[3] * poly[160];
    poly[471] = poly[7] * poly[105];
    poly[472] = poly[7] * poly[106];
    poly[473] = poly[7] * poly[107];
    poly[474] = poly[7] * poly[108];
    poly[475] = poly[7] * poly[109];
    poly[476] = poly[1] * poly[110];
    poly[477] = poly[1] * poly[111];
    poly[478] = poly[1] * poly[162];
    poly[479] = poly[1] * poly[112];
    poly[480] = poly[2] * poly[113];
    poly[481] = poly[1] * poly[114];
    poly[482] = poly[2] * poly[115];
    poly[483] = poly[1] * poly[163];
    poly[484] = poly[2] * poly[163];
    poly[485] = poly[1] * poly[116];
    poly[486] = poly[2] * poly[117];
    poly[487] = poly[3] * poly[118];
    poly[488] = poly[1] * poly[119];
    poly[489] = poly[2] * poly[120];
    poly[490] = poly[3] * poly[121];
    poly[491] = poly[14] * poly[14];
    poly[492] = poly[1] * poly[164];
    poly[493] = poly[2] * poly[164];
    poly[494] = poly[3] * poly[164];
    poly[495] = poly[14] * poly[38];
    poly[496] = poly[1] * poly[123];
    poly[497] = poly[2] * poly[124];
    poly[498] = poly[3] * poly[125];
    poly[499] = poly[4] * poly[126] - poly[320];
}

#[inline(never)]
fn f_polynomials10(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[500] = poly[4] * poly[127] - poly[324];
    poly[501] = poly[1] * poly[128];
    poly[502] = poly[2] * poly[129];
    poly[503] = poly[3] * poly[130];
    poly[504] = poly[4] * poly[131] - poly[339];
    poly[505] = poly[19] * poly[19];
    poly[506] = poly[4] * poly[133] - poly[339];
    poly[507] = poly[1] * poly[165];
    poly[508] = poly[2] * poly[165];
    poly[509] = poly[3] * poly[165];
    poly[510] = poly[5] * poly[131] - poly[335];
    poly[511] = poly[19] * poly[39];
    poly[512] = poly[4] * poly[165] - poly[510];
    poly[513] = poly[1] * poly[134];
    poly[514] = poly[2] * poly[135];
    poly[515] = poly[3] * poly[136];
    poly[516] = poly[4] * poly[137] - poly[367];
    poly[517] = poly[5] * poly[138] - poly[379];
    poly[518] = poly[4] * poly[139] - poly[372];
    poly[519] = poly[5] * poly[140] - poly[385];
    poly[520] = poly[1] * poly[141];
    poly[521] = poly[2] * poly[142];
    poly[522] = poly[3] * poly[143];
    poly[523] = poly[4] * poly[144] - poly[406];
    poly[524] = poly[5] * poly[145] - poly[413];
    poly[525] = poly[26] * poly[26];
    poly[526] = poly[4] * poly[147] - poly[406];
    poly[527] = poly[5] * poly[148] - poly[413];
    poly[528] = poly[1] * poly[166];
    poly[529] = poly[2] * poly[166];
    poly[530] = poly[3] * poly[166];
    poly[531] = poly[6] * poly[144] - poly[401];
    poly[532] = poly[6] * poly[145] - poly[402];
    poly[533] = poly[26] * poly[40];
    poly[534] = poly[4] * poly[166] - poly[531];
    poly[535] = poly[5] * poly[166] - poly[532];
    poly[536] = poly[1] * poly[149];
    poly[537] = poly[2] * poly[150];
    poly[538] = poly[3] * poly[151];
    poly[539] = poly[7] * poly[164];
    poly[540] = poly[7] * poly[165];
    poly[541] = poly[7] * poly[166];
    poly[542] = poly[1] * poly[155];
    poly[543] = poly[2] * poly[156];
    poly[544] = poly[3] * poly[157];
    poly[545] = poly[7] * poly[152];
    poly[546] = poly[7] * poly[153];
    poly[547] = poly[7] * poly[154];
    poly[548] = poly[1] * poly[167];
    poly[549] = poly[2] * poly[167];
}

#[inline(never)]
fn f_polynomials11(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[550] = poly[3] * poly[167];
    poly[551] = poly[7] * poly[158];
    poly[552] = poly[7] * poly[159];
    poly[553] = poly[7] * poly[160];
    poly[554] = poly[1] * poly[161];
    poly[555] = poly[2] * poly[162];
    poly[556] = poly[3] * poly[163];
    poly[557] = poly[4] * poly[164] - poly[495];
    poly[558] = poly[5] * poly[165] - poly[511];
    poly[559] = poly[6] * poly[166] - poly[533];
    poly[560] = poly[7] * poly[167];
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
    f_polynomials7(&mut poly, &mono);
    f_polynomials8(&mut poly, &mono);
    f_polynomials9(&mut poly, &mono);
    f_polynomials10(&mut poly, &mono);
    f_polynomials11(&mut poly, &mono);

    return poly;
}
