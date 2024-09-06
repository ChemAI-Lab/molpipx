#![allow(unused_variables)]
use super::monomials_deg_5::*;


pub const N_POLYS: usize = 208;

// File created from ../Data/Methane/PIP_deg_5/MOL_4_1_5.POLY 


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
    poly[83] = poly[30] * poly[2];
    poly[84] = poly[5] * poly[10];
    poly[85] = mono[262] + mono[263] + mono[264] + mono[265] + mono[266] + mono[267] + mono[268] + mono[269] + mono[270] + mono[271] + mono[272] + mono[273];
    poly[86] = mono[274] + mono[275] + mono[276] + mono[277] + mono[278] + mono[279] + mono[280] + mono[281] + mono[282] + mono[283] + mono[284] + mono[285];
    poly[87] = mono[286] + mono[287] + mono[288] + mono[289] + mono[290] + mono[291] + mono[292] + mono[293] + mono[294] + mono[295] + mono[296] + mono[297] + mono[298] + mono[299] + mono[300] + mono[301] + mono[302] + mono[303] + mono[304] + mono[305] + mono[306] + mono[307] + mono[308] + mono[309];
    poly[88] = poly[7] * poly[10] - poly[87] - poly[85];
    poly[89] = mono[310] + mono[311] + mono[312] + mono[313] + mono[314] + mono[315] + mono[316] + mono[317] + mono[318] + mono[319] + mono[320] + mono[321] + mono[322] + mono[323] + mono[324] + mono[325] + mono[326] + mono[327] + mono[328] + mono[329] + mono[330] + mono[331] + mono[332] + mono[333];
    poly[90] = mono[334] + mono[335] + mono[336] + mono[337] + mono[338] + mono[339] + mono[340] + mono[341] + mono[342] + mono[343] + mono[344] + mono[345] + mono[346] + mono[347] + mono[348] + mono[349] + mono[350] + mono[351] + mono[352] + mono[353] + mono[354] + mono[355] + mono[356] + mono[357];
    poly[91] = poly[1] * poly[38];
    poly[92] = poly[3] * poly[17] - poly[89] - poly[90] - poly[86];
    poly[93] = mono[358] + mono[359] + mono[360] + mono[361] + mono[362] + mono[363] + mono[364] + mono[365] + mono[366] + mono[367] + mono[368] + mono[369];
    poly[94] = poly[3] * poly[18] - poly[93];
    poly[95] = mono[370] + mono[371] + mono[372] + mono[373] + mono[374] + mono[375] + mono[376] + mono[377] + mono[378] + mono[379] + mono[380] + mono[381];
    poly[96] = mono[382] + mono[383] + mono[384] + mono[385] + mono[386] + mono[387] + mono[388] + mono[389] + mono[390] + mono[391] + mono[392] + mono[393];
    poly[97] = mono[394] + mono[395] + mono[396] + mono[397] + mono[398] + mono[399] + mono[400] + mono[401] + mono[402] + mono[403] + mono[404] + mono[405] + mono[406] + mono[407] + mono[408] + mono[409] + mono[410] + mono[411] + mono[412] + mono[413] + mono[414] + mono[415] + mono[416] + mono[417];
    poly[98] = mono[418] + mono[419] + mono[420] + mono[421] + mono[422] + mono[423];
    poly[99] = poly[3] * poly[20] - poly[95];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[45] - poly[97] - poly[96];
    poly[101] = poly[30] * poly[1];
    poly[102] = mono[424] + mono[425] + mono[426] + mono[427] + mono[428] + mono[429] + mono[430] + mono[431] + mono[432] + mono[433] + mono[434] + mono[435] + mono[436] + mono[437] + mono[438] + mono[439] + mono[440] + mono[441] + mono[442] + mono[443] + mono[444] + mono[445] + mono[446] + mono[447];
    poly[103] = poly[4] * poly[10] - poly[83] - poly[102] - poly[83];
    poly[104] = poly[1] * poly[31] - poly[83] - poly[102] - poly[83];
    poly[105] = poly[1] * poly[32] - poly[84] - poly[84];
    poly[106] = poly[1] * poly[33] - poly[83] - poly[103] - poly[83];
    poly[107] = poly[1] * poly[34] - poly[84];
    poly[108] = mono[448] + mono[449] + mono[450] + mono[451] + mono[452] + mono[453] + mono[454] + mono[455] + mono[456] + mono[457] + mono[458] + mono[459] + mono[460] + mono[461] + mono[462] + mono[463] + mono[464] + mono[465] + mono[466] + mono[467] + mono[468] + mono[469] + mono[470] + mono[471];
    poly[109] = poly[1] * poly[35] - poly[87] - poly[85] - poly[108] - poly[85];
    poly[110] = poly[1] * poly[36] - poly[88] - poly[85];
    poly[111] = poly[1] * poly[37] - poly[89] - poly[90] - poly[86] - poly[86];
    poly[112] = poly[1] * poly[39] - poly[93];
    poly[113] = poly[3] * poly[15] - poly[87] - poly[85] - poly[108];
    poly[114] = poly[3] * poly[16] - poly[88] - poly[87] - poly[85] - poly[110] - poly[109] - poly[88] - poly[85];
    poly[115] = poly[1] * poly[40] - poly[87] - poly[113];
    poly[116] = poly[3] * poly[19] - poly[88] - poly[87] - poly[115];
    poly[117] = poly[8] * poly[17] - poly[111];
    poly[118] = poly[8] * poly[18] - poly[112];
    poly[119] = poly[1] * poly[44] - poly[99] - poly[95] - poly[95];
    poly[120] = poly[1] * poly[46] - poly[99];
    poly[121] = poly[5] * poly[11] - poly[86];
    poly[122] = poly[6] * poly[11] - poly[87] - poly[109] - poly[107];
    poly[123] = poly[5] * poly[12] - poly[89] - poly[90];
    poly[124] = poly[9] * poly[10] - poly[122];
    poly[125] = poly[5] * poly[14] - poly[92];
    poly[126] = poly[7] * poly[11] - poly[93] - poly[95] - poly[90];
    poly[127] = poly[5] * poly[15] - poly[96];
    poly[128] = mono[472] + mono[473] + mono[474] + mono[475] + mono[476] + mono[477] + mono[478] + mono[479] + mono[480] + mono[481] + mono[482] + mono[483] + mono[484] + mono[485] + mono[486] + mono[487] + mono[488] + mono[489] + mono[490] + mono[491] + mono[492] + mono[493] + mono[494] + mono[495];
    poly[129] = poly[2] * poly[35] - poly[93] - poly[95] - poly[90] - poly[86] - poly[126] - poly[93] - poly[95] - poly[86];
    poly[130] = poly[2] * poly[36] - poly[94] - poly[95] - poly[89];
    poly[131] = poly[5] * poly[16] - poly[97] - poly[128];
    poly[132] = poly[2] * poly[37] - poly[97] - poly[96] - poly[91] - poly[131] - poly[127] - poly[96] - poly[91];
    poly[133] = poly[2] * poly[38] - poly[98];
    poly[134] = poly[2] * poly[39] - poly[96];
    poly[135] = poly[4] * poly[18] - poly[97] - poly[134];
    poly[136] = poly[5] * poly[18];
    poly[137] = poly[2] * poly[40] - poly[93] - poly[99] - poly[89];
    poly[138] = poly[4] * poly[19] - poly[93] - poly[99] - poly[90] - poly[118] - poly[137] - poly[117] - poly[99];
    poly[139] = poly[5] * poly[19] - poly[100];
    poly[140] = poly[7] * poly[14] - poly[94] - poly[99] - poly[90];
    poly[141] = poly[1] * poly[65] - poly[132];
    poly[142] = poly[1] * poly[66] - poly[135] - poly[134];
    poly[143] = poly[4] * poly[20] - poly[100] - poly[96];
    poly[144] = poly[5] * poly[20];
    poly[145] = poly[2] * poly[44] - poly[97] - poly[96] - poly[143];
    poly[146] = poly[2] * poly[45] - poly[98] - poly[136] - poly[144] - poly[98] - poly[98] - poly[98];
    poly[147] = poly[2] * poly[46] - poly[100];
    poly[148] = poly[3] * poly[10] - poly[101] - poly[101] - poly[101];
    poly[149] = poly[8] * poly[10] - poly[101];
}

fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[3] * poly[11] - poly[83] - poly[102];
    poly[151] = poly[8] * poly[11] - poly[104];
    poly[152] = poly[3] * poly[22] - poly[103] - poly[102] - poly[151];
    poly[153] = poly[1] * poly[49] - poly[103] - poly[102] - poly[152] - poly[103];
    poly[154] = poly[2] * poly[69] - poly[153] - poly[150];
    poly[155] = poly[8] * poly[12] - poly[106] - poly[102] - poly[152];
    poly[156] = poly[5] * poly[28];
    poly[157] = poly[8] * poly[14] - poly[103];
    poly[158] = poly[1] * poly[53] - poly[113] - poly[108];
    poly[159] = poly[1] * poly[54] - poly[114] - poly[110] - poly[109];
    poly[160] = poly[1] * poly[55] - poly[116] - poly[115];
    poly[161] = poly[1] * poly[56] - poly[122];
    poly[162] = poly[5] * poly[22] - poly[111];
    poly[163] = poly[3] * poly[24] - poly[124] - poly[122] - poly[161] - poly[122];
    poly[164] = poly[1] * poly[74];
    poly[165] = poly[1] * poly[58] - poly[124] - poly[122] - poly[163] - poly[124] - poly[122];
    poly[166] = poly[5] * poly[23] - poly[117];
    poly[167] = poly[1] * poly[60] - poly[124];
    poly[168] = poly[1] * poly[61] - poly[137] - poly[129] - poly[126];
    poly[169] = poly[1] * poly[62] - poly[138] - poly[130] - poly[126];
    poly[170] = poly[5] * poly[17] - poly[98] - poly[133] - poly[98];
    poly[171] = poly[1] * poly[64] - poly[140] - poly[130] - poly[129];
    poly[172] = poly[1] * poly[67] - poly[140] - poly[138] - poly[137];
    poly[173] = poly[7] * poly[15] - poly[97] - poly[96] - poly[91] - poly[134] - poly[143] - poly[132] - poly[96] - poly[134];
    poly[174] = poly[7] * poly[16] - poly[100] - poly[97] - poly[96] - poly[91] - poly[142] - poly[135] - poly[145] - poly[143] - poly[141] - poly[132] - poly[100] - poly[97] - poly[96] - poly[91] - poly[135] - poly[145];
    poly[175] = poly[7] * poly[17] - poly[98] - poly[146] - poly[136] - poly[144] - poly[133] - poly[98] - poly[136] - poly[144] - poly[133] - poly[98] - poly[98];
    poly[176] = poly[7] * poly[18] - poly[98] - poly[146] - poly[98];
    poly[177] = poly[1] * poly[76] - poly[174] - poly[173];
    poly[178] = poly[7] * poly[20] - poly[98] - poly[146] - poly[98];
    poly[179] = poly[2] * poly[56] - poly[126] - poly[121];
    poly[180] = poly[5] * poly[24] - poly[132];
    poly[181] = poly[2] * poly[58] - poly[138] - poly[130] - poly[137] - poly[129] - poly[123];
    poly[182] = poly[1] * poly[78] - poly[180];
    poly[183] = poly[2] * poly[60] - poly[140] - poly[125];
    poly[184] = poly[9] * poly[15] - poly[134] - poly[145] - poly[131];
    poly[185] = poly[2] * poly[62] - poly[135] - poly[143] - poly[132] - poly[128] - poly[174] - poly[135];
    poly[186] = poly[5] * poly[27] - poly[146] - poly[175];
    poly[187] = poly[9] * poly[16] - poly[142] - poly[143] - poly[139] - poly[127] - poly[185];
    poly[188] = poly[2] * poly[65] - poly[146] - poly[133] - poly[175];
    poly[189] = poly[9] * poly[18] - poly[144];
    poly[190] = poly[1] * poly[80] - poly[187] - poly[185] - poly[184];
    poly[191] = poly[9] * poly[20] - poly[136];
    poly[192] = poly[1] * poly[69] - poly[148];
    poly[193] = poly[3] * poly[28] - poly[149];
    poly[194] = poly[1] * poly[71] - poly[152] - poly[151];
    poly[195] = poly[2] * poly[81] - poly[194];
    poly[196] = poly[1] * poly[73] - poly[163] - poly[161];
    poly[197] = poly[9] * poly[28] - poly[196];
    poly[198] = poly[1] * poly[77] - poly[181] - poly[179] - poly[179];
    poly[199] = poly[2] * poly[74] - poly[170];
}

fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[8] * poly[29] - poly[198];
    poly[201] = poly[2] * poly[76] - poly[176] - poly[178] - poly[175];
    poly[202] = poly[2] * poly[77] - poly[185] - poly[184] - poly[180];
    poly[203] = poly[5] * poly[29] - poly[188];
    poly[204] = poly[1] * poly[82] - poly[202];
    poly[205] = poly[7] * poly[29] - poly[189] - poly[191] - poly[186];
    poly[206] = poly[1] * poly[81] - poly[193];
    poly[207] = poly[2] * poly[82] - poly[205] - poly[203];
}

// Total number of monomials = 208 

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
