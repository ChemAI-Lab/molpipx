use crate::monomials_::*;

pub const N_POLYS: usize = 10737;

// File created from data/MOL_2_2_2_2_1_1_1_1_3.POLY

fn f_polynomials0(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4];
    poly[5] = mono[5];
    poly[6] = mono[6];
    poly[7] = mono[7] + mono[8];
    poly[8] = mono[9] + mono[10];
    poly[9] = mono[11] + mono[12];
    poly[10] = mono[13] + mono[14];
    poly[11] = mono[15];
    poly[12] = mono[16] + mono[17];
    poly[13] = mono[18] + mono[19];
    poly[14] = mono[20] + mono[21];
    poly[15] = mono[22] + mono[23];
    poly[16] = mono[24] + mono[25] + mono[26] + mono[27];
    poly[17] = mono[28];
    poly[18] = mono[29] + mono[30];
    poly[19] = mono[31] + mono[32];
    poly[20] = mono[33] + mono[34];
    poly[21] = mono[35] + mono[36];
    poly[22] = mono[37] + mono[38] + mono[39] + mono[40];
    poly[23] = mono[41] + mono[42] + mono[43] + mono[44];
    poly[24] = mono[45];
    poly[25] = mono[46] + mono[47];
    poly[26] = mono[48] + mono[49];
    poly[27] = mono[50] + mono[51];
    poly[28] = mono[52] + mono[53];
    poly[29] = mono[54] + mono[55] + mono[56] + mono[57];
    poly[30] = mono[58] + mono[59] + mono[60] + mono[61];
    poly[31] = mono[62] + mono[63] + mono[64] + mono[65];
    poly[32] = mono[66];
    poly[33] = poly[1] * poly[2];
    poly[34] = poly[1] * poly[3];
    poly[35] = poly[2] * poly[3];
    poly[36] = poly[1] * poly[4];
    poly[37] = poly[2] * poly[4];
    poly[38] = poly[3] * poly[4];
    poly[39] = poly[1] * poly[5];
    poly[40] = poly[2] * poly[5];
    poly[41] = poly[3] * poly[5];
    poly[42] = poly[4] * poly[5];
    poly[43] = poly[1] * poly[6];
    poly[44] = poly[2] * poly[6];
    poly[45] = poly[3] * poly[6];
    poly[46] = poly[4] * poly[6];
    poly[47] = poly[5] * poly[6];
    poly[48] = poly[1] * poly[7];
    poly[49] = poly[2] * poly[7];
}

fn f_polynomials1(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[50] = poly[3] * poly[7];
    poly[51] = poly[4] * poly[7];
    poly[52] = poly[5] * poly[7];
    poly[53] = poly[6] * poly[7];
    poly[54] = mono[67];
    poly[55] = poly[1] * poly[8];
    poly[56] = poly[2] * poly[8];
    poly[57] = poly[3] * poly[8];
    poly[58] = poly[4] * poly[8];
    poly[59] = poly[5] * poly[8];
    poly[60] = poly[6] * poly[8];
    poly[61] = mono[68] + mono[69];
    poly[62] = mono[70];
    poly[63] = poly[7] * poly[8] - poly[61];
    poly[64] = poly[1] * poly[9];
    poly[65] = poly[2] * poly[9];
    poly[66] = poly[3] * poly[9];
    poly[67] = poly[4] * poly[9];
    poly[68] = poly[5] * poly[9];
    poly[69] = poly[6] * poly[9];
    poly[70] = mono[71] + mono[72];
    poly[71] = mono[73] + mono[74];
    poly[72] = mono[75];
    poly[73] = poly[7] * poly[9] - poly[70];
    poly[74] = poly[8] * poly[9] - poly[71];
    poly[75] = poly[1] * poly[10];
    poly[76] = poly[2] * poly[10];
    poly[77] = poly[3] * poly[10];
    poly[78] = poly[4] * poly[10];
    poly[79] = poly[5] * poly[10];
    poly[80] = poly[6] * poly[10];
    poly[81] = mono[76] + mono[77];
    poly[82] = mono[78] + mono[79];
    poly[83] = mono[80] + mono[81];
    poly[84] = mono[82];
    poly[85] = poly[7] * poly[10] - poly[81];
    poly[86] = poly[8] * poly[10] - poly[82];
    poly[87] = poly[9] * poly[10] - poly[83];
    poly[88] = poly[1] * poly[11];
    poly[89] = poly[2] * poly[11];
    poly[90] = poly[3] * poly[11];
    poly[91] = poly[4] * poly[11];
    poly[92] = poly[5] * poly[11];
    poly[93] = poly[6] * poly[11];
    poly[94] = poly[11] * poly[7];
    poly[95] = poly[11] * poly[8];
    poly[96] = poly[11] * poly[9];
    poly[97] = poly[11] * poly[10];
    poly[98] = poly[1] * poly[12];
    poly[99] = poly[2] * poly[12];
}

fn f_polynomials2(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[100] = poly[3] * poly[12];
    poly[101] = poly[4] * poly[12];
    poly[102] = poly[5] * poly[12];
    poly[103] = poly[6] * poly[12];
    poly[104] = poly[7] * poly[12];
    poly[105] = poly[8] * poly[12];
    poly[106] = poly[9] * poly[12];
    poly[107] = poly[10] * poly[12];
    poly[108] = poly[11] * poly[12];
    poly[109] = mono[83];
    poly[110] = poly[1] * poly[13];
    poly[111] = poly[2] * poly[13];
    poly[112] = poly[3] * poly[13];
    poly[113] = poly[4] * poly[13];
    poly[114] = poly[5] * poly[13];
    poly[115] = poly[6] * poly[13];
    poly[116] = poly[7] * poly[13];
    poly[117] = poly[8] * poly[13];
    poly[118] = poly[9] * poly[13];
    poly[119] = poly[10] * poly[13];
    poly[120] = poly[11] * poly[13];
    poly[121] = mono[84] + mono[85];
    poly[122] = mono[86];
    poly[123] = poly[12] * poly[13] - poly[121];
    poly[124] = poly[1] * poly[14];
    poly[125] = poly[2] * poly[14];
    poly[126] = poly[3] * poly[14];
    poly[127] = poly[4] * poly[14];
    poly[128] = poly[5] * poly[14];
    poly[129] = poly[6] * poly[14];
    poly[130] = poly[7] * poly[14];
    poly[131] = poly[8] * poly[14];
    poly[132] = poly[9] * poly[14];
    poly[133] = poly[10] * poly[14];
    poly[134] = poly[11] * poly[14];
    poly[135] = mono[87] + mono[88];
    poly[136] = mono[89] + mono[90];
    poly[137] = mono[91];
    poly[138] = poly[12] * poly[14] - poly[135];
    poly[139] = poly[13] * poly[14] - poly[136];
    poly[140] = poly[1] * poly[15];
    poly[141] = poly[2] * poly[15];
    poly[142] = poly[3] * poly[15];
    poly[143] = poly[4] * poly[15];
    poly[144] = poly[5] * poly[15];
    poly[145] = poly[6] * poly[15];
    poly[146] = poly[7] * poly[15];
    poly[147] = poly[8] * poly[15];
    poly[148] = poly[9] * poly[15];
    poly[149] = poly[10] * poly[15];
}

fn f_polynomials3(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[150] = poly[11] * poly[15];
    poly[151] = mono[92] + mono[93];
    poly[152] = mono[94] + mono[95];
    poly[153] = mono[96] + mono[97];
    poly[154] = mono[98];
    poly[155] = poly[12] * poly[15] - poly[151];
    poly[156] = poly[13] * poly[15] - poly[152];
    poly[157] = poly[14] * poly[15] - poly[153];
    poly[158] = poly[1] * poly[16];
    poly[159] = poly[2] * poly[16];
    poly[160] = poly[3] * poly[16];
    poly[161] = poly[4] * poly[16];
    poly[162] = poly[5] * poly[16];
    poly[163] = poly[6] * poly[16];
    poly[164] = mono[99] + mono[100] + mono[101] + mono[102];
    poly[165] = mono[103] + mono[104] + mono[105] + mono[106];
    poly[166] = mono[107] + mono[108] + mono[109] + mono[110];
    poly[167] = mono[111] + mono[112] + mono[113] + mono[114];
    poly[168] = poly[7] * poly[16] - poly[164];
    poly[169] = poly[8] * poly[16] - poly[165];
    poly[170] = poly[9] * poly[16] - poly[166];
    poly[171] = poly[10] * poly[16] - poly[167];
    poly[172] = poly[11] * poly[16];
    poly[173] = mono[115] + mono[116] + mono[117] + mono[118];
    poly[174] = mono[119] + mono[120] + mono[121] + mono[122];
    poly[175] = mono[123] + mono[124] + mono[125] + mono[126];
    poly[176] = mono[127] + mono[128] + mono[129] + mono[130];
    poly[177] = mono[131] + mono[132];
    poly[178] = mono[133] + mono[134];
    poly[179] = poly[12] * poly[16] - poly[173];
    poly[180] = poly[13] * poly[16] - poly[174];
    poly[181] = poly[14] * poly[16] - poly[175];
    poly[182] = poly[15] * poly[16] - poly[176];
    poly[183] = mono[135] + mono[136];
    poly[184] = poly[1] * poly[17];
    poly[185] = poly[2] * poly[17];
    poly[186] = poly[3] * poly[17];
    poly[187] = poly[4] * poly[17];
    poly[188] = poly[5] * poly[17];
    poly[189] = poly[6] * poly[17];
    poly[190] = poly[17] * poly[7];
    poly[191] = poly[17] * poly[8];
    poly[192] = poly[17] * poly[9];
    poly[193] = poly[17] * poly[10];
    poly[194] = poly[11] * poly[17];
    poly[195] = poly[17] * poly[12];
    poly[196] = poly[17] * poly[13];
    poly[197] = poly[17] * poly[14];
    poly[198] = poly[17] * poly[15];
    poly[199] = poly[17] * poly[16];
}

fn f_polynomials4(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[200] = poly[1] * poly[18];
    poly[201] = poly[2] * poly[18];
    poly[202] = poly[3] * poly[18];
    poly[203] = poly[4] * poly[18];
    poly[204] = poly[5] * poly[18];
    poly[205] = poly[6] * poly[18];
    poly[206] = poly[7] * poly[18];
    poly[207] = poly[8] * poly[18];
    poly[208] = poly[9] * poly[18];
    poly[209] = poly[10] * poly[18];
    poly[210] = poly[11] * poly[18];
    poly[211] = poly[12] * poly[18];
    poly[212] = poly[13] * poly[18];
    poly[213] = poly[14] * poly[18];
    poly[214] = poly[15] * poly[18];
    poly[215] = poly[16] * poly[18];
    poly[216] = poly[17] * poly[18];
    poly[217] = mono[137];
    poly[218] = poly[1] * poly[19];
    poly[219] = poly[2] * poly[19];
    poly[220] = poly[3] * poly[19];
    poly[221] = poly[4] * poly[19];
    poly[222] = poly[5] * poly[19];
    poly[223] = poly[6] * poly[19];
    poly[224] = poly[7] * poly[19];
    poly[225] = poly[8] * poly[19];
    poly[226] = poly[9] * poly[19];
    poly[227] = poly[10] * poly[19];
    poly[228] = poly[11] * poly[19];
    poly[229] = poly[12] * poly[19];
    poly[230] = poly[13] * poly[19];
    poly[231] = poly[14] * poly[19];
    poly[232] = poly[15] * poly[19];
    poly[233] = poly[16] * poly[19];
    poly[234] = poly[17] * poly[19];
    poly[235] = mono[138] + mono[139];
    poly[236] = mono[140];
    poly[237] = poly[18] * poly[19] - poly[235];
    poly[238] = poly[1] * poly[20];
    poly[239] = poly[2] * poly[20];
    poly[240] = poly[3] * poly[20];
    poly[241] = poly[4] * poly[20];
    poly[242] = poly[5] * poly[20];
    poly[243] = poly[6] * poly[20];
    poly[244] = poly[7] * poly[20];
    poly[245] = poly[8] * poly[20];
    poly[246] = poly[9] * poly[20];
    poly[247] = poly[10] * poly[20];
    poly[248] = poly[11] * poly[20];
    poly[249] = poly[12] * poly[20];
}

fn f_polynomials5(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[250] = poly[13] * poly[20];
    poly[251] = poly[14] * poly[20];
    poly[252] = poly[15] * poly[20];
    poly[253] = poly[16] * poly[20];
    poly[254] = poly[17] * poly[20];
    poly[255] = mono[141] + mono[142];
    poly[256] = mono[143] + mono[144];
    poly[257] = mono[145];
    poly[258] = poly[18] * poly[20] - poly[255];
    poly[259] = poly[19] * poly[20] - poly[256];
    poly[260] = poly[1] * poly[21];
    poly[261] = poly[2] * poly[21];
    poly[262] = poly[3] * poly[21];
    poly[263] = poly[4] * poly[21];
    poly[264] = poly[5] * poly[21];
    poly[265] = poly[6] * poly[21];
    poly[266] = poly[7] * poly[21];
    poly[267] = poly[8] * poly[21];
    poly[268] = poly[9] * poly[21];
    poly[269] = poly[10] * poly[21];
    poly[270] = poly[11] * poly[21];
    poly[271] = poly[12] * poly[21];
    poly[272] = poly[13] * poly[21];
    poly[273] = poly[14] * poly[21];
    poly[274] = poly[15] * poly[21];
    poly[275] = poly[16] * poly[21];
    poly[276] = poly[17] * poly[21];
    poly[277] = mono[146] + mono[147];
    poly[278] = mono[148] + mono[149];
    poly[279] = mono[150] + mono[151];
    poly[280] = mono[152];
    poly[281] = poly[18] * poly[21] - poly[277];
    poly[282] = poly[19] * poly[21] - poly[278];
    poly[283] = poly[20] * poly[21] - poly[279];
    poly[284] = poly[1] * poly[22];
    poly[285] = poly[2] * poly[22];
    poly[286] = poly[3] * poly[22];
    poly[287] = poly[4] * poly[22];
    poly[288] = poly[5] * poly[22];
    poly[289] = poly[6] * poly[22];
    poly[290] = mono[153] + mono[154] + mono[155] + mono[156];
    poly[291] = mono[157] + mono[158] + mono[159] + mono[160];
    poly[292] = mono[161] + mono[162] + mono[163] + mono[164];
    poly[293] = mono[165] + mono[166] + mono[167] + mono[168];
    poly[294] = poly[7] * poly[22] - poly[290];
    poly[295] = poly[8] * poly[22] - poly[291];
    poly[296] = poly[9] * poly[22] - poly[292];
    poly[297] = poly[10] * poly[22] - poly[293];
    poly[298] = poly[11] * poly[22];
    poly[299] = poly[12] * poly[22];
}

fn f_polynomials6(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[300] = poly[13] * poly[22];
    poly[301] = poly[14] * poly[22];
    poly[302] = poly[15] * poly[22];
    poly[303] = mono[169]
        + mono[170]
        + mono[171]
        + mono[172]
        + mono[173]
        + mono[174]
        + mono[175]
        + mono[176];
    poly[304] = poly[16] * poly[22] - poly[303];
    poly[305] = poly[17] * poly[22];
    poly[306] = mono[177] + mono[178] + mono[179] + mono[180];
    poly[307] = mono[181] + mono[182] + mono[183] + mono[184];
    poly[308] = mono[185] + mono[186] + mono[187] + mono[188];
    poly[309] = mono[189] + mono[190] + mono[191] + mono[192];
    poly[310] = mono[193] + mono[194];
    poly[311] = mono[195] + mono[196];
    poly[312] = poly[18] * poly[22] - poly[306];
    poly[313] = poly[19] * poly[22] - poly[307];
    poly[314] = poly[20] * poly[22] - poly[308];
    poly[315] = poly[21] * poly[22] - poly[309];
    poly[316] = mono[197] + mono[198];
    poly[317] = poly[1] * poly[23];
    poly[318] = poly[2] * poly[23];
    poly[319] = poly[3] * poly[23];
    poly[320] = poly[4] * poly[23];
    poly[321] = poly[5] * poly[23];
    poly[322] = poly[6] * poly[23];
    poly[323] = poly[7] * poly[23];
    poly[324] = poly[8] * poly[23];
    poly[325] = poly[9] * poly[23];
    poly[326] = poly[10] * poly[23];
    poly[327] = poly[11] * poly[23];
    poly[328] = mono[199] + mono[200] + mono[201] + mono[202];
    poly[329] = mono[203] + mono[204] + mono[205] + mono[206];
    poly[330] = mono[207] + mono[208] + mono[209] + mono[210];
    poly[331] = mono[211] + mono[212] + mono[213] + mono[214];
    poly[332] = mono[215]
        + mono[216]
        + mono[217]
        + mono[218]
        + mono[219]
        + mono[220]
        + mono[221]
        + mono[222];
    poly[333] = poly[12] * poly[23] - poly[328];
    poly[334] = poly[13] * poly[23] - poly[329];
    poly[335] = poly[14] * poly[23] - poly[330];
    poly[336] = poly[15] * poly[23] - poly[331];
    poly[337] = poly[16] * poly[23] - poly[332];
    poly[338] = poly[17] * poly[23];
    poly[339] = mono[223] + mono[224] + mono[225] + mono[226];
    poly[340] = mono[227] + mono[228] + mono[229] + mono[230];
    poly[341] = mono[231] + mono[232] + mono[233] + mono[234];
    poly[342] = mono[235] + mono[236] + mono[237] + mono[238];
    poly[343] = mono[239]
        + mono[240]
        + mono[241]
        + mono[242]
        + mono[243]
        + mono[244]
        + mono[245]
        + mono[246];
    poly[344] = mono[247] + mono[248];
    poly[345] = mono[249] + mono[250];
    poly[346] = poly[18] * poly[23] - poly[339];
    poly[347] = poly[19] * poly[23] - poly[340];
    poly[348] = poly[20] * poly[23] - poly[341];
    poly[349] = poly[21] * poly[23] - poly[342];
}

fn f_polynomials7(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[350] = poly[22] * poly[23] - poly[343];
    poly[351] = mono[251] + mono[252];
    poly[352] = poly[1] * poly[24];
    poly[353] = poly[2] * poly[24];
    poly[354] = poly[3] * poly[24];
    poly[355] = poly[4] * poly[24];
    poly[356] = poly[5] * poly[24];
    poly[357] = poly[6] * poly[24];
    poly[358] = poly[24] * poly[7];
    poly[359] = poly[24] * poly[8];
    poly[360] = poly[24] * poly[9];
    poly[361] = poly[24] * poly[10];
    poly[362] = poly[11] * poly[24];
    poly[363] = poly[24] * poly[12];
    poly[364] = poly[24] * poly[13];
    poly[365] = poly[24] * poly[14];
    poly[366] = poly[24] * poly[15];
    poly[367] = poly[24] * poly[16];
    poly[368] = poly[17] * poly[24];
    poly[369] = poly[24] * poly[18];
    poly[370] = poly[24] * poly[19];
    poly[371] = poly[24] * poly[20];
    poly[372] = poly[24] * poly[21];
    poly[373] = poly[24] * poly[22];
    poly[374] = poly[24] * poly[23];
    poly[375] = poly[1] * poly[25];
    poly[376] = poly[2] * poly[25];
    poly[377] = poly[3] * poly[25];
    poly[378] = poly[4] * poly[25];
    poly[379] = poly[5] * poly[25];
    poly[380] = poly[6] * poly[25];
    poly[381] = poly[7] * poly[25];
    poly[382] = poly[8] * poly[25];
    poly[383] = poly[9] * poly[25];
    poly[384] = poly[10] * poly[25];
    poly[385] = poly[11] * poly[25];
    poly[386] = poly[12] * poly[25];
    poly[387] = poly[13] * poly[25];
    poly[388] = poly[14] * poly[25];
    poly[389] = poly[15] * poly[25];
    poly[390] = poly[16] * poly[25];
    poly[391] = poly[17] * poly[25];
    poly[392] = poly[18] * poly[25];
    poly[393] = poly[19] * poly[25];
    poly[394] = poly[20] * poly[25];
    poly[395] = poly[21] * poly[25];
    poly[396] = poly[22] * poly[25];
    poly[397] = poly[23] * poly[25];
    poly[398] = poly[24] * poly[25];
    poly[399] = mono[253];
}

fn f_polynomials8(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[400] = poly[1] * poly[26];
    poly[401] = poly[2] * poly[26];
    poly[402] = poly[3] * poly[26];
    poly[403] = poly[4] * poly[26];
    poly[404] = poly[5] * poly[26];
    poly[405] = poly[6] * poly[26];
    poly[406] = poly[7] * poly[26];
    poly[407] = poly[8] * poly[26];
    poly[408] = poly[9] * poly[26];
    poly[409] = poly[10] * poly[26];
    poly[410] = poly[11] * poly[26];
    poly[411] = poly[12] * poly[26];
    poly[412] = poly[13] * poly[26];
    poly[413] = poly[14] * poly[26];
    poly[414] = poly[15] * poly[26];
    poly[415] = poly[16] * poly[26];
    poly[416] = poly[17] * poly[26];
    poly[417] = poly[18] * poly[26];
    poly[418] = poly[19] * poly[26];
    poly[419] = poly[20] * poly[26];
    poly[420] = poly[21] * poly[26];
    poly[421] = poly[22] * poly[26];
    poly[422] = poly[23] * poly[26];
    poly[423] = poly[24] * poly[26];
    poly[424] = mono[254] + mono[255];
    poly[425] = mono[256];
    poly[426] = poly[25] * poly[26] - poly[424];
    poly[427] = poly[1] * poly[27];
    poly[428] = poly[2] * poly[27];
    poly[429] = poly[3] * poly[27];
    poly[430] = poly[4] * poly[27];
    poly[431] = poly[5] * poly[27];
    poly[432] = poly[6] * poly[27];
    poly[433] = poly[7] * poly[27];
    poly[434] = poly[8] * poly[27];
    poly[435] = poly[9] * poly[27];
    poly[436] = poly[10] * poly[27];
    poly[437] = poly[11] * poly[27];
    poly[438] = poly[12] * poly[27];
    poly[439] = poly[13] * poly[27];
    poly[440] = poly[14] * poly[27];
    poly[441] = poly[15] * poly[27];
    poly[442] = poly[16] * poly[27];
    poly[443] = poly[17] * poly[27];
    poly[444] = poly[18] * poly[27];
    poly[445] = poly[19] * poly[27];
    poly[446] = poly[20] * poly[27];
    poly[447] = poly[21] * poly[27];
    poly[448] = poly[22] * poly[27];
    poly[449] = poly[23] * poly[27];
}

fn f_polynomials9(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[450] = poly[24] * poly[27];
    poly[451] = mono[257] + mono[258];
    poly[452] = mono[259] + mono[260];
    poly[453] = mono[261];
    poly[454] = poly[25] * poly[27] - poly[451];
    poly[455] = poly[26] * poly[27] - poly[452];
    poly[456] = poly[1] * poly[28];
    poly[457] = poly[2] * poly[28];
    poly[458] = poly[3] * poly[28];
    poly[459] = poly[4] * poly[28];
    poly[460] = poly[5] * poly[28];
    poly[461] = poly[6] * poly[28];
    poly[462] = poly[7] * poly[28];
    poly[463] = poly[8] * poly[28];
    poly[464] = poly[9] * poly[28];
    poly[465] = poly[10] * poly[28];
    poly[466] = poly[11] * poly[28];
    poly[467] = poly[12] * poly[28];
    poly[468] = poly[13] * poly[28];
    poly[469] = poly[14] * poly[28];
    poly[470] = poly[15] * poly[28];
    poly[471] = poly[16] * poly[28];
    poly[472] = poly[17] * poly[28];
    poly[473] = poly[18] * poly[28];
    poly[474] = poly[19] * poly[28];
    poly[475] = poly[20] * poly[28];
    poly[476] = poly[21] * poly[28];
    poly[477] = poly[22] * poly[28];
    poly[478] = poly[23] * poly[28];
    poly[479] = poly[24] * poly[28];
    poly[480] = mono[262] + mono[263];
    poly[481] = mono[264] + mono[265];
    poly[482] = mono[266] + mono[267];
    poly[483] = mono[268];
    poly[484] = poly[25] * poly[28] - poly[480];
    poly[485] = poly[26] * poly[28] - poly[481];
    poly[486] = poly[27] * poly[28] - poly[482];
    poly[487] = poly[1] * poly[29];
    poly[488] = poly[2] * poly[29];
    poly[489] = poly[3] * poly[29];
    poly[490] = poly[4] * poly[29];
    poly[491] = poly[5] * poly[29];
    poly[492] = poly[6] * poly[29];
    poly[493] = mono[269] + mono[270] + mono[271] + mono[272];
    poly[494] = mono[273] + mono[274] + mono[275] + mono[276];
    poly[495] = mono[277] + mono[278] + mono[279] + mono[280];
    poly[496] = mono[281] + mono[282] + mono[283] + mono[284];
    poly[497] = poly[7] * poly[29] - poly[493];
    poly[498] = poly[8] * poly[29] - poly[494];
    poly[499] = poly[9] * poly[29] - poly[495];
}

fn f_polynomials10(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[500] = poly[10] * poly[29] - poly[496];
    poly[501] = poly[11] * poly[29];
    poly[502] = poly[12] * poly[29];
    poly[503] = poly[13] * poly[29];
    poly[504] = poly[14] * poly[29];
    poly[505] = poly[15] * poly[29];
    poly[506] = mono[285]
        + mono[286]
        + mono[287]
        + mono[288]
        + mono[289]
        + mono[290]
        + mono[291]
        + mono[292];
    poly[507] = poly[16] * poly[29] - poly[506];
    poly[508] = poly[17] * poly[29];
    poly[509] = poly[18] * poly[29];
    poly[510] = poly[19] * poly[29];
    poly[511] = poly[20] * poly[29];
    poly[512] = poly[21] * poly[29];
    poly[513] = mono[293]
        + mono[294]
        + mono[295]
        + mono[296]
        + mono[297]
        + mono[298]
        + mono[299]
        + mono[300];
    poly[514] = poly[22] * poly[29] - poly[513];
    poly[515] = poly[23] * poly[29];
    poly[516] = poly[24] * poly[29];
    poly[517] = mono[301] + mono[302] + mono[303] + mono[304];
    poly[518] = mono[305] + mono[306] + mono[307] + mono[308];
    poly[519] = mono[309] + mono[310] + mono[311] + mono[312];
    poly[520] = mono[313] + mono[314] + mono[315] + mono[316];
    poly[521] = mono[317] + mono[318];
    poly[522] = mono[319] + mono[320];
    poly[523] = poly[25] * poly[29] - poly[517];
    poly[524] = poly[26] * poly[29] - poly[518];
    poly[525] = poly[27] * poly[29] - poly[519];
    poly[526] = poly[28] * poly[29] - poly[520];
    poly[527] = mono[321] + mono[322];
    poly[528] = poly[1] * poly[30];
    poly[529] = poly[2] * poly[30];
    poly[530] = poly[3] * poly[30];
    poly[531] = poly[4] * poly[30];
    poly[532] = poly[5] * poly[30];
    poly[533] = poly[6] * poly[30];
    poly[534] = poly[7] * poly[30];
    poly[535] = poly[8] * poly[30];
    poly[536] = poly[9] * poly[30];
    poly[537] = poly[10] * poly[30];
    poly[538] = poly[11] * poly[30];
    poly[539] = mono[323] + mono[324] + mono[325] + mono[326];
    poly[540] = mono[327] + mono[328] + mono[329] + mono[330];
    poly[541] = mono[331] + mono[332] + mono[333] + mono[334];
    poly[542] = mono[335] + mono[336] + mono[337] + mono[338];
    poly[543] = mono[339]
        + mono[340]
        + mono[341]
        + mono[342]
        + mono[343]
        + mono[344]
        + mono[345]
        + mono[346];
    poly[544] = poly[12] * poly[30] - poly[539];
    poly[545] = poly[13] * poly[30] - poly[540];
    poly[546] = poly[14] * poly[30] - poly[541];
    poly[547] = poly[15] * poly[30] - poly[542];
    poly[548] = poly[16] * poly[30] - poly[543];
    poly[549] = poly[17] * poly[30];
}

fn f_polynomials11(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[550] = poly[18] * poly[30];
    poly[551] = poly[19] * poly[30];
    poly[552] = poly[20] * poly[30];
    poly[553] = poly[21] * poly[30];
    poly[554] = poly[22] * poly[30];
    poly[555] = mono[347]
        + mono[348]
        + mono[349]
        + mono[350]
        + mono[351]
        + mono[352]
        + mono[353]
        + mono[354];
    poly[556] = poly[23] * poly[30] - poly[555];
    poly[557] = poly[24] * poly[30];
    poly[558] = mono[355] + mono[356] + mono[357] + mono[358];
    poly[559] = mono[359] + mono[360] + mono[361] + mono[362];
    poly[560] = mono[363] + mono[364] + mono[365] + mono[366];
    poly[561] = mono[367] + mono[368] + mono[369] + mono[370];
    poly[562] = mono[371]
        + mono[372]
        + mono[373]
        + mono[374]
        + mono[375]
        + mono[376]
        + mono[377]
        + mono[378];
    poly[563] = mono[379] + mono[380];
    poly[564] = mono[381] + mono[382];
    poly[565] = poly[25] * poly[30] - poly[558];
    poly[566] = poly[26] * poly[30] - poly[559];
    poly[567] = poly[27] * poly[30] - poly[560];
    poly[568] = poly[28] * poly[30] - poly[561];
    poly[569] = poly[29] * poly[30] - poly[562];
    poly[570] = mono[383] + mono[384];
    poly[571] = poly[1] * poly[31];
    poly[572] = poly[2] * poly[31];
    poly[573] = poly[3] * poly[31];
    poly[574] = poly[4] * poly[31];
    poly[575] = poly[5] * poly[31];
    poly[576] = poly[6] * poly[31];
    poly[577] = poly[7] * poly[31];
    poly[578] = poly[8] * poly[31];
    poly[579] = poly[9] * poly[31];
    poly[580] = poly[10] * poly[31];
    poly[581] = poly[11] * poly[31];
    poly[582] = poly[12] * poly[31];
    poly[583] = poly[13] * poly[31];
    poly[584] = poly[14] * poly[31];
    poly[585] = poly[15] * poly[31];
    poly[586] = poly[16] * poly[31];
    poly[587] = poly[17] * poly[31];
    poly[588] = mono[385] + mono[386] + mono[387] + mono[388];
    poly[589] = mono[389] + mono[390] + mono[391] + mono[392];
    poly[590] = mono[393] + mono[394] + mono[395] + mono[396];
    poly[591] = mono[397] + mono[398] + mono[399] + mono[400];
    poly[592] = mono[401]
        + mono[402]
        + mono[403]
        + mono[404]
        + mono[405]
        + mono[406]
        + mono[407]
        + mono[408];
    poly[593] = mono[409]
        + mono[410]
        + mono[411]
        + mono[412]
        + mono[413]
        + mono[414]
        + mono[415]
        + mono[416];
    poly[594] = poly[18] * poly[31] - poly[588];
    poly[595] = poly[19] * poly[31] - poly[589];
    poly[596] = poly[20] * poly[31] - poly[590];
    poly[597] = poly[21] * poly[31] - poly[591];
    poly[598] = poly[22] * poly[31] - poly[592];
    poly[599] = poly[23] * poly[31] - poly[593];
}

fn f_polynomials12(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[600] = poly[24] * poly[31];
    poly[601] = mono[417] + mono[418] + mono[419] + mono[420];
    poly[602] = mono[421] + mono[422] + mono[423] + mono[424];
    poly[603] = mono[425] + mono[426] + mono[427] + mono[428];
    poly[604] = mono[429] + mono[430] + mono[431] + mono[432];
    poly[605] = mono[433]
        + mono[434]
        + mono[435]
        + mono[436]
        + mono[437]
        + mono[438]
        + mono[439]
        + mono[440];
    poly[606] = mono[441]
        + mono[442]
        + mono[443]
        + mono[444]
        + mono[445]
        + mono[446]
        + mono[447]
        + mono[448];
    poly[607] = mono[449] + mono[450];
    poly[608] = mono[451] + mono[452];
    poly[609] = poly[25] * poly[31] - poly[601];
    poly[610] = poly[26] * poly[31] - poly[602];
    poly[611] = poly[27] * poly[31] - poly[603];
    poly[612] = poly[28] * poly[31] - poly[604];
    poly[613] = poly[29] * poly[31] - poly[605];
    poly[614] = poly[30] * poly[31] - poly[606];
    poly[615] = mono[453] + mono[454];
    poly[616] = poly[1] * poly[32];
    poly[617] = poly[2] * poly[32];
    poly[618] = poly[3] * poly[32];
    poly[619] = poly[4] * poly[32];
    poly[620] = poly[5] * poly[32];
    poly[621] = poly[6] * poly[32];
    poly[622] = poly[32] * poly[7];
    poly[623] = poly[32] * poly[8];
    poly[624] = poly[32] * poly[9];
    poly[625] = poly[32] * poly[10];
    poly[626] = poly[11] * poly[32];
    poly[627] = poly[32] * poly[12];
    poly[628] = poly[32] * poly[13];
    poly[629] = poly[32] * poly[14];
    poly[630] = poly[32] * poly[15];
    poly[631] = poly[32] * poly[16];
    poly[632] = poly[17] * poly[32];
    poly[633] = poly[32] * poly[18];
    poly[634] = poly[32] * poly[19];
    poly[635] = poly[32] * poly[20];
    poly[636] = poly[32] * poly[21];
    poly[637] = poly[32] * poly[22];
    poly[638] = poly[32] * poly[23];
    poly[639] = poly[24] * poly[32];
    poly[640] = poly[32] * poly[25];
    poly[641] = poly[32] * poly[26];
    poly[642] = poly[32] * poly[27];
    poly[643] = poly[32] * poly[28];
    poly[644] = poly[32] * poly[29];
    poly[645] = poly[32] * poly[30];
    poly[646] = poly[32] * poly[31];
    poly[647] = poly[1] * poly[1];
    poly[648] = poly[2] * poly[2];
    poly[649] = poly[3] * poly[3];
}

fn f_polynomials13(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[650] = poly[4] * poly[4];
    poly[651] = poly[5] * poly[5];
    poly[652] = poly[6] * poly[6];
    poly[653] = poly[7] * poly[7] - poly[54] - poly[54];
    poly[654] = poly[8] * poly[8] - poly[62] - poly[62];
    poly[655] = poly[9] * poly[9] - poly[72] - poly[72];
    poly[656] = poly[10] * poly[10] - poly[84] - poly[84];
    poly[657] = poly[11] * poly[11];
    poly[658] = poly[12] * poly[12] - poly[109] - poly[109];
    poly[659] = poly[13] * poly[13] - poly[122] - poly[122];
    poly[660] = poly[14] * poly[14] - poly[137] - poly[137];
    poly[661] = poly[15] * poly[15] - poly[154] - poly[154];
    poly[662] =
        poly[16] * poly[16] - poly[183] - poly[178] - poly[177] - poly[183] - poly[178] - poly[177];
    poly[663] = poly[17] * poly[17];
    poly[664] = poly[18] * poly[18] - poly[217] - poly[217];
    poly[665] = poly[19] * poly[19] - poly[236] - poly[236];
    poly[666] = poly[20] * poly[20] - poly[257] - poly[257];
    poly[667] = poly[21] * poly[21] - poly[280] - poly[280];
    poly[668] =
        poly[22] * poly[22] - poly[316] - poly[311] - poly[310] - poly[316] - poly[311] - poly[310];
    poly[669] =
        poly[23] * poly[23] - poly[351] - poly[345] - poly[344] - poly[351] - poly[345] - poly[344];
    poly[670] = poly[24] * poly[24];
    poly[671] = poly[25] * poly[25] - poly[399] - poly[399];
    poly[672] = poly[26] * poly[26] - poly[425] - poly[425];
    poly[673] = poly[27] * poly[27] - poly[453] - poly[453];
    poly[674] = poly[28] * poly[28] - poly[483] - poly[483];
    poly[675] =
        poly[29] * poly[29] - poly[527] - poly[522] - poly[521] - poly[527] - poly[522] - poly[521];
    poly[676] =
        poly[30] * poly[30] - poly[570] - poly[564] - poly[563] - poly[570] - poly[564] - poly[563];
    poly[677] =
        poly[31] * poly[31] - poly[615] - poly[608] - poly[607] - poly[615] - poly[608] - poly[607];
    poly[678] = poly[32] * poly[32];
    poly[679] = poly[1] * poly[35];
    poly[680] = poly[1] * poly[37];
    poly[681] = poly[1] * poly[38];
    poly[682] = poly[2] * poly[38];
    poly[683] = poly[1] * poly[40];
    poly[684] = poly[1] * poly[41];
    poly[685] = poly[2] * poly[41];
    poly[686] = poly[1] * poly[42];
    poly[687] = poly[2] * poly[42];
    poly[688] = poly[3] * poly[42];
    poly[689] = poly[1] * poly[44];
    poly[690] = poly[1] * poly[45];
    poly[691] = poly[2] * poly[45];
    poly[692] = poly[1] * poly[46];
    poly[693] = poly[2] * poly[46];
    poly[694] = poly[3] * poly[46];
    poly[695] = poly[1] * poly[47];
    poly[696] = poly[2] * poly[47];
    poly[697] = poly[3] * poly[47];
    poly[698] = poly[4] * poly[47];
    poly[699] = poly[1] * poly[49];
}

fn f_polynomials14(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[700] = poly[1] * poly[50];
    poly[701] = poly[2] * poly[50];
    poly[702] = poly[1] * poly[51];
    poly[703] = poly[2] * poly[51];
    poly[704] = poly[3] * poly[51];
    poly[705] = poly[1] * poly[52];
    poly[706] = poly[2] * poly[52];
    poly[707] = poly[3] * poly[52];
    poly[708] = poly[4] * poly[52];
    poly[709] = poly[1] * poly[53];
    poly[710] = poly[2] * poly[53];
    poly[711] = poly[3] * poly[53];
    poly[712] = poly[4] * poly[53];
    poly[713] = poly[5] * poly[53];
    poly[714] = poly[1] * poly[54];
    poly[715] = poly[2] * poly[54];
    poly[716] = poly[3] * poly[54];
    poly[717] = poly[4] * poly[54];
    poly[718] = poly[5] * poly[54];
    poly[719] = poly[6] * poly[54];
    poly[720] = poly[1] * poly[56];
    poly[721] = poly[1] * poly[57];
    poly[722] = poly[2] * poly[57];
    poly[723] = poly[1] * poly[58];
    poly[724] = poly[2] * poly[58];
    poly[725] = poly[3] * poly[58];
    poly[726] = poly[1] * poly[59];
    poly[727] = poly[2] * poly[59];
    poly[728] = poly[3] * poly[59];
    poly[729] = poly[4] * poly[59];
    poly[730] = poly[1] * poly[60];
    poly[731] = poly[2] * poly[60];
    poly[732] = poly[3] * poly[60];
    poly[733] = poly[4] * poly[60];
    poly[734] = poly[5] * poly[60];
    poly[735] = poly[1] * poly[61];
    poly[736] = poly[2] * poly[61];
    poly[737] = poly[3] * poly[61];
    poly[738] = poly[4] * poly[61];
    poly[739] = poly[5] * poly[61];
    poly[740] = poly[6] * poly[61];
    poly[741] = poly[1] * poly[62];
    poly[742] = poly[2] * poly[62];
    poly[743] = poly[3] * poly[62];
    poly[744] = poly[4] * poly[62];
    poly[745] = poly[5] * poly[62];
    poly[746] = poly[6] * poly[62];
    poly[747] = poly[1] * poly[63];
    poly[748] = poly[2] * poly[63];
    poly[749] = poly[3] * poly[63];
}

fn f_polynomials15(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[750] = poly[4] * poly[63];
    poly[751] = poly[5] * poly[63];
    poly[752] = poly[6] * poly[63];
    poly[753] = poly[54] * poly[8];
    poly[754] = poly[62] * poly[7];
    poly[755] = poly[1] * poly[65];
    poly[756] = poly[1] * poly[66];
    poly[757] = poly[2] * poly[66];
    poly[758] = poly[1] * poly[67];
    poly[759] = poly[2] * poly[67];
    poly[760] = poly[3] * poly[67];
    poly[761] = poly[1] * poly[68];
    poly[762] = poly[2] * poly[68];
    poly[763] = poly[3] * poly[68];
    poly[764] = poly[4] * poly[68];
    poly[765] = poly[1] * poly[69];
    poly[766] = poly[2] * poly[69];
    poly[767] = poly[3] * poly[69];
    poly[768] = poly[4] * poly[69];
    poly[769] = poly[5] * poly[69];
    poly[770] = poly[1] * poly[70];
    poly[771] = poly[2] * poly[70];
    poly[772] = poly[3] * poly[70];
    poly[773] = poly[4] * poly[70];
    poly[774] = poly[5] * poly[70];
    poly[775] = poly[6] * poly[70];
    poly[776] = poly[1] * poly[71];
    poly[777] = poly[2] * poly[71];
    poly[778] = poly[3] * poly[71];
    poly[779] = poly[4] * poly[71];
    poly[780] = poly[5] * poly[71];
    poly[781] = poly[6] * poly[71];
    poly[782] = mono[455] + mono[456];
    poly[783] = poly[1] * poly[72];
    poly[784] = poly[2] * poly[72];
    poly[785] = poly[3] * poly[72];
    poly[786] = poly[4] * poly[72];
    poly[787] = poly[5] * poly[72];
    poly[788] = poly[6] * poly[72];
    poly[789] = poly[1] * poly[73];
    poly[790] = poly[2] * poly[73];
    poly[791] = poly[3] * poly[73];
    poly[792] = poly[4] * poly[73];
    poly[793] = poly[5] * poly[73];
    poly[794] = poly[6] * poly[73];
    poly[795] = poly[54] * poly[9];
    poly[796] = poly[7] * poly[71] - poly[782];
    poly[797] = poly[72] * poly[7];
    poly[798] = poly[1] * poly[74];
    poly[799] = poly[2] * poly[74];
}

fn f_polynomials16(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[800] = poly[3] * poly[74];
    poly[801] = poly[4] * poly[74];
    poly[802] = poly[5] * poly[74];
    poly[803] = poly[6] * poly[74];
    poly[804] = poly[8] * poly[70] - poly[782];
    poly[805] = poly[62] * poly[9];
    poly[806] = poly[72] * poly[8];
    poly[807] = poly[7] * poly[74] - poly[804];
    poly[808] = poly[1] * poly[76];
    poly[809] = poly[1] * poly[77];
    poly[810] = poly[2] * poly[77];
    poly[811] = poly[1] * poly[78];
    poly[812] = poly[2] * poly[78];
    poly[813] = poly[3] * poly[78];
    poly[814] = poly[1] * poly[79];
    poly[815] = poly[2] * poly[79];
    poly[816] = poly[3] * poly[79];
    poly[817] = poly[4] * poly[79];
    poly[818] = poly[1] * poly[80];
    poly[819] = poly[2] * poly[80];
    poly[820] = poly[3] * poly[80];
    poly[821] = poly[4] * poly[80];
    poly[822] = poly[5] * poly[80];
    poly[823] = poly[1] * poly[81];
    poly[824] = poly[2] * poly[81];
    poly[825] = poly[3] * poly[81];
    poly[826] = poly[4] * poly[81];
    poly[827] = poly[5] * poly[81];
    poly[828] = poly[6] * poly[81];
    poly[829] = poly[1] * poly[82];
    poly[830] = poly[2] * poly[82];
    poly[831] = poly[3] * poly[82];
    poly[832] = poly[4] * poly[82];
    poly[833] = poly[5] * poly[82];
    poly[834] = poly[6] * poly[82];
    poly[835] = mono[457] + mono[458];
    poly[836] = poly[1] * poly[83];
    poly[837] = poly[2] * poly[83];
    poly[838] = poly[3] * poly[83];
    poly[839] = poly[4] * poly[83];
    poly[840] = poly[5] * poly[83];
    poly[841] = poly[6] * poly[83];
    poly[842] = mono[459] + mono[460];
    poly[843] = mono[461] + mono[462];
    poly[844] = poly[1] * poly[84];
    poly[845] = poly[2] * poly[84];
    poly[846] = poly[3] * poly[84];
    poly[847] = poly[4] * poly[84];
    poly[848] = poly[5] * poly[84];
    poly[849] = poly[6] * poly[84];
}

fn f_polynomials17(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[850] = poly[1] * poly[85];
    poly[851] = poly[2] * poly[85];
    poly[852] = poly[3] * poly[85];
    poly[853] = poly[4] * poly[85];
    poly[854] = poly[5] * poly[85];
    poly[855] = poly[6] * poly[85];
    poly[856] = poly[54] * poly[10];
    poly[857] = poly[7] * poly[82] - poly[835];
    poly[858] = poly[7] * poly[83] - poly[842];
    poly[859] = poly[84] * poly[7];
    poly[860] = poly[1] * poly[86];
    poly[861] = poly[2] * poly[86];
    poly[862] = poly[3] * poly[86];
    poly[863] = poly[4] * poly[86];
    poly[864] = poly[5] * poly[86];
    poly[865] = poly[6] * poly[86];
    poly[866] = poly[8] * poly[81] - poly[835];
    poly[867] = poly[62] * poly[10];
    poly[868] = poly[8] * poly[83] - poly[843];
    poly[869] = poly[84] * poly[8];
    poly[870] = poly[7] * poly[86] - poly[866];
    poly[871] = poly[1] * poly[87];
    poly[872] = poly[2] * poly[87];
    poly[873] = poly[3] * poly[87];
    poly[874] = poly[4] * poly[87];
    poly[875] = poly[5] * poly[87];
    poly[876] = poly[6] * poly[87];
    poly[877] = poly[9] * poly[81] - poly[842];
    poly[878] = poly[9] * poly[82] - poly[843];
    poly[879] = poly[72] * poly[10];
    poly[880] = poly[84] * poly[9];
    poly[881] = poly[7] * poly[87] - poly[877];
    poly[882] = poly[8] * poly[87] - poly[878];
    poly[883] = poly[1] * poly[89];
    poly[884] = poly[1] * poly[90];
    poly[885] = poly[2] * poly[90];
    poly[886] = poly[1] * poly[91];
    poly[887] = poly[2] * poly[91];
    poly[888] = poly[3] * poly[91];
    poly[889] = poly[1] * poly[92];
    poly[890] = poly[2] * poly[92];
    poly[891] = poly[3] * poly[92];
    poly[892] = poly[4] * poly[92];
    poly[893] = poly[1] * poly[93];
    poly[894] = poly[2] * poly[93];
    poly[895] = poly[3] * poly[93];
    poly[896] = poly[4] * poly[93];
    poly[897] = poly[5] * poly[93];
    poly[898] = poly[1] * poly[94];
    poly[899] = poly[2] * poly[94];
}

fn f_polynomials18(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[900] = poly[3] * poly[94];
    poly[901] = poly[4] * poly[94];
    poly[902] = poly[5] * poly[94];
    poly[903] = poly[6] * poly[94];
    poly[904] = poly[11] * poly[54];
    poly[905] = poly[1] * poly[95];
    poly[906] = poly[2] * poly[95];
    poly[907] = poly[3] * poly[95];
    poly[908] = poly[4] * poly[95];
    poly[909] = poly[5] * poly[95];
    poly[910] = poly[6] * poly[95];
    poly[911] = poly[11] * poly[61];
    poly[912] = poly[11] * poly[62];
    poly[913] = poly[11] * poly[63];
    poly[914] = poly[1] * poly[96];
    poly[915] = poly[2] * poly[96];
    poly[916] = poly[3] * poly[96];
    poly[917] = poly[4] * poly[96];
    poly[918] = poly[5] * poly[96];
    poly[919] = poly[6] * poly[96];
    poly[920] = poly[11] * poly[70];
    poly[921] = poly[11] * poly[71];
    poly[922] = poly[11] * poly[72];
    poly[923] = poly[11] * poly[73];
    poly[924] = poly[11] * poly[74];
    poly[925] = poly[1] * poly[97];
    poly[926] = poly[2] * poly[97];
    poly[927] = poly[3] * poly[97];
    poly[928] = poly[4] * poly[97];
    poly[929] = poly[5] * poly[97];
    poly[930] = poly[6] * poly[97];
    poly[931] = poly[11] * poly[81];
    poly[932] = poly[11] * poly[82];
    poly[933] = poly[11] * poly[83];
    poly[934] = poly[11] * poly[84];
    poly[935] = poly[11] * poly[85];
    poly[936] = poly[11] * poly[86];
    poly[937] = poly[11] * poly[87];
    poly[938] = poly[1] * poly[99];
    poly[939] = poly[1] * poly[100];
    poly[940] = poly[2] * poly[100];
    poly[941] = poly[1] * poly[101];
    poly[942] = poly[2] * poly[101];
    poly[943] = poly[3] * poly[101];
    poly[944] = poly[1] * poly[102];
    poly[945] = poly[2] * poly[102];
    poly[946] = poly[3] * poly[102];
    poly[947] = poly[4] * poly[102];
    poly[948] = poly[1] * poly[103];
    poly[949] = poly[2] * poly[103];
}

fn f_polynomials19(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[950] = poly[3] * poly[103];
    poly[951] = poly[4] * poly[103];
    poly[952] = poly[5] * poly[103];
    poly[953] = poly[1] * poly[104];
    poly[954] = poly[2] * poly[104];
    poly[955] = poly[3] * poly[104];
    poly[956] = poly[4] * poly[104];
    poly[957] = poly[5] * poly[104];
    poly[958] = poly[6] * poly[104];
    poly[959] = poly[54] * poly[12];
    poly[960] = poly[1] * poly[105];
    poly[961] = poly[2] * poly[105];
    poly[962] = poly[3] * poly[105];
    poly[963] = poly[4] * poly[105];
    poly[964] = poly[5] * poly[105];
    poly[965] = poly[6] * poly[105];
    poly[966] = poly[12] * poly[61];
    poly[967] = poly[62] * poly[12];
    poly[968] = poly[12] * poly[63];
    poly[969] = poly[1] * poly[106];
    poly[970] = poly[2] * poly[106];
    poly[971] = poly[3] * poly[106];
    poly[972] = poly[4] * poly[106];
    poly[973] = poly[5] * poly[106];
    poly[974] = poly[6] * poly[106];
    poly[975] = poly[12] * poly[70];
    poly[976] = poly[12] * poly[71];
    poly[977] = poly[72] * poly[12];
    poly[978] = poly[12] * poly[73];
    poly[979] = poly[12] * poly[74];
    poly[980] = poly[1] * poly[107];
    poly[981] = poly[2] * poly[107];
    poly[982] = poly[3] * poly[107];
    poly[983] = poly[4] * poly[107];
    poly[984] = poly[5] * poly[107];
    poly[985] = poly[6] * poly[107];
    poly[986] = poly[12] * poly[81];
    poly[987] = poly[12] * poly[82];
    poly[988] = poly[12] * poly[83];
    poly[989] = poly[84] * poly[12];
    poly[990] = poly[12] * poly[85];
    poly[991] = poly[12] * poly[86];
    poly[992] = poly[12] * poly[87];
    poly[993] = poly[1] * poly[108];
    poly[994] = poly[2] * poly[108];
    poly[995] = poly[3] * poly[108];
    poly[996] = poly[4] * poly[108];
    poly[997] = poly[5] * poly[108];
    poly[998] = poly[6] * poly[108];
    poly[999] = poly[11] * poly[104];
}

fn f_polynomials20(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1000] = poly[11] * poly[105];
    poly[1001] = poly[11] * poly[106];
    poly[1002] = poly[11] * poly[107];
    poly[1003] = poly[1] * poly[109];
    poly[1004] = poly[2] * poly[109];
    poly[1005] = poly[3] * poly[109];
    poly[1006] = poly[4] * poly[109];
    poly[1007] = poly[5] * poly[109];
    poly[1008] = poly[6] * poly[109];
    poly[1009] = poly[109] * poly[7];
    poly[1010] = poly[109] * poly[8];
    poly[1011] = poly[109] * poly[9];
    poly[1012] = poly[109] * poly[10];
    poly[1013] = poly[11] * poly[109];
    poly[1014] = poly[1] * poly[111];
    poly[1015] = poly[1] * poly[112];
    poly[1016] = poly[2] * poly[112];
    poly[1017] = poly[1] * poly[113];
    poly[1018] = poly[2] * poly[113];
    poly[1019] = poly[3] * poly[113];
    poly[1020] = poly[1] * poly[114];
    poly[1021] = poly[2] * poly[114];
    poly[1022] = poly[3] * poly[114];
    poly[1023] = poly[4] * poly[114];
    poly[1024] = poly[1] * poly[115];
    poly[1025] = poly[2] * poly[115];
    poly[1026] = poly[3] * poly[115];
    poly[1027] = poly[4] * poly[115];
    poly[1028] = poly[5] * poly[115];
    poly[1029] = poly[1] * poly[116];
    poly[1030] = poly[2] * poly[116];
    poly[1031] = poly[3] * poly[116];
    poly[1032] = poly[4] * poly[116];
    poly[1033] = poly[5] * poly[116];
    poly[1034] = poly[6] * poly[116];
    poly[1035] = poly[54] * poly[13];
    poly[1036] = poly[1] * poly[117];
    poly[1037] = poly[2] * poly[117];
    poly[1038] = poly[3] * poly[117];
    poly[1039] = poly[4] * poly[117];
    poly[1040] = poly[5] * poly[117];
    poly[1041] = poly[6] * poly[117];
    poly[1042] = poly[13] * poly[61];
    poly[1043] = poly[62] * poly[13];
    poly[1044] = poly[13] * poly[63];
    poly[1045] = poly[1] * poly[118];
    poly[1046] = poly[2] * poly[118];
    poly[1047] = poly[3] * poly[118];
    poly[1048] = poly[4] * poly[118];
    poly[1049] = poly[5] * poly[118];
}

fn f_polynomials21(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1050] = poly[6] * poly[118];
    poly[1051] = poly[13] * poly[70];
    poly[1052] = poly[13] * poly[71];
    poly[1053] = poly[72] * poly[13];
    poly[1054] = poly[13] * poly[73];
    poly[1055] = poly[13] * poly[74];
    poly[1056] = poly[1] * poly[119];
    poly[1057] = poly[2] * poly[119];
    poly[1058] = poly[3] * poly[119];
    poly[1059] = poly[4] * poly[119];
    poly[1060] = poly[5] * poly[119];
    poly[1061] = poly[6] * poly[119];
    poly[1062] = poly[13] * poly[81];
    poly[1063] = poly[13] * poly[82];
    poly[1064] = poly[13] * poly[83];
    poly[1065] = poly[84] * poly[13];
    poly[1066] = poly[13] * poly[85];
    poly[1067] = poly[13] * poly[86];
    poly[1068] = poly[13] * poly[87];
    poly[1069] = poly[1] * poly[120];
    poly[1070] = poly[2] * poly[120];
    poly[1071] = poly[3] * poly[120];
    poly[1072] = poly[4] * poly[120];
    poly[1073] = poly[5] * poly[120];
    poly[1074] = poly[6] * poly[120];
    poly[1075] = poly[11] * poly[116];
    poly[1076] = poly[11] * poly[117];
    poly[1077] = poly[11] * poly[118];
    poly[1078] = poly[11] * poly[119];
    poly[1079] = poly[1] * poly[121];
    poly[1080] = poly[2] * poly[121];
    poly[1081] = poly[3] * poly[121];
    poly[1082] = poly[4] * poly[121];
    poly[1083] = poly[5] * poly[121];
    poly[1084] = poly[6] * poly[121];
    poly[1085] = poly[7] * poly[121];
    poly[1086] = poly[8] * poly[121];
    poly[1087] = poly[9] * poly[121];
    poly[1088] = poly[10] * poly[121];
    poly[1089] = poly[11] * poly[121];
    poly[1090] = poly[1] * poly[122];
    poly[1091] = poly[2] * poly[122];
    poly[1092] = poly[3] * poly[122];
    poly[1093] = poly[4] * poly[122];
    poly[1094] = poly[5] * poly[122];
    poly[1095] = poly[6] * poly[122];
    poly[1096] = poly[122] * poly[7];
    poly[1097] = poly[122] * poly[8];
    poly[1098] = poly[122] * poly[9];
    poly[1099] = poly[122] * poly[10];
}

fn f_polynomials22(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1100] = poly[11] * poly[122];
    poly[1101] = poly[1] * poly[123];
    poly[1102] = poly[2] * poly[123];
    poly[1103] = poly[3] * poly[123];
    poly[1104] = poly[4] * poly[123];
    poly[1105] = poly[5] * poly[123];
    poly[1106] = poly[6] * poly[123];
    poly[1107] = poly[7] * poly[123];
    poly[1108] = poly[8] * poly[123];
    poly[1109] = poly[9] * poly[123];
    poly[1110] = poly[10] * poly[123];
    poly[1111] = poly[11] * poly[123];
    poly[1112] = poly[109] * poly[13];
    poly[1113] = poly[122] * poly[12];
    poly[1114] = poly[1] * poly[125];
    poly[1115] = poly[1] * poly[126];
    poly[1116] = poly[2] * poly[126];
    poly[1117] = poly[1] * poly[127];
    poly[1118] = poly[2] * poly[127];
    poly[1119] = poly[3] * poly[127];
    poly[1120] = poly[1] * poly[128];
    poly[1121] = poly[2] * poly[128];
    poly[1122] = poly[3] * poly[128];
    poly[1123] = poly[4] * poly[128];
    poly[1124] = poly[1] * poly[129];
    poly[1125] = poly[2] * poly[129];
    poly[1126] = poly[3] * poly[129];
    poly[1127] = poly[4] * poly[129];
    poly[1128] = poly[5] * poly[129];
    poly[1129] = poly[1] * poly[130];
    poly[1130] = poly[2] * poly[130];
    poly[1131] = poly[3] * poly[130];
    poly[1132] = poly[4] * poly[130];
    poly[1133] = poly[5] * poly[130];
    poly[1134] = poly[6] * poly[130];
    poly[1135] = poly[54] * poly[14];
    poly[1136] = poly[1] * poly[131];
    poly[1137] = poly[2] * poly[131];
    poly[1138] = poly[3] * poly[131];
    poly[1139] = poly[4] * poly[131];
    poly[1140] = poly[5] * poly[131];
    poly[1141] = poly[6] * poly[131];
    poly[1142] = poly[14] * poly[61];
    poly[1143] = poly[62] * poly[14];
    poly[1144] = poly[14] * poly[63];
    poly[1145] = poly[1] * poly[132];
    poly[1146] = poly[2] * poly[132];
    poly[1147] = poly[3] * poly[132];
    poly[1148] = poly[4] * poly[132];
    poly[1149] = poly[5] * poly[132];
}

fn f_polynomials23(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1150] = poly[6] * poly[132];
    poly[1151] = poly[14] * poly[70];
    poly[1152] = poly[14] * poly[71];
    poly[1153] = poly[72] * poly[14];
    poly[1154] = poly[14] * poly[73];
    poly[1155] = poly[14] * poly[74];
    poly[1156] = poly[1] * poly[133];
    poly[1157] = poly[2] * poly[133];
    poly[1158] = poly[3] * poly[133];
    poly[1159] = poly[4] * poly[133];
    poly[1160] = poly[5] * poly[133];
    poly[1161] = poly[6] * poly[133];
    poly[1162] = poly[14] * poly[81];
    poly[1163] = poly[14] * poly[82];
    poly[1164] = poly[14] * poly[83];
    poly[1165] = poly[84] * poly[14];
    poly[1166] = poly[14] * poly[85];
    poly[1167] = poly[14] * poly[86];
    poly[1168] = poly[14] * poly[87];
    poly[1169] = poly[1] * poly[134];
    poly[1170] = poly[2] * poly[134];
    poly[1171] = poly[3] * poly[134];
    poly[1172] = poly[4] * poly[134];
    poly[1173] = poly[5] * poly[134];
    poly[1174] = poly[6] * poly[134];
    poly[1175] = poly[11] * poly[130];
    poly[1176] = poly[11] * poly[131];
    poly[1177] = poly[11] * poly[132];
    poly[1178] = poly[11] * poly[133];
    poly[1179] = poly[1] * poly[135];
    poly[1180] = poly[2] * poly[135];
    poly[1181] = poly[3] * poly[135];
    poly[1182] = poly[4] * poly[135];
    poly[1183] = poly[5] * poly[135];
    poly[1184] = poly[6] * poly[135];
    poly[1185] = poly[7] * poly[135];
    poly[1186] = poly[8] * poly[135];
    poly[1187] = poly[9] * poly[135];
    poly[1188] = poly[10] * poly[135];
    poly[1189] = poly[11] * poly[135];
    poly[1190] = poly[1] * poly[136];
    poly[1191] = poly[2] * poly[136];
    poly[1192] = poly[3] * poly[136];
    poly[1193] = poly[4] * poly[136];
    poly[1194] = poly[5] * poly[136];
    poly[1195] = poly[6] * poly[136];
    poly[1196] = poly[7] * poly[136];
    poly[1197] = poly[8] * poly[136];
    poly[1198] = poly[9] * poly[136];
    poly[1199] = poly[10] * poly[136];
}

fn f_polynomials24(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1200] = poly[11] * poly[136];
    poly[1201] = mono[463] + mono[464];
    poly[1202] = poly[1] * poly[137];
    poly[1203] = poly[2] * poly[137];
    poly[1204] = poly[3] * poly[137];
    poly[1205] = poly[4] * poly[137];
    poly[1206] = poly[5] * poly[137];
    poly[1207] = poly[6] * poly[137];
    poly[1208] = poly[137] * poly[7];
    poly[1209] = poly[137] * poly[8];
    poly[1210] = poly[137] * poly[9];
    poly[1211] = poly[137] * poly[10];
    poly[1212] = poly[11] * poly[137];
    poly[1213] = poly[1] * poly[138];
    poly[1214] = poly[2] * poly[138];
    poly[1215] = poly[3] * poly[138];
    poly[1216] = poly[4] * poly[138];
    poly[1217] = poly[5] * poly[138];
    poly[1218] = poly[6] * poly[138];
    poly[1219] = poly[7] * poly[138];
    poly[1220] = poly[8] * poly[138];
    poly[1221] = poly[9] * poly[138];
    poly[1222] = poly[10] * poly[138];
    poly[1223] = poly[11] * poly[138];
    poly[1224] = poly[109] * poly[14];
    poly[1225] = poly[12] * poly[136] - poly[1201];
    poly[1226] = poly[137] * poly[12];
    poly[1227] = poly[1] * poly[139];
    poly[1228] = poly[2] * poly[139];
    poly[1229] = poly[3] * poly[139];
    poly[1230] = poly[4] * poly[139];
    poly[1231] = poly[5] * poly[139];
    poly[1232] = poly[6] * poly[139];
    poly[1233] = poly[7] * poly[139];
    poly[1234] = poly[8] * poly[139];
    poly[1235] = poly[9] * poly[139];
    poly[1236] = poly[10] * poly[139];
    poly[1237] = poly[11] * poly[139];
    poly[1238] = poly[13] * poly[135] - poly[1201];
    poly[1239] = poly[122] * poly[14];
    poly[1240] = poly[137] * poly[13];
    poly[1241] = poly[12] * poly[139] - poly[1238];
    poly[1242] = poly[1] * poly[141];
    poly[1243] = poly[1] * poly[142];
    poly[1244] = poly[2] * poly[142];
    poly[1245] = poly[1] * poly[143];
    poly[1246] = poly[2] * poly[143];
    poly[1247] = poly[3] * poly[143];
    poly[1248] = poly[1] * poly[144];
    poly[1249] = poly[2] * poly[144];
}

fn f_polynomials25(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1250] = poly[3] * poly[144];
    poly[1251] = poly[4] * poly[144];
    poly[1252] = poly[1] * poly[145];
    poly[1253] = poly[2] * poly[145];
    poly[1254] = poly[3] * poly[145];
    poly[1255] = poly[4] * poly[145];
    poly[1256] = poly[5] * poly[145];
    poly[1257] = poly[1] * poly[146];
    poly[1258] = poly[2] * poly[146];
    poly[1259] = poly[3] * poly[146];
    poly[1260] = poly[4] * poly[146];
    poly[1261] = poly[5] * poly[146];
    poly[1262] = poly[6] * poly[146];
    poly[1263] = poly[54] * poly[15];
    poly[1264] = poly[1] * poly[147];
    poly[1265] = poly[2] * poly[147];
    poly[1266] = poly[3] * poly[147];
    poly[1267] = poly[4] * poly[147];
    poly[1268] = poly[5] * poly[147];
    poly[1269] = poly[6] * poly[147];
    poly[1270] = poly[15] * poly[61];
    poly[1271] = poly[62] * poly[15];
    poly[1272] = poly[15] * poly[63];
    poly[1273] = poly[1] * poly[148];
    poly[1274] = poly[2] * poly[148];
    poly[1275] = poly[3] * poly[148];
    poly[1276] = poly[4] * poly[148];
    poly[1277] = poly[5] * poly[148];
    poly[1278] = poly[6] * poly[148];
    poly[1279] = poly[15] * poly[70];
    poly[1280] = poly[15] * poly[71];
    poly[1281] = poly[72] * poly[15];
    poly[1282] = poly[15] * poly[73];
    poly[1283] = poly[15] * poly[74];
    poly[1284] = poly[1] * poly[149];
    poly[1285] = poly[2] * poly[149];
    poly[1286] = poly[3] * poly[149];
    poly[1287] = poly[4] * poly[149];
    poly[1288] = poly[5] * poly[149];
    poly[1289] = poly[6] * poly[149];
    poly[1290] = poly[15] * poly[81];
    poly[1291] = poly[15] * poly[82];
    poly[1292] = poly[15] * poly[83];
    poly[1293] = poly[84] * poly[15];
    poly[1294] = poly[15] * poly[85];
    poly[1295] = poly[15] * poly[86];
    poly[1296] = poly[15] * poly[87];
    poly[1297] = poly[1] * poly[150];
    poly[1298] = poly[2] * poly[150];
    poly[1299] = poly[3] * poly[150];
}

fn f_polynomials26(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1300] = poly[4] * poly[150];
    poly[1301] = poly[5] * poly[150];
    poly[1302] = poly[6] * poly[150];
    poly[1303] = poly[11] * poly[146];
    poly[1304] = poly[11] * poly[147];
    poly[1305] = poly[11] * poly[148];
    poly[1306] = poly[11] * poly[149];
    poly[1307] = poly[1] * poly[151];
    poly[1308] = poly[2] * poly[151];
    poly[1309] = poly[3] * poly[151];
    poly[1310] = poly[4] * poly[151];
    poly[1311] = poly[5] * poly[151];
    poly[1312] = poly[6] * poly[151];
    poly[1313] = poly[7] * poly[151];
    poly[1314] = poly[8] * poly[151];
    poly[1315] = poly[9] * poly[151];
    poly[1316] = poly[10] * poly[151];
    poly[1317] = poly[11] * poly[151];
    poly[1318] = poly[1] * poly[152];
    poly[1319] = poly[2] * poly[152];
    poly[1320] = poly[3] * poly[152];
    poly[1321] = poly[4] * poly[152];
    poly[1322] = poly[5] * poly[152];
    poly[1323] = poly[6] * poly[152];
    poly[1324] = poly[7] * poly[152];
    poly[1325] = poly[8] * poly[152];
    poly[1326] = poly[9] * poly[152];
    poly[1327] = poly[10] * poly[152];
    poly[1328] = poly[11] * poly[152];
    poly[1329] = mono[465] + mono[466];
    poly[1330] = poly[1] * poly[153];
    poly[1331] = poly[2] * poly[153];
    poly[1332] = poly[3] * poly[153];
    poly[1333] = poly[4] * poly[153];
    poly[1334] = poly[5] * poly[153];
    poly[1335] = poly[6] * poly[153];
    poly[1336] = poly[7] * poly[153];
    poly[1337] = poly[8] * poly[153];
    poly[1338] = poly[9] * poly[153];
    poly[1339] = poly[10] * poly[153];
    poly[1340] = poly[11] * poly[153];
    poly[1341] = mono[467] + mono[468];
    poly[1342] = mono[469] + mono[470];
    poly[1343] = poly[1] * poly[154];
    poly[1344] = poly[2] * poly[154];
    poly[1345] = poly[3] * poly[154];
    poly[1346] = poly[4] * poly[154];
    poly[1347] = poly[5] * poly[154];
    poly[1348] = poly[6] * poly[154];
    poly[1349] = poly[154] * poly[7];
}

fn f_polynomials27(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1350] = poly[154] * poly[8];
    poly[1351] = poly[154] * poly[9];
    poly[1352] = poly[154] * poly[10];
    poly[1353] = poly[11] * poly[154];
    poly[1354] = poly[1] * poly[155];
    poly[1355] = poly[2] * poly[155];
    poly[1356] = poly[3] * poly[155];
    poly[1357] = poly[4] * poly[155];
    poly[1358] = poly[5] * poly[155];
    poly[1359] = poly[6] * poly[155];
    poly[1360] = poly[7] * poly[155];
    poly[1361] = poly[8] * poly[155];
    poly[1362] = poly[9] * poly[155];
    poly[1363] = poly[10] * poly[155];
    poly[1364] = poly[11] * poly[155];
    poly[1365] = poly[109] * poly[15];
    poly[1366] = poly[12] * poly[152] - poly[1329];
    poly[1367] = poly[12] * poly[153] - poly[1341];
    poly[1368] = poly[154] * poly[12];
    poly[1369] = poly[1] * poly[156];
    poly[1370] = poly[2] * poly[156];
    poly[1371] = poly[3] * poly[156];
    poly[1372] = poly[4] * poly[156];
    poly[1373] = poly[5] * poly[156];
    poly[1374] = poly[6] * poly[156];
    poly[1375] = poly[7] * poly[156];
    poly[1376] = poly[8] * poly[156];
    poly[1377] = poly[9] * poly[156];
    poly[1378] = poly[10] * poly[156];
    poly[1379] = poly[11] * poly[156];
    poly[1380] = poly[13] * poly[151] - poly[1329];
    poly[1381] = poly[122] * poly[15];
    poly[1382] = poly[13] * poly[153] - poly[1342];
    poly[1383] = poly[154] * poly[13];
    poly[1384] = poly[12] * poly[156] - poly[1380];
    poly[1385] = poly[1] * poly[157];
    poly[1386] = poly[2] * poly[157];
    poly[1387] = poly[3] * poly[157];
    poly[1388] = poly[4] * poly[157];
    poly[1389] = poly[5] * poly[157];
    poly[1390] = poly[6] * poly[157];
    poly[1391] = poly[7] * poly[157];
    poly[1392] = poly[8] * poly[157];
    poly[1393] = poly[9] * poly[157];
    poly[1394] = poly[10] * poly[157];
    poly[1395] = poly[11] * poly[157];
    poly[1396] = poly[14] * poly[151] - poly[1341];
    poly[1397] = poly[14] * poly[152] - poly[1342];
    poly[1398] = poly[137] * poly[15];
    poly[1399] = poly[154] * poly[14];
}

fn f_polynomials28(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1400] = poly[12] * poly[157] - poly[1396];
    poly[1401] = poly[13] * poly[157] - poly[1397];
    poly[1402] = poly[1] * poly[159];
    poly[1403] = poly[1] * poly[160];
    poly[1404] = poly[2] * poly[160];
    poly[1405] = poly[1] * poly[161];
    poly[1406] = poly[2] * poly[161];
    poly[1407] = poly[3] * poly[161];
    poly[1408] = poly[1] * poly[162];
    poly[1409] = poly[2] * poly[162];
    poly[1410] = poly[3] * poly[162];
    poly[1411] = poly[4] * poly[162];
    poly[1412] = poly[1] * poly[163];
    poly[1413] = poly[2] * poly[163];
    poly[1414] = poly[3] * poly[163];
    poly[1415] = poly[4] * poly[163];
    poly[1416] = poly[5] * poly[163];
    poly[1417] = poly[1] * poly[164];
    poly[1418] = poly[2] * poly[164];
    poly[1419] = poly[3] * poly[164];
    poly[1420] = poly[4] * poly[164];
    poly[1421] = poly[5] * poly[164];
    poly[1422] = poly[6] * poly[164];
    poly[1423] = poly[1] * poly[165];
    poly[1424] = poly[2] * poly[165];
    poly[1425] = poly[3] * poly[165];
    poly[1426] = poly[4] * poly[165];
    poly[1427] = poly[5] * poly[165];
    poly[1428] = poly[6] * poly[165];
    poly[1429] = mono[471] + mono[472] + mono[473] + mono[474];
    poly[1430] = poly[1] * poly[166];
    poly[1431] = poly[2] * poly[166];
    poly[1432] = poly[3] * poly[166];
    poly[1433] = poly[4] * poly[166];
    poly[1434] = poly[5] * poly[166];
    poly[1435] = poly[6] * poly[166];
    poly[1436] = mono[475] + mono[476] + mono[477] + mono[478];
    poly[1437] = mono[479] + mono[480] + mono[481] + mono[482];
    poly[1438] = poly[1] * poly[167];
    poly[1439] = poly[2] * poly[167];
    poly[1440] = poly[3] * poly[167];
    poly[1441] = poly[4] * poly[167];
    poly[1442] = poly[5] * poly[167];
    poly[1443] = poly[6] * poly[167];
    poly[1444] = mono[483] + mono[484] + mono[485] + mono[486];
    poly[1445] = mono[487] + mono[488] + mono[489] + mono[490];
    poly[1446] = mono[491] + mono[492] + mono[493] + mono[494];
    poly[1447] = poly[1] * poly[168];
    poly[1448] = poly[2] * poly[168];
    poly[1449] = poly[3] * poly[168];
}

fn f_polynomials29(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1450] = poly[4] * poly[168];
    poly[1451] = poly[5] * poly[168];
    poly[1452] = poly[6] * poly[168];
    poly[1453] = poly[54] * poly[16];
    poly[1454] = poly[7] * poly[165] - poly[1429];
    poly[1455] = poly[7] * poly[166] - poly[1436];
    poly[1456] = poly[7] * poly[167] - poly[1444];
    poly[1457] = poly[1] * poly[169];
    poly[1458] = poly[2] * poly[169];
    poly[1459] = poly[3] * poly[169];
    poly[1460] = poly[4] * poly[169];
    poly[1461] = poly[5] * poly[169];
    poly[1462] = poly[6] * poly[169];
    poly[1463] = poly[8] * poly[164] - poly[1429];
    poly[1464] = poly[62] * poly[16];
    poly[1465] = poly[8] * poly[166] - poly[1437];
    poly[1466] = poly[8] * poly[167] - poly[1445];
    poly[1467] = poly[7] * poly[169] - poly[1463];
    poly[1468] = poly[1] * poly[170];
    poly[1469] = poly[2] * poly[170];
    poly[1470] = poly[3] * poly[170];
    poly[1471] = poly[4] * poly[170];
    poly[1472] = poly[5] * poly[170];
    poly[1473] = poly[6] * poly[170];
    poly[1474] = poly[9] * poly[164] - poly[1436];
    poly[1475] = poly[9] * poly[165] - poly[1437];
    poly[1476] = poly[72] * poly[16];
    poly[1477] = poly[9] * poly[167] - poly[1446];
    poly[1478] = poly[7] * poly[170] - poly[1474];
    poly[1479] = poly[8] * poly[170] - poly[1475];
    poly[1480] = poly[1] * poly[171];
    poly[1481] = poly[2] * poly[171];
    poly[1482] = poly[3] * poly[171];
    poly[1483] = poly[4] * poly[171];
    poly[1484] = poly[5] * poly[171];
    poly[1485] = poly[6] * poly[171];
    poly[1486] = poly[10] * poly[164] - poly[1444];
    poly[1487] = poly[10] * poly[165] - poly[1445];
    poly[1488] = poly[10] * poly[166] - poly[1446];
    poly[1489] = poly[84] * poly[16];
    poly[1490] = poly[7] * poly[171] - poly[1486];
    poly[1491] = poly[8] * poly[171] - poly[1487];
    poly[1492] = poly[9] * poly[171] - poly[1488];
    poly[1493] = poly[1] * poly[172];
    poly[1494] = poly[2] * poly[172];
    poly[1495] = poly[3] * poly[172];
    poly[1496] = poly[4] * poly[172];
    poly[1497] = poly[5] * poly[172];
    poly[1498] = poly[6] * poly[172];
    poly[1499] = poly[11] * poly[164];
}

fn f_polynomials30(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1500] = poly[11] * poly[165];
    poly[1501] = poly[11] * poly[166];
    poly[1502] = poly[11] * poly[167];
    poly[1503] = poly[11] * poly[168];
    poly[1504] = poly[11] * poly[169];
    poly[1505] = poly[11] * poly[170];
    poly[1506] = poly[11] * poly[171];
    poly[1507] = poly[1] * poly[173];
    poly[1508] = poly[2] * poly[173];
    poly[1509] = poly[3] * poly[173];
    poly[1510] = poly[4] * poly[173];
    poly[1511] = poly[5] * poly[173];
    poly[1512] = poly[6] * poly[173];
    poly[1513] = mono[495] + mono[496] + mono[497] + mono[498];
    poly[1514] = mono[499] + mono[500] + mono[501] + mono[502];
    poly[1515] = mono[503] + mono[504] + mono[505] + mono[506];
    poly[1516] = mono[507] + mono[508] + mono[509] + mono[510];
    poly[1517] = poly[7] * poly[173] - poly[1513];
    poly[1518] = poly[8] * poly[173] - poly[1514];
    poly[1519] = poly[9] * poly[173] - poly[1515];
    poly[1520] = poly[10] * poly[173] - poly[1516];
    poly[1521] = poly[11] * poly[173];
    poly[1522] = poly[1] * poly[174];
    poly[1523] = poly[2] * poly[174];
    poly[1524] = poly[3] * poly[174];
    poly[1525] = poly[4] * poly[174];
    poly[1526] = poly[5] * poly[174];
    poly[1527] = poly[6] * poly[174];
    poly[1528] = mono[511] + mono[512] + mono[513] + mono[514];
    poly[1529] = mono[515] + mono[516] + mono[517] + mono[518];
    poly[1530] = mono[519] + mono[520] + mono[521] + mono[522];
    poly[1531] = mono[523] + mono[524] + mono[525] + mono[526];
    poly[1532] = poly[7] * poly[174] - poly[1528];
    poly[1533] = poly[8] * poly[174] - poly[1529];
    poly[1534] = poly[9] * poly[174] - poly[1530];
    poly[1535] = poly[10] * poly[174] - poly[1531];
    poly[1536] = poly[11] * poly[174];
    poly[1537] = mono[527] + mono[528] + mono[529] + mono[530];
    poly[1538] = poly[1] * poly[175];
    poly[1539] = poly[2] * poly[175];
    poly[1540] = poly[3] * poly[175];
    poly[1541] = poly[4] * poly[175];
    poly[1542] = poly[5] * poly[175];
    poly[1543] = poly[6] * poly[175];
    poly[1544] = mono[531] + mono[532] + mono[533] + mono[534];
    poly[1545] = mono[535] + mono[536] + mono[537] + mono[538];
    poly[1546] = mono[539] + mono[540] + mono[541] + mono[542];
    poly[1547] = mono[543] + mono[544] + mono[545] + mono[546];
    poly[1548] = poly[7] * poly[175] - poly[1544];
    poly[1549] = poly[8] * poly[175] - poly[1545];
}

fn f_polynomials31(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1550] = poly[9] * poly[175] - poly[1546];
    poly[1551] = poly[10] * poly[175] - poly[1547];
    poly[1552] = poly[11] * poly[175];
    poly[1553] = mono[547] + mono[548] + mono[549] + mono[550];
    poly[1554] = mono[551] + mono[552] + mono[553] + mono[554];
    poly[1555] = poly[1] * poly[176];
    poly[1556] = poly[2] * poly[176];
    poly[1557] = poly[3] * poly[176];
    poly[1558] = poly[4] * poly[176];
    poly[1559] = poly[5] * poly[176];
    poly[1560] = poly[6] * poly[176];
    poly[1561] = mono[555] + mono[556] + mono[557] + mono[558];
    poly[1562] = mono[559] + mono[560] + mono[561] + mono[562];
    poly[1563] = mono[563] + mono[564] + mono[565] + mono[566];
    poly[1564] = mono[567] + mono[568] + mono[569] + mono[570];
    poly[1565] = poly[7] * poly[176] - poly[1561];
    poly[1566] = poly[8] * poly[176] - poly[1562];
    poly[1567] = poly[9] * poly[176] - poly[1563];
    poly[1568] = poly[10] * poly[176] - poly[1564];
    poly[1569] = poly[11] * poly[176];
    poly[1570] = mono[571] + mono[572] + mono[573] + mono[574];
    poly[1571] = mono[575] + mono[576] + mono[577] + mono[578];
    poly[1572] = mono[579] + mono[580] + mono[581] + mono[582];
    poly[1573] = poly[1] * poly[177];
    poly[1574] = poly[2] * poly[177];
    poly[1575] = poly[3] * poly[177];
    poly[1576] = poly[4] * poly[177];
    poly[1577] = poly[5] * poly[177];
    poly[1578] = poly[6] * poly[177];
    poly[1579] = poly[7] * poly[177];
    poly[1580] = poly[8] * poly[177];
    poly[1581] = poly[9] * poly[177];
    poly[1582] = poly[10] * poly[177];
    poly[1583] = poly[11] * poly[177];
    poly[1584] = poly[1] * poly[178];
    poly[1585] = poly[2] * poly[178];
    poly[1586] = poly[3] * poly[178];
    poly[1587] = poly[4] * poly[178];
    poly[1588] = poly[5] * poly[178];
    poly[1589] = poly[6] * poly[178];
    poly[1590] = mono[583] + mono[584];
    poly[1591] = mono[585] + mono[586];
    poly[1592] = mono[587] + mono[588];
    poly[1593] = mono[589] + mono[590];
    poly[1594] = poly[7] * poly[178] - poly[1590];
    poly[1595] = poly[8] * poly[178] - poly[1591];
    poly[1596] = poly[9] * poly[178] - poly[1592];
    poly[1597] = poly[10] * poly[178] - poly[1593];
    poly[1598] = poly[11] * poly[178];
    poly[1599] = poly[1] * poly[179];
}

fn f_polynomials32(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1600] = poly[2] * poly[179];
    poly[1601] = poly[3] * poly[179];
    poly[1602] = poly[4] * poly[179];
    poly[1603] = poly[5] * poly[179];
    poly[1604] = poly[6] * poly[179];
    poly[1605] = poly[12] * poly[164] - poly[1513];
    poly[1606] = poly[12] * poly[165] - poly[1514];
    poly[1607] = poly[12] * poly[166] - poly[1515];
    poly[1608] = poly[12] * poly[167] - poly[1516];
    poly[1609] = poly[7] * poly[179] - poly[1605];
    poly[1610] = poly[8] * poly[179] - poly[1606];
    poly[1611] = poly[9] * poly[179] - poly[1607];
    poly[1612] = poly[10] * poly[179] - poly[1608];
    poly[1613] = poly[11] * poly[179];
    poly[1614] = poly[109] * poly[16];
    poly[1615] = poly[12] * poly[174] - poly[1537];
    poly[1616] = poly[12] * poly[175] - poly[1553];
    poly[1617] = poly[12] * poly[176] - poly[1570];
    poly[1618] = poly[12] * poly[177];
    poly[1619] = poly[12] * poly[178];
    poly[1620] = poly[1] * poly[180];
    poly[1621] = poly[2] * poly[180];
    poly[1622] = poly[3] * poly[180];
    poly[1623] = poly[4] * poly[180];
    poly[1624] = poly[5] * poly[180];
    poly[1625] = poly[6] * poly[180];
    poly[1626] = poly[13] * poly[164] - poly[1528];
    poly[1627] = poly[13] * poly[165] - poly[1529];
    poly[1628] = poly[13] * poly[166] - poly[1530];
    poly[1629] = poly[13] * poly[167] - poly[1531];
    poly[1630] = poly[7] * poly[180] - poly[1626];
    poly[1631] = poly[8] * poly[180] - poly[1627];
    poly[1632] = poly[9] * poly[180] - poly[1628];
    poly[1633] = poly[10] * poly[180] - poly[1629];
    poly[1634] = poly[11] * poly[180];
    poly[1635] = poly[13] * poly[173] - poly[1537];
    poly[1636] = poly[122] * poly[16];
    poly[1637] = poly[13] * poly[175] - poly[1554];
    poly[1638] = poly[13] * poly[176] - poly[1571];
    poly[1639] = poly[13] * poly[177];
    poly[1640] = poly[13] * poly[178];
    poly[1641] = poly[12] * poly[180] - poly[1635];
    poly[1642] = poly[1] * poly[181];
    poly[1643] = poly[2] * poly[181];
    poly[1644] = poly[3] * poly[181];
    poly[1645] = poly[4] * poly[181];
    poly[1646] = poly[5] * poly[181];
    poly[1647] = poly[6] * poly[181];
    poly[1648] = poly[14] * poly[164] - poly[1544];
    poly[1649] = poly[14] * poly[165] - poly[1545];
}

fn f_polynomials33(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1650] = poly[14] * poly[166] - poly[1546];
    poly[1651] = poly[14] * poly[167] - poly[1547];
    poly[1652] = poly[7] * poly[181] - poly[1648];
    poly[1653] = poly[8] * poly[181] - poly[1649];
    poly[1654] = poly[9] * poly[181] - poly[1650];
    poly[1655] = poly[10] * poly[181] - poly[1651];
    poly[1656] = poly[11] * poly[181];
    poly[1657] = poly[14] * poly[173] - poly[1553];
    poly[1658] = poly[14] * poly[174] - poly[1554];
    poly[1659] = poly[137] * poly[16];
    poly[1660] = poly[14] * poly[176] - poly[1572];
    poly[1661] = poly[14] * poly[177];
    poly[1662] = poly[14] * poly[178];
    poly[1663] = poly[12] * poly[181] - poly[1657];
    poly[1664] = poly[13] * poly[181] - poly[1658];
    poly[1665] = poly[1] * poly[182];
    poly[1666] = poly[2] * poly[182];
    poly[1667] = poly[3] * poly[182];
    poly[1668] = poly[4] * poly[182];
    poly[1669] = poly[5] * poly[182];
    poly[1670] = poly[6] * poly[182];
    poly[1671] = poly[15] * poly[164] - poly[1561];
    poly[1672] = poly[15] * poly[165] - poly[1562];
    poly[1673] = poly[15] * poly[166] - poly[1563];
    poly[1674] = poly[15] * poly[167] - poly[1564];
    poly[1675] = poly[7] * poly[182] - poly[1671];
    poly[1676] = poly[8] * poly[182] - poly[1672];
    poly[1677] = poly[9] * poly[182] - poly[1673];
    poly[1678] = poly[10] * poly[182] - poly[1674];
    poly[1679] = poly[11] * poly[182];
    poly[1680] = poly[15] * poly[173] - poly[1570];
    poly[1681] = poly[15] * poly[174] - poly[1571];
    poly[1682] = poly[15] * poly[175] - poly[1572];
    poly[1683] = poly[154] * poly[16];
    poly[1684] = poly[15] * poly[177];
    poly[1685] = poly[15] * poly[178];
    poly[1686] = poly[12] * poly[182] - poly[1680];
    poly[1687] = poly[13] * poly[182] - poly[1681];
    poly[1688] = poly[14] * poly[182] - poly[1682];
    poly[1689] = poly[1] * poly[183];
    poly[1690] = poly[2] * poly[183];
    poly[1691] = poly[3] * poly[183];
    poly[1692] = poly[4] * poly[183];
    poly[1693] = poly[5] * poly[183];
    poly[1694] = poly[6] * poly[183];
    poly[1695] = poly[7] * poly[183];
    poly[1696] = poly[8] * poly[183];
    poly[1697] = poly[9] * poly[183];
    poly[1698] = poly[10] * poly[183];
    poly[1699] = poly[11] * poly[183];
}

fn f_polynomials34(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1700] = mono[591] + mono[592];
    poly[1701] = mono[593] + mono[594];
    poly[1702] = mono[595] + mono[596];
    poly[1703] = mono[597] + mono[598];
    poly[1704] = mono[599] + mono[600] + mono[601] + mono[602];
    poly[1705] = poly[12] * poly[183] - poly[1700];
    poly[1706] = poly[13] * poly[183] - poly[1701];
    poly[1707] = poly[14] * poly[183] - poly[1702];
    poly[1708] = poly[15] * poly[183] - poly[1703];
    poly[1709] = poly[1] * poly[185];
    poly[1710] = poly[1] * poly[186];
    poly[1711] = poly[2] * poly[186];
    poly[1712] = poly[1] * poly[187];
    poly[1713] = poly[2] * poly[187];
    poly[1714] = poly[3] * poly[187];
    poly[1715] = poly[1] * poly[188];
    poly[1716] = poly[2] * poly[188];
    poly[1717] = poly[3] * poly[188];
    poly[1718] = poly[4] * poly[188];
    poly[1719] = poly[1] * poly[189];
    poly[1720] = poly[2] * poly[189];
    poly[1721] = poly[3] * poly[189];
    poly[1722] = poly[4] * poly[189];
    poly[1723] = poly[5] * poly[189];
    poly[1724] = poly[1] * poly[190];
    poly[1725] = poly[2] * poly[190];
    poly[1726] = poly[3] * poly[190];
    poly[1727] = poly[4] * poly[190];
    poly[1728] = poly[5] * poly[190];
    poly[1729] = poly[6] * poly[190];
    poly[1730] = poly[17] * poly[54];
    poly[1731] = poly[1] * poly[191];
    poly[1732] = poly[2] * poly[191];
    poly[1733] = poly[3] * poly[191];
    poly[1734] = poly[4] * poly[191];
    poly[1735] = poly[5] * poly[191];
    poly[1736] = poly[6] * poly[191];
    poly[1737] = poly[17] * poly[61];
    poly[1738] = poly[17] * poly[62];
    poly[1739] = poly[17] * poly[63];
    poly[1740] = poly[1] * poly[192];
    poly[1741] = poly[2] * poly[192];
    poly[1742] = poly[3] * poly[192];
    poly[1743] = poly[4] * poly[192];
    poly[1744] = poly[5] * poly[192];
    poly[1745] = poly[6] * poly[192];
    poly[1746] = poly[17] * poly[70];
    poly[1747] = poly[17] * poly[71];
    poly[1748] = poly[17] * poly[72];
    poly[1749] = poly[17] * poly[73];
}

fn f_polynomials35(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1750] = poly[17] * poly[74];
    poly[1751] = poly[1] * poly[193];
    poly[1752] = poly[2] * poly[193];
    poly[1753] = poly[3] * poly[193];
    poly[1754] = poly[4] * poly[193];
    poly[1755] = poly[5] * poly[193];
    poly[1756] = poly[6] * poly[193];
    poly[1757] = poly[17] * poly[81];
    poly[1758] = poly[17] * poly[82];
    poly[1759] = poly[17] * poly[83];
    poly[1760] = poly[17] * poly[84];
    poly[1761] = poly[17] * poly[85];
    poly[1762] = poly[17] * poly[86];
    poly[1763] = poly[17] * poly[87];
    poly[1764] = poly[1] * poly[194];
    poly[1765] = poly[2] * poly[194];
    poly[1766] = poly[3] * poly[194];
    poly[1767] = poly[4] * poly[194];
    poly[1768] = poly[5] * poly[194];
    poly[1769] = poly[6] * poly[194];
    poly[1770] = poly[11] * poly[190];
    poly[1771] = poly[11] * poly[191];
    poly[1772] = poly[11] * poly[192];
    poly[1773] = poly[11] * poly[193];
    poly[1774] = poly[1] * poly[195];
    poly[1775] = poly[2] * poly[195];
    poly[1776] = poly[3] * poly[195];
    poly[1777] = poly[4] * poly[195];
    poly[1778] = poly[5] * poly[195];
    poly[1779] = poly[6] * poly[195];
    poly[1780] = poly[17] * poly[104];
    poly[1781] = poly[17] * poly[105];
    poly[1782] = poly[17] * poly[106];
    poly[1783] = poly[17] * poly[107];
    poly[1784] = poly[11] * poly[195];
    poly[1785] = poly[17] * poly[109];
    poly[1786] = poly[1] * poly[196];
    poly[1787] = poly[2] * poly[196];
    poly[1788] = poly[3] * poly[196];
    poly[1789] = poly[4] * poly[196];
    poly[1790] = poly[5] * poly[196];
    poly[1791] = poly[6] * poly[196];
    poly[1792] = poly[17] * poly[116];
    poly[1793] = poly[17] * poly[117];
    poly[1794] = poly[17] * poly[118];
    poly[1795] = poly[17] * poly[119];
    poly[1796] = poly[11] * poly[196];
    poly[1797] = poly[17] * poly[121];
    poly[1798] = poly[17] * poly[122];
    poly[1799] = poly[17] * poly[123];
}

fn f_polynomials36(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1800] = poly[1] * poly[197];
    poly[1801] = poly[2] * poly[197];
    poly[1802] = poly[3] * poly[197];
    poly[1803] = poly[4] * poly[197];
    poly[1804] = poly[5] * poly[197];
    poly[1805] = poly[6] * poly[197];
    poly[1806] = poly[17] * poly[130];
    poly[1807] = poly[17] * poly[131];
    poly[1808] = poly[17] * poly[132];
    poly[1809] = poly[17] * poly[133];
    poly[1810] = poly[11] * poly[197];
    poly[1811] = poly[17] * poly[135];
    poly[1812] = poly[17] * poly[136];
    poly[1813] = poly[17] * poly[137];
    poly[1814] = poly[17] * poly[138];
    poly[1815] = poly[17] * poly[139];
    poly[1816] = poly[1] * poly[198];
    poly[1817] = poly[2] * poly[198];
    poly[1818] = poly[3] * poly[198];
    poly[1819] = poly[4] * poly[198];
    poly[1820] = poly[5] * poly[198];
    poly[1821] = poly[6] * poly[198];
    poly[1822] = poly[17] * poly[146];
    poly[1823] = poly[17] * poly[147];
    poly[1824] = poly[17] * poly[148];
    poly[1825] = poly[17] * poly[149];
    poly[1826] = poly[11] * poly[198];
    poly[1827] = poly[17] * poly[151];
    poly[1828] = poly[17] * poly[152];
    poly[1829] = poly[17] * poly[153];
    poly[1830] = poly[17] * poly[154];
    poly[1831] = poly[17] * poly[155];
    poly[1832] = poly[17] * poly[156];
    poly[1833] = poly[17] * poly[157];
    poly[1834] = poly[1] * poly[199];
    poly[1835] = poly[2] * poly[199];
    poly[1836] = poly[3] * poly[199];
    poly[1837] = poly[4] * poly[199];
    poly[1838] = poly[5] * poly[199];
    poly[1839] = poly[6] * poly[199];
    poly[1840] = poly[17] * poly[164];
    poly[1841] = poly[17] * poly[165];
    poly[1842] = poly[17] * poly[166];
    poly[1843] = poly[17] * poly[167];
    poly[1844] = poly[17] * poly[168];
    poly[1845] = poly[17] * poly[169];
    poly[1846] = poly[17] * poly[170];
    poly[1847] = poly[17] * poly[171];
    poly[1848] = poly[11] * poly[199];
    poly[1849] = poly[17] * poly[173];
}

fn f_polynomials37(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1850] = poly[17] * poly[174];
    poly[1851] = poly[17] * poly[175];
    poly[1852] = poly[17] * poly[176];
    poly[1853] = poly[17] * poly[177];
    poly[1854] = poly[17] * poly[178];
    poly[1855] = poly[17] * poly[179];
    poly[1856] = poly[17] * poly[180];
    poly[1857] = poly[17] * poly[181];
    poly[1858] = poly[17] * poly[182];
    poly[1859] = poly[17] * poly[183];
    poly[1860] = poly[1] * poly[201];
    poly[1861] = poly[1] * poly[202];
    poly[1862] = poly[2] * poly[202];
    poly[1863] = poly[1] * poly[203];
    poly[1864] = poly[2] * poly[203];
    poly[1865] = poly[3] * poly[203];
    poly[1866] = poly[1] * poly[204];
    poly[1867] = poly[2] * poly[204];
    poly[1868] = poly[3] * poly[204];
    poly[1869] = poly[4] * poly[204];
    poly[1870] = poly[1] * poly[205];
    poly[1871] = poly[2] * poly[205];
    poly[1872] = poly[3] * poly[205];
    poly[1873] = poly[4] * poly[205];
    poly[1874] = poly[5] * poly[205];
    poly[1875] = poly[1] * poly[206];
    poly[1876] = poly[2] * poly[206];
    poly[1877] = poly[3] * poly[206];
    poly[1878] = poly[4] * poly[206];
    poly[1879] = poly[5] * poly[206];
    poly[1880] = poly[6] * poly[206];
    poly[1881] = poly[54] * poly[18];
    poly[1882] = poly[1] * poly[207];
    poly[1883] = poly[2] * poly[207];
    poly[1884] = poly[3] * poly[207];
    poly[1885] = poly[4] * poly[207];
    poly[1886] = poly[5] * poly[207];
    poly[1887] = poly[6] * poly[207];
    poly[1888] = poly[18] * poly[61];
    poly[1889] = poly[62] * poly[18];
    poly[1890] = poly[18] * poly[63];
    poly[1891] = poly[1] * poly[208];
    poly[1892] = poly[2] * poly[208];
    poly[1893] = poly[3] * poly[208];
    poly[1894] = poly[4] * poly[208];
    poly[1895] = poly[5] * poly[208];
    poly[1896] = poly[6] * poly[208];
    poly[1897] = poly[18] * poly[70];
    poly[1898] = poly[18] * poly[71];
    poly[1899] = poly[72] * poly[18];
}

fn f_polynomials38(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1900] = poly[18] * poly[73];
    poly[1901] = poly[18] * poly[74];
    poly[1902] = poly[1] * poly[209];
    poly[1903] = poly[2] * poly[209];
    poly[1904] = poly[3] * poly[209];
    poly[1905] = poly[4] * poly[209];
    poly[1906] = poly[5] * poly[209];
    poly[1907] = poly[6] * poly[209];
    poly[1908] = poly[18] * poly[81];
    poly[1909] = poly[18] * poly[82];
    poly[1910] = poly[18] * poly[83];
    poly[1911] = poly[84] * poly[18];
    poly[1912] = poly[18] * poly[85];
    poly[1913] = poly[18] * poly[86];
    poly[1914] = poly[18] * poly[87];
    poly[1915] = poly[1] * poly[210];
    poly[1916] = poly[2] * poly[210];
    poly[1917] = poly[3] * poly[210];
    poly[1918] = poly[4] * poly[210];
    poly[1919] = poly[5] * poly[210];
    poly[1920] = poly[6] * poly[210];
    poly[1921] = poly[11] * poly[206];
    poly[1922] = poly[11] * poly[207];
    poly[1923] = poly[11] * poly[208];
    poly[1924] = poly[11] * poly[209];
    poly[1925] = poly[1] * poly[211];
    poly[1926] = poly[2] * poly[211];
    poly[1927] = poly[3] * poly[211];
    poly[1928] = poly[4] * poly[211];
    poly[1929] = poly[5] * poly[211];
    poly[1930] = poly[6] * poly[211];
    poly[1931] = poly[7] * poly[211];
    poly[1932] = poly[8] * poly[211];
    poly[1933] = poly[9] * poly[211];
    poly[1934] = poly[10] * poly[211];
    poly[1935] = poly[11] * poly[211];
    poly[1936] = poly[109] * poly[18];
    poly[1937] = poly[1] * poly[212];
    poly[1938] = poly[2] * poly[212];
    poly[1939] = poly[3] * poly[212];
    poly[1940] = poly[4] * poly[212];
    poly[1941] = poly[5] * poly[212];
    poly[1942] = poly[6] * poly[212];
    poly[1943] = poly[7] * poly[212];
    poly[1944] = poly[8] * poly[212];
    poly[1945] = poly[9] * poly[212];
    poly[1946] = poly[10] * poly[212];
    poly[1947] = poly[11] * poly[212];
    poly[1948] = poly[18] * poly[121];
    poly[1949] = poly[122] * poly[18];
}

fn f_polynomials39(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[1950] = poly[18] * poly[123];
    poly[1951] = poly[1] * poly[213];
    poly[1952] = poly[2] * poly[213];
    poly[1953] = poly[3] * poly[213];
    poly[1954] = poly[4] * poly[213];
    poly[1955] = poly[5] * poly[213];
    poly[1956] = poly[6] * poly[213];
    poly[1957] = poly[7] * poly[213];
    poly[1958] = poly[8] * poly[213];
    poly[1959] = poly[9] * poly[213];
    poly[1960] = poly[10] * poly[213];
    poly[1961] = poly[11] * poly[213];
    poly[1962] = poly[18] * poly[135];
    poly[1963] = poly[18] * poly[136];
    poly[1964] = poly[137] * poly[18];
    poly[1965] = poly[18] * poly[138];
    poly[1966] = poly[18] * poly[139];
    poly[1967] = poly[1] * poly[214];
    poly[1968] = poly[2] * poly[214];
    poly[1969] = poly[3] * poly[214];
    poly[1970] = poly[4] * poly[214];
    poly[1971] = poly[5] * poly[214];
    poly[1972] = poly[6] * poly[214];
    poly[1973] = poly[7] * poly[214];
    poly[1974] = poly[8] * poly[214];
    poly[1975] = poly[9] * poly[214];
    poly[1976] = poly[10] * poly[214];
    poly[1977] = poly[11] * poly[214];
    poly[1978] = poly[18] * poly[151];
    poly[1979] = poly[18] * poly[152];
    poly[1980] = poly[18] * poly[153];
    poly[1981] = poly[154] * poly[18];
    poly[1982] = poly[18] * poly[155];
    poly[1983] = poly[18] * poly[156];
    poly[1984] = poly[18] * poly[157];
    poly[1985] = poly[1] * poly[215];
    poly[1986] = poly[2] * poly[215];
    poly[1987] = poly[3] * poly[215];
    poly[1988] = poly[4] * poly[215];
    poly[1989] = poly[5] * poly[215];
    poly[1990] = poly[6] * poly[215];
    poly[1991] = poly[18] * poly[164];
    poly[1992] = poly[18] * poly[165];
    poly[1993] = poly[18] * poly[166];
    poly[1994] = poly[18] * poly[167];
    poly[1995] = poly[18] * poly[168];
    poly[1996] = poly[18] * poly[169];
    poly[1997] = poly[18] * poly[170];
    poly[1998] = poly[18] * poly[171];
    poly[1999] = poly[11] * poly[215];
}

fn f_polynomials40(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2000] = poly[18] * poly[173];
    poly[2001] = poly[18] * poly[174];
    poly[2002] = poly[18] * poly[175];
    poly[2003] = poly[18] * poly[176];
    poly[2004] = poly[18] * poly[177];
    poly[2005] = poly[18] * poly[178];
    poly[2006] = poly[18] * poly[179];
    poly[2007] = poly[18] * poly[180];
    poly[2008] = poly[18] * poly[181];
    poly[2009] = poly[18] * poly[182];
    poly[2010] = poly[18] * poly[183];
    poly[2011] = poly[1] * poly[216];
    poly[2012] = poly[2] * poly[216];
    poly[2013] = poly[3] * poly[216];
    poly[2014] = poly[4] * poly[216];
    poly[2015] = poly[5] * poly[216];
    poly[2016] = poly[6] * poly[216];
    poly[2017] = poly[17] * poly[206];
    poly[2018] = poly[17] * poly[207];
    poly[2019] = poly[17] * poly[208];
    poly[2020] = poly[17] * poly[209];
    poly[2021] = poly[11] * poly[216];
    poly[2022] = poly[17] * poly[211];
    poly[2023] = poly[17] * poly[212];
    poly[2024] = poly[17] * poly[213];
    poly[2025] = poly[17] * poly[214];
    poly[2026] = poly[17] * poly[215];
    poly[2027] = poly[1] * poly[217];
    poly[2028] = poly[2] * poly[217];
    poly[2029] = poly[3] * poly[217];
    poly[2030] = poly[4] * poly[217];
    poly[2031] = poly[5] * poly[217];
    poly[2032] = poly[6] * poly[217];
    poly[2033] = poly[217] * poly[7];
    poly[2034] = poly[217] * poly[8];
    poly[2035] = poly[217] * poly[9];
    poly[2036] = poly[217] * poly[10];
    poly[2037] = poly[11] * poly[217];
    poly[2038] = poly[217] * poly[12];
    poly[2039] = poly[217] * poly[13];
    poly[2040] = poly[217] * poly[14];
    poly[2041] = poly[217] * poly[15];
    poly[2042] = poly[217] * poly[16];
    poly[2043] = poly[17] * poly[217];
    poly[2044] = poly[1] * poly[219];
    poly[2045] = poly[1] * poly[220];
    poly[2046] = poly[2] * poly[220];
    poly[2047] = poly[1] * poly[221];
    poly[2048] = poly[2] * poly[221];
    poly[2049] = poly[3] * poly[221];
}

fn f_polynomials41(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2050] = poly[1] * poly[222];
    poly[2051] = poly[2] * poly[222];
    poly[2052] = poly[3] * poly[222];
    poly[2053] = poly[4] * poly[222];
    poly[2054] = poly[1] * poly[223];
    poly[2055] = poly[2] * poly[223];
    poly[2056] = poly[3] * poly[223];
    poly[2057] = poly[4] * poly[223];
    poly[2058] = poly[5] * poly[223];
    poly[2059] = poly[1] * poly[224];
    poly[2060] = poly[2] * poly[224];
    poly[2061] = poly[3] * poly[224];
    poly[2062] = poly[4] * poly[224];
    poly[2063] = poly[5] * poly[224];
    poly[2064] = poly[6] * poly[224];
    poly[2065] = poly[54] * poly[19];
    poly[2066] = poly[1] * poly[225];
    poly[2067] = poly[2] * poly[225];
    poly[2068] = poly[3] * poly[225];
    poly[2069] = poly[4] * poly[225];
    poly[2070] = poly[5] * poly[225];
    poly[2071] = poly[6] * poly[225];
    poly[2072] = poly[19] * poly[61];
    poly[2073] = poly[62] * poly[19];
    poly[2074] = poly[19] * poly[63];
    poly[2075] = poly[1] * poly[226];
    poly[2076] = poly[2] * poly[226];
    poly[2077] = poly[3] * poly[226];
    poly[2078] = poly[4] * poly[226];
    poly[2079] = poly[5] * poly[226];
    poly[2080] = poly[6] * poly[226];
    poly[2081] = poly[19] * poly[70];
    poly[2082] = poly[19] * poly[71];
    poly[2083] = poly[72] * poly[19];
    poly[2084] = poly[19] * poly[73];
    poly[2085] = poly[19] * poly[74];
    poly[2086] = poly[1] * poly[227];
    poly[2087] = poly[2] * poly[227];
    poly[2088] = poly[3] * poly[227];
    poly[2089] = poly[4] * poly[227];
    poly[2090] = poly[5] * poly[227];
    poly[2091] = poly[6] * poly[227];
    poly[2092] = poly[19] * poly[81];
    poly[2093] = poly[19] * poly[82];
    poly[2094] = poly[19] * poly[83];
    poly[2095] = poly[84] * poly[19];
    poly[2096] = poly[19] * poly[85];
    poly[2097] = poly[19] * poly[86];
    poly[2098] = poly[19] * poly[87];
    poly[2099] = poly[1] * poly[228];
}

fn f_polynomials42(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2100] = poly[2] * poly[228];
    poly[2101] = poly[3] * poly[228];
    poly[2102] = poly[4] * poly[228];
    poly[2103] = poly[5] * poly[228];
    poly[2104] = poly[6] * poly[228];
    poly[2105] = poly[11] * poly[224];
    poly[2106] = poly[11] * poly[225];
    poly[2107] = poly[11] * poly[226];
    poly[2108] = poly[11] * poly[227];
    poly[2109] = poly[1] * poly[229];
    poly[2110] = poly[2] * poly[229];
    poly[2111] = poly[3] * poly[229];
    poly[2112] = poly[4] * poly[229];
    poly[2113] = poly[5] * poly[229];
    poly[2114] = poly[6] * poly[229];
    poly[2115] = poly[7] * poly[229];
    poly[2116] = poly[8] * poly[229];
    poly[2117] = poly[9] * poly[229];
    poly[2118] = poly[10] * poly[229];
    poly[2119] = poly[11] * poly[229];
    poly[2120] = poly[109] * poly[19];
    poly[2121] = poly[1] * poly[230];
    poly[2122] = poly[2] * poly[230];
    poly[2123] = poly[3] * poly[230];
    poly[2124] = poly[4] * poly[230];
    poly[2125] = poly[5] * poly[230];
    poly[2126] = poly[6] * poly[230];
    poly[2127] = poly[7] * poly[230];
    poly[2128] = poly[8] * poly[230];
    poly[2129] = poly[9] * poly[230];
    poly[2130] = poly[10] * poly[230];
    poly[2131] = poly[11] * poly[230];
    poly[2132] = poly[19] * poly[121];
    poly[2133] = poly[122] * poly[19];
    poly[2134] = poly[19] * poly[123];
    poly[2135] = poly[1] * poly[231];
    poly[2136] = poly[2] * poly[231];
    poly[2137] = poly[3] * poly[231];
    poly[2138] = poly[4] * poly[231];
    poly[2139] = poly[5] * poly[231];
    poly[2140] = poly[6] * poly[231];
    poly[2141] = poly[7] * poly[231];
    poly[2142] = poly[8] * poly[231];
    poly[2143] = poly[9] * poly[231];
    poly[2144] = poly[10] * poly[231];
    poly[2145] = poly[11] * poly[231];
    poly[2146] = poly[19] * poly[135];
    poly[2147] = poly[19] * poly[136];
    poly[2148] = poly[137] * poly[19];
    poly[2149] = poly[19] * poly[138];
}

fn f_polynomials43(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2150] = poly[19] * poly[139];
    poly[2151] = poly[1] * poly[232];
    poly[2152] = poly[2] * poly[232];
    poly[2153] = poly[3] * poly[232];
    poly[2154] = poly[4] * poly[232];
    poly[2155] = poly[5] * poly[232];
    poly[2156] = poly[6] * poly[232];
    poly[2157] = poly[7] * poly[232];
    poly[2158] = poly[8] * poly[232];
    poly[2159] = poly[9] * poly[232];
    poly[2160] = poly[10] * poly[232];
    poly[2161] = poly[11] * poly[232];
    poly[2162] = poly[19] * poly[151];
    poly[2163] = poly[19] * poly[152];
    poly[2164] = poly[19] * poly[153];
    poly[2165] = poly[154] * poly[19];
    poly[2166] = poly[19] * poly[155];
    poly[2167] = poly[19] * poly[156];
    poly[2168] = poly[19] * poly[157];
    poly[2169] = poly[1] * poly[233];
    poly[2170] = poly[2] * poly[233];
    poly[2171] = poly[3] * poly[233];
    poly[2172] = poly[4] * poly[233];
    poly[2173] = poly[5] * poly[233];
    poly[2174] = poly[6] * poly[233];
    poly[2175] = poly[19] * poly[164];
    poly[2176] = poly[19] * poly[165];
    poly[2177] = poly[19] * poly[166];
    poly[2178] = poly[19] * poly[167];
    poly[2179] = poly[19] * poly[168];
    poly[2180] = poly[19] * poly[169];
    poly[2181] = poly[19] * poly[170];
    poly[2182] = poly[19] * poly[171];
    poly[2183] = poly[11] * poly[233];
    poly[2184] = poly[19] * poly[173];
    poly[2185] = poly[19] * poly[174];
    poly[2186] = poly[19] * poly[175];
    poly[2187] = poly[19] * poly[176];
    poly[2188] = poly[19] * poly[177];
    poly[2189] = poly[19] * poly[178];
    poly[2190] = poly[19] * poly[179];
    poly[2191] = poly[19] * poly[180];
    poly[2192] = poly[19] * poly[181];
    poly[2193] = poly[19] * poly[182];
    poly[2194] = poly[19] * poly[183];
    poly[2195] = poly[1] * poly[234];
    poly[2196] = poly[2] * poly[234];
    poly[2197] = poly[3] * poly[234];
    poly[2198] = poly[4] * poly[234];
    poly[2199] = poly[5] * poly[234];
}

fn f_polynomials44(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2200] = poly[6] * poly[234];
    poly[2201] = poly[17] * poly[224];
    poly[2202] = poly[17] * poly[225];
    poly[2203] = poly[17] * poly[226];
    poly[2204] = poly[17] * poly[227];
    poly[2205] = poly[11] * poly[234];
    poly[2206] = poly[17] * poly[229];
    poly[2207] = poly[17] * poly[230];
    poly[2208] = poly[17] * poly[231];
    poly[2209] = poly[17] * poly[232];
    poly[2210] = poly[17] * poly[233];
    poly[2211] = poly[1] * poly[235];
    poly[2212] = poly[2] * poly[235];
    poly[2213] = poly[3] * poly[235];
    poly[2214] = poly[4] * poly[235];
    poly[2215] = poly[5] * poly[235];
    poly[2216] = poly[6] * poly[235];
    poly[2217] = poly[7] * poly[235];
    poly[2218] = poly[8] * poly[235];
    poly[2219] = poly[9] * poly[235];
    poly[2220] = poly[10] * poly[235];
    poly[2221] = poly[11] * poly[235];
    poly[2222] = poly[12] * poly[235];
    poly[2223] = poly[13] * poly[235];
    poly[2224] = poly[14] * poly[235];
    poly[2225] = poly[15] * poly[235];
    poly[2226] = poly[16] * poly[235];
    poly[2227] = poly[17] * poly[235];
    poly[2228] = poly[1] * poly[236];
    poly[2229] = poly[2] * poly[236];
    poly[2230] = poly[3] * poly[236];
    poly[2231] = poly[4] * poly[236];
    poly[2232] = poly[5] * poly[236];
    poly[2233] = poly[6] * poly[236];
    poly[2234] = poly[236] * poly[7];
    poly[2235] = poly[236] * poly[8];
    poly[2236] = poly[236] * poly[9];
    poly[2237] = poly[236] * poly[10];
    poly[2238] = poly[11] * poly[236];
    poly[2239] = poly[236] * poly[12];
    poly[2240] = poly[236] * poly[13];
    poly[2241] = poly[236] * poly[14];
    poly[2242] = poly[236] * poly[15];
    poly[2243] = poly[236] * poly[16];
    poly[2244] = poly[17] * poly[236];
    poly[2245] = poly[1] * poly[237];
    poly[2246] = poly[2] * poly[237];
    poly[2247] = poly[3] * poly[237];
    poly[2248] = poly[4] * poly[237];
    poly[2249] = poly[5] * poly[237];
}

fn f_polynomials45(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2250] = poly[6] * poly[237];
    poly[2251] = poly[7] * poly[237];
    poly[2252] = poly[8] * poly[237];
    poly[2253] = poly[9] * poly[237];
    poly[2254] = poly[10] * poly[237];
    poly[2255] = poly[11] * poly[237];
    poly[2256] = poly[12] * poly[237];
    poly[2257] = poly[13] * poly[237];
    poly[2258] = poly[14] * poly[237];
    poly[2259] = poly[15] * poly[237];
    poly[2260] = poly[16] * poly[237];
    poly[2261] = poly[17] * poly[237];
    poly[2262] = poly[217] * poly[19];
    poly[2263] = poly[236] * poly[18];
    poly[2264] = poly[1] * poly[239];
    poly[2265] = poly[1] * poly[240];
    poly[2266] = poly[2] * poly[240];
    poly[2267] = poly[1] * poly[241];
    poly[2268] = poly[2] * poly[241];
    poly[2269] = poly[3] * poly[241];
    poly[2270] = poly[1] * poly[242];
    poly[2271] = poly[2] * poly[242];
    poly[2272] = poly[3] * poly[242];
    poly[2273] = poly[4] * poly[242];
    poly[2274] = poly[1] * poly[243];
    poly[2275] = poly[2] * poly[243];
    poly[2276] = poly[3] * poly[243];
    poly[2277] = poly[4] * poly[243];
    poly[2278] = poly[5] * poly[243];
    poly[2279] = poly[1] * poly[244];
    poly[2280] = poly[2] * poly[244];
    poly[2281] = poly[3] * poly[244];
    poly[2282] = poly[4] * poly[244];
    poly[2283] = poly[5] * poly[244];
    poly[2284] = poly[6] * poly[244];
    poly[2285] = poly[54] * poly[20];
    poly[2286] = poly[1] * poly[245];
    poly[2287] = poly[2] * poly[245];
    poly[2288] = poly[3] * poly[245];
    poly[2289] = poly[4] * poly[245];
    poly[2290] = poly[5] * poly[245];
    poly[2291] = poly[6] * poly[245];
    poly[2292] = poly[20] * poly[61];
    poly[2293] = poly[62] * poly[20];
    poly[2294] = poly[20] * poly[63];
    poly[2295] = poly[1] * poly[246];
    poly[2296] = poly[2] * poly[246];
    poly[2297] = poly[3] * poly[246];
    poly[2298] = poly[4] * poly[246];
    poly[2299] = poly[5] * poly[246];
}

fn f_polynomials46(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2300] = poly[6] * poly[246];
    poly[2301] = poly[20] * poly[70];
    poly[2302] = poly[20] * poly[71];
    poly[2303] = poly[72] * poly[20];
    poly[2304] = poly[20] * poly[73];
    poly[2305] = poly[20] * poly[74];
    poly[2306] = poly[1] * poly[247];
    poly[2307] = poly[2] * poly[247];
    poly[2308] = poly[3] * poly[247];
    poly[2309] = poly[4] * poly[247];
    poly[2310] = poly[5] * poly[247];
    poly[2311] = poly[6] * poly[247];
    poly[2312] = poly[20] * poly[81];
    poly[2313] = poly[20] * poly[82];
    poly[2314] = poly[20] * poly[83];
    poly[2315] = poly[84] * poly[20];
    poly[2316] = poly[20] * poly[85];
    poly[2317] = poly[20] * poly[86];
    poly[2318] = poly[20] * poly[87];
    poly[2319] = poly[1] * poly[248];
    poly[2320] = poly[2] * poly[248];
    poly[2321] = poly[3] * poly[248];
    poly[2322] = poly[4] * poly[248];
    poly[2323] = poly[5] * poly[248];
    poly[2324] = poly[6] * poly[248];
    poly[2325] = poly[11] * poly[244];
    poly[2326] = poly[11] * poly[245];
    poly[2327] = poly[11] * poly[246];
    poly[2328] = poly[11] * poly[247];
    poly[2329] = poly[1] * poly[249];
    poly[2330] = poly[2] * poly[249];
    poly[2331] = poly[3] * poly[249];
    poly[2332] = poly[4] * poly[249];
    poly[2333] = poly[5] * poly[249];
    poly[2334] = poly[6] * poly[249];
    poly[2335] = poly[7] * poly[249];
    poly[2336] = poly[8] * poly[249];
    poly[2337] = poly[9] * poly[249];
    poly[2338] = poly[10] * poly[249];
    poly[2339] = poly[11] * poly[249];
    poly[2340] = poly[109] * poly[20];
    poly[2341] = poly[1] * poly[250];
    poly[2342] = poly[2] * poly[250];
    poly[2343] = poly[3] * poly[250];
    poly[2344] = poly[4] * poly[250];
    poly[2345] = poly[5] * poly[250];
    poly[2346] = poly[6] * poly[250];
    poly[2347] = poly[7] * poly[250];
    poly[2348] = poly[8] * poly[250];
    poly[2349] = poly[9] * poly[250];
}

fn f_polynomials47(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2350] = poly[10] * poly[250];
    poly[2351] = poly[11] * poly[250];
    poly[2352] = poly[20] * poly[121];
    poly[2353] = poly[122] * poly[20];
    poly[2354] = poly[20] * poly[123];
    poly[2355] = poly[1] * poly[251];
    poly[2356] = poly[2] * poly[251];
    poly[2357] = poly[3] * poly[251];
    poly[2358] = poly[4] * poly[251];
    poly[2359] = poly[5] * poly[251];
    poly[2360] = poly[6] * poly[251];
    poly[2361] = poly[7] * poly[251];
    poly[2362] = poly[8] * poly[251];
    poly[2363] = poly[9] * poly[251];
    poly[2364] = poly[10] * poly[251];
    poly[2365] = poly[11] * poly[251];
    poly[2366] = poly[20] * poly[135];
    poly[2367] = poly[20] * poly[136];
    poly[2368] = poly[137] * poly[20];
    poly[2369] = poly[20] * poly[138];
    poly[2370] = poly[20] * poly[139];
    poly[2371] = poly[1] * poly[252];
    poly[2372] = poly[2] * poly[252];
    poly[2373] = poly[3] * poly[252];
    poly[2374] = poly[4] * poly[252];
    poly[2375] = poly[5] * poly[252];
    poly[2376] = poly[6] * poly[252];
    poly[2377] = poly[7] * poly[252];
    poly[2378] = poly[8] * poly[252];
    poly[2379] = poly[9] * poly[252];
    poly[2380] = poly[10] * poly[252];
    poly[2381] = poly[11] * poly[252];
    poly[2382] = poly[20] * poly[151];
    poly[2383] = poly[20] * poly[152];
    poly[2384] = poly[20] * poly[153];
    poly[2385] = poly[154] * poly[20];
    poly[2386] = poly[20] * poly[155];
    poly[2387] = poly[20] * poly[156];
    poly[2388] = poly[20] * poly[157];
    poly[2389] = poly[1] * poly[253];
    poly[2390] = poly[2] * poly[253];
    poly[2391] = poly[3] * poly[253];
    poly[2392] = poly[4] * poly[253];
    poly[2393] = poly[5] * poly[253];
    poly[2394] = poly[6] * poly[253];
    poly[2395] = poly[20] * poly[164];
    poly[2396] = poly[20] * poly[165];
    poly[2397] = poly[20] * poly[166];
    poly[2398] = poly[20] * poly[167];
    poly[2399] = poly[20] * poly[168];
}

fn f_polynomials48(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2400] = poly[20] * poly[169];
    poly[2401] = poly[20] * poly[170];
    poly[2402] = poly[20] * poly[171];
    poly[2403] = poly[11] * poly[253];
    poly[2404] = poly[20] * poly[173];
    poly[2405] = poly[20] * poly[174];
    poly[2406] = poly[20] * poly[175];
    poly[2407] = poly[20] * poly[176];
    poly[2408] = poly[20] * poly[177];
    poly[2409] = poly[20] * poly[178];
    poly[2410] = poly[20] * poly[179];
    poly[2411] = poly[20] * poly[180];
    poly[2412] = poly[20] * poly[181];
    poly[2413] = poly[20] * poly[182];
    poly[2414] = poly[20] * poly[183];
    poly[2415] = poly[1] * poly[254];
    poly[2416] = poly[2] * poly[254];
    poly[2417] = poly[3] * poly[254];
    poly[2418] = poly[4] * poly[254];
    poly[2419] = poly[5] * poly[254];
    poly[2420] = poly[6] * poly[254];
    poly[2421] = poly[17] * poly[244];
    poly[2422] = poly[17] * poly[245];
    poly[2423] = poly[17] * poly[246];
    poly[2424] = poly[17] * poly[247];
    poly[2425] = poly[11] * poly[254];
    poly[2426] = poly[17] * poly[249];
    poly[2427] = poly[17] * poly[250];
    poly[2428] = poly[17] * poly[251];
    poly[2429] = poly[17] * poly[252];
    poly[2430] = poly[17] * poly[253];
    poly[2431] = poly[1] * poly[255];
    poly[2432] = poly[2] * poly[255];
    poly[2433] = poly[3] * poly[255];
    poly[2434] = poly[4] * poly[255];
    poly[2435] = poly[5] * poly[255];
    poly[2436] = poly[6] * poly[255];
    poly[2437] = poly[7] * poly[255];
    poly[2438] = poly[8] * poly[255];
    poly[2439] = poly[9] * poly[255];
    poly[2440] = poly[10] * poly[255];
    poly[2441] = poly[11] * poly[255];
    poly[2442] = poly[12] * poly[255];
    poly[2443] = poly[13] * poly[255];
    poly[2444] = poly[14] * poly[255];
    poly[2445] = poly[15] * poly[255];
    poly[2446] = poly[16] * poly[255];
    poly[2447] = poly[17] * poly[255];
    poly[2448] = poly[1] * poly[256];
    poly[2449] = poly[2] * poly[256];
}

fn f_polynomials49(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2450] = poly[3] * poly[256];
    poly[2451] = poly[4] * poly[256];
    poly[2452] = poly[5] * poly[256];
    poly[2453] = poly[6] * poly[256];
    poly[2454] = poly[7] * poly[256];
    poly[2455] = poly[8] * poly[256];
    poly[2456] = poly[9] * poly[256];
    poly[2457] = poly[10] * poly[256];
    poly[2458] = poly[11] * poly[256];
    poly[2459] = poly[12] * poly[256];
    poly[2460] = poly[13] * poly[256];
    poly[2461] = poly[14] * poly[256];
    poly[2462] = poly[15] * poly[256];
    poly[2463] = poly[16] * poly[256];
    poly[2464] = poly[17] * poly[256];
    poly[2465] = mono[603] + mono[604];
    poly[2466] = poly[1] * poly[257];
    poly[2467] = poly[2] * poly[257];
    poly[2468] = poly[3] * poly[257];
    poly[2469] = poly[4] * poly[257];
    poly[2470] = poly[5] * poly[257];
    poly[2471] = poly[6] * poly[257];
    poly[2472] = poly[257] * poly[7];
    poly[2473] = poly[257] * poly[8];
    poly[2474] = poly[257] * poly[9];
    poly[2475] = poly[257] * poly[10];
    poly[2476] = poly[11] * poly[257];
    poly[2477] = poly[257] * poly[12];
    poly[2478] = poly[257] * poly[13];
    poly[2479] = poly[257] * poly[14];
    poly[2480] = poly[257] * poly[15];
    poly[2481] = poly[257] * poly[16];
    poly[2482] = poly[17] * poly[257];
    poly[2483] = poly[1] * poly[258];
    poly[2484] = poly[2] * poly[258];
    poly[2485] = poly[3] * poly[258];
    poly[2486] = poly[4] * poly[258];
    poly[2487] = poly[5] * poly[258];
    poly[2488] = poly[6] * poly[258];
    poly[2489] = poly[7] * poly[258];
    poly[2490] = poly[8] * poly[258];
    poly[2491] = poly[9] * poly[258];
    poly[2492] = poly[10] * poly[258];
    poly[2493] = poly[11] * poly[258];
    poly[2494] = poly[12] * poly[258];
    poly[2495] = poly[13] * poly[258];
    poly[2496] = poly[14] * poly[258];
    poly[2497] = poly[15] * poly[258];
    poly[2498] = poly[16] * poly[258];
    poly[2499] = poly[17] * poly[258];
}

fn f_polynomials50(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2500] = poly[217] * poly[20];
    poly[2501] = poly[18] * poly[256] - poly[2465];
    poly[2502] = poly[257] * poly[18];
    poly[2503] = poly[1] * poly[259];
    poly[2504] = poly[2] * poly[259];
    poly[2505] = poly[3] * poly[259];
    poly[2506] = poly[4] * poly[259];
    poly[2507] = poly[5] * poly[259];
    poly[2508] = poly[6] * poly[259];
    poly[2509] = poly[7] * poly[259];
    poly[2510] = poly[8] * poly[259];
    poly[2511] = poly[9] * poly[259];
    poly[2512] = poly[10] * poly[259];
    poly[2513] = poly[11] * poly[259];
    poly[2514] = poly[12] * poly[259];
    poly[2515] = poly[13] * poly[259];
    poly[2516] = poly[14] * poly[259];
    poly[2517] = poly[15] * poly[259];
    poly[2518] = poly[16] * poly[259];
    poly[2519] = poly[17] * poly[259];
    poly[2520] = poly[19] * poly[255] - poly[2465];
    poly[2521] = poly[236] * poly[20];
    poly[2522] = poly[257] * poly[19];
    poly[2523] = poly[18] * poly[259] - poly[2520];
    poly[2524] = poly[1] * poly[261];
    poly[2525] = poly[1] * poly[262];
    poly[2526] = poly[2] * poly[262];
    poly[2527] = poly[1] * poly[263];
    poly[2528] = poly[2] * poly[263];
    poly[2529] = poly[3] * poly[263];
    poly[2530] = poly[1] * poly[264];
    poly[2531] = poly[2] * poly[264];
    poly[2532] = poly[3] * poly[264];
    poly[2533] = poly[4] * poly[264];
    poly[2534] = poly[1] * poly[265];
    poly[2535] = poly[2] * poly[265];
    poly[2536] = poly[3] * poly[265];
    poly[2537] = poly[4] * poly[265];
    poly[2538] = poly[5] * poly[265];
    poly[2539] = poly[1] * poly[266];
    poly[2540] = poly[2] * poly[266];
    poly[2541] = poly[3] * poly[266];
    poly[2542] = poly[4] * poly[266];
    poly[2543] = poly[5] * poly[266];
    poly[2544] = poly[6] * poly[266];
    poly[2545] = poly[54] * poly[21];
    poly[2546] = poly[1] * poly[267];
    poly[2547] = poly[2] * poly[267];
    poly[2548] = poly[3] * poly[267];
    poly[2549] = poly[4] * poly[267];
}

fn f_polynomials51(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2550] = poly[5] * poly[267];
    poly[2551] = poly[6] * poly[267];
    poly[2552] = poly[21] * poly[61];
    poly[2553] = poly[62] * poly[21];
    poly[2554] = poly[21] * poly[63];
    poly[2555] = poly[1] * poly[268];
    poly[2556] = poly[2] * poly[268];
    poly[2557] = poly[3] * poly[268];
    poly[2558] = poly[4] * poly[268];
    poly[2559] = poly[5] * poly[268];
    poly[2560] = poly[6] * poly[268];
    poly[2561] = poly[21] * poly[70];
    poly[2562] = poly[21] * poly[71];
    poly[2563] = poly[72] * poly[21];
    poly[2564] = poly[21] * poly[73];
    poly[2565] = poly[21] * poly[74];
    poly[2566] = poly[1] * poly[269];
    poly[2567] = poly[2] * poly[269];
    poly[2568] = poly[3] * poly[269];
    poly[2569] = poly[4] * poly[269];
    poly[2570] = poly[5] * poly[269];
    poly[2571] = poly[6] * poly[269];
    poly[2572] = poly[21] * poly[81];
    poly[2573] = poly[21] * poly[82];
    poly[2574] = poly[21] * poly[83];
    poly[2575] = poly[84] * poly[21];
    poly[2576] = poly[21] * poly[85];
    poly[2577] = poly[21] * poly[86];
    poly[2578] = poly[21] * poly[87];
    poly[2579] = poly[1] * poly[270];
    poly[2580] = poly[2] * poly[270];
    poly[2581] = poly[3] * poly[270];
    poly[2582] = poly[4] * poly[270];
    poly[2583] = poly[5] * poly[270];
    poly[2584] = poly[6] * poly[270];
    poly[2585] = poly[11] * poly[266];
    poly[2586] = poly[11] * poly[267];
    poly[2587] = poly[11] * poly[268];
    poly[2588] = poly[11] * poly[269];
    poly[2589] = poly[1] * poly[271];
    poly[2590] = poly[2] * poly[271];
    poly[2591] = poly[3] * poly[271];
    poly[2592] = poly[4] * poly[271];
    poly[2593] = poly[5] * poly[271];
    poly[2594] = poly[6] * poly[271];
    poly[2595] = poly[7] * poly[271];
    poly[2596] = poly[8] * poly[271];
    poly[2597] = poly[9] * poly[271];
    poly[2598] = poly[10] * poly[271];
    poly[2599] = poly[11] * poly[271];
}

fn f_polynomials52(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2600] = poly[109] * poly[21];
    poly[2601] = poly[1] * poly[272];
    poly[2602] = poly[2] * poly[272];
    poly[2603] = poly[3] * poly[272];
    poly[2604] = poly[4] * poly[272];
    poly[2605] = poly[5] * poly[272];
    poly[2606] = poly[6] * poly[272];
    poly[2607] = poly[7] * poly[272];
    poly[2608] = poly[8] * poly[272];
    poly[2609] = poly[9] * poly[272];
    poly[2610] = poly[10] * poly[272];
    poly[2611] = poly[11] * poly[272];
    poly[2612] = poly[21] * poly[121];
    poly[2613] = poly[122] * poly[21];
    poly[2614] = poly[21] * poly[123];
    poly[2615] = poly[1] * poly[273];
    poly[2616] = poly[2] * poly[273];
    poly[2617] = poly[3] * poly[273];
    poly[2618] = poly[4] * poly[273];
    poly[2619] = poly[5] * poly[273];
    poly[2620] = poly[6] * poly[273];
    poly[2621] = poly[7] * poly[273];
    poly[2622] = poly[8] * poly[273];
    poly[2623] = poly[9] * poly[273];
    poly[2624] = poly[10] * poly[273];
    poly[2625] = poly[11] * poly[273];
    poly[2626] = poly[21] * poly[135];
    poly[2627] = poly[21] * poly[136];
    poly[2628] = poly[137] * poly[21];
    poly[2629] = poly[21] * poly[138];
    poly[2630] = poly[21] * poly[139];
    poly[2631] = poly[1] * poly[274];
    poly[2632] = poly[2] * poly[274];
    poly[2633] = poly[3] * poly[274];
    poly[2634] = poly[4] * poly[274];
    poly[2635] = poly[5] * poly[274];
    poly[2636] = poly[6] * poly[274];
    poly[2637] = poly[7] * poly[274];
    poly[2638] = poly[8] * poly[274];
    poly[2639] = poly[9] * poly[274];
    poly[2640] = poly[10] * poly[274];
    poly[2641] = poly[11] * poly[274];
    poly[2642] = poly[21] * poly[151];
    poly[2643] = poly[21] * poly[152];
    poly[2644] = poly[21] * poly[153];
    poly[2645] = poly[154] * poly[21];
    poly[2646] = poly[21] * poly[155];
    poly[2647] = poly[21] * poly[156];
    poly[2648] = poly[21] * poly[157];
    poly[2649] = poly[1] * poly[275];
}

fn f_polynomials53(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2650] = poly[2] * poly[275];
    poly[2651] = poly[3] * poly[275];
    poly[2652] = poly[4] * poly[275];
    poly[2653] = poly[5] * poly[275];
    poly[2654] = poly[6] * poly[275];
    poly[2655] = poly[21] * poly[164];
    poly[2656] = poly[21] * poly[165];
    poly[2657] = poly[21] * poly[166];
    poly[2658] = poly[21] * poly[167];
    poly[2659] = poly[21] * poly[168];
    poly[2660] = poly[21] * poly[169];
    poly[2661] = poly[21] * poly[170];
    poly[2662] = poly[21] * poly[171];
    poly[2663] = poly[11] * poly[275];
    poly[2664] = poly[21] * poly[173];
    poly[2665] = poly[21] * poly[174];
    poly[2666] = poly[21] * poly[175];
    poly[2667] = poly[21] * poly[176];
    poly[2668] = poly[21] * poly[177];
    poly[2669] = poly[21] * poly[178];
    poly[2670] = poly[21] * poly[179];
    poly[2671] = poly[21] * poly[180];
    poly[2672] = poly[21] * poly[181];
    poly[2673] = poly[21] * poly[182];
    poly[2674] = poly[21] * poly[183];
    poly[2675] = poly[1] * poly[276];
    poly[2676] = poly[2] * poly[276];
    poly[2677] = poly[3] * poly[276];
    poly[2678] = poly[4] * poly[276];
    poly[2679] = poly[5] * poly[276];
    poly[2680] = poly[6] * poly[276];
    poly[2681] = poly[17] * poly[266];
    poly[2682] = poly[17] * poly[267];
    poly[2683] = poly[17] * poly[268];
    poly[2684] = poly[17] * poly[269];
    poly[2685] = poly[11] * poly[276];
    poly[2686] = poly[17] * poly[271];
    poly[2687] = poly[17] * poly[272];
    poly[2688] = poly[17] * poly[273];
    poly[2689] = poly[17] * poly[274];
    poly[2690] = poly[17] * poly[275];
    poly[2691] = poly[1] * poly[277];
    poly[2692] = poly[2] * poly[277];
    poly[2693] = poly[3] * poly[277];
    poly[2694] = poly[4] * poly[277];
    poly[2695] = poly[5] * poly[277];
    poly[2696] = poly[6] * poly[277];
    poly[2697] = poly[7] * poly[277];
    poly[2698] = poly[8] * poly[277];
    poly[2699] = poly[9] * poly[277];
}

fn f_polynomials54(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2700] = poly[10] * poly[277];
    poly[2701] = poly[11] * poly[277];
    poly[2702] = poly[12] * poly[277];
    poly[2703] = poly[13] * poly[277];
    poly[2704] = poly[14] * poly[277];
    poly[2705] = poly[15] * poly[277];
    poly[2706] = poly[16] * poly[277];
    poly[2707] = poly[17] * poly[277];
    poly[2708] = poly[1] * poly[278];
    poly[2709] = poly[2] * poly[278];
    poly[2710] = poly[3] * poly[278];
    poly[2711] = poly[4] * poly[278];
    poly[2712] = poly[5] * poly[278];
    poly[2713] = poly[6] * poly[278];
    poly[2714] = poly[7] * poly[278];
    poly[2715] = poly[8] * poly[278];
    poly[2716] = poly[9] * poly[278];
    poly[2717] = poly[10] * poly[278];
    poly[2718] = poly[11] * poly[278];
    poly[2719] = poly[12] * poly[278];
    poly[2720] = poly[13] * poly[278];
    poly[2721] = poly[14] * poly[278];
    poly[2722] = poly[15] * poly[278];
    poly[2723] = poly[16] * poly[278];
    poly[2724] = poly[17] * poly[278];
    poly[2725] = mono[605] + mono[606];
    poly[2726] = poly[1] * poly[279];
    poly[2727] = poly[2] * poly[279];
    poly[2728] = poly[3] * poly[279];
    poly[2729] = poly[4] * poly[279];
    poly[2730] = poly[5] * poly[279];
    poly[2731] = poly[6] * poly[279];
    poly[2732] = poly[7] * poly[279];
    poly[2733] = poly[8] * poly[279];
    poly[2734] = poly[9] * poly[279];
    poly[2735] = poly[10] * poly[279];
    poly[2736] = poly[11] * poly[279];
    poly[2737] = poly[12] * poly[279];
    poly[2738] = poly[13] * poly[279];
    poly[2739] = poly[14] * poly[279];
    poly[2740] = poly[15] * poly[279];
    poly[2741] = poly[16] * poly[279];
    poly[2742] = poly[17] * poly[279];
    poly[2743] = mono[607] + mono[608];
    poly[2744] = mono[609] + mono[610];
    poly[2745] = poly[1] * poly[280];
    poly[2746] = poly[2] * poly[280];
    poly[2747] = poly[3] * poly[280];
    poly[2748] = poly[4] * poly[280];
    poly[2749] = poly[5] * poly[280];
}

fn f_polynomials55(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2750] = poly[6] * poly[280];
    poly[2751] = poly[280] * poly[7];
    poly[2752] = poly[280] * poly[8];
    poly[2753] = poly[280] * poly[9];
    poly[2754] = poly[280] * poly[10];
    poly[2755] = poly[11] * poly[280];
    poly[2756] = poly[280] * poly[12];
    poly[2757] = poly[280] * poly[13];
    poly[2758] = poly[280] * poly[14];
    poly[2759] = poly[280] * poly[15];
    poly[2760] = poly[280] * poly[16];
    poly[2761] = poly[17] * poly[280];
    poly[2762] = poly[1] * poly[281];
    poly[2763] = poly[2] * poly[281];
    poly[2764] = poly[3] * poly[281];
    poly[2765] = poly[4] * poly[281];
    poly[2766] = poly[5] * poly[281];
    poly[2767] = poly[6] * poly[281];
    poly[2768] = poly[7] * poly[281];
    poly[2769] = poly[8] * poly[281];
    poly[2770] = poly[9] * poly[281];
    poly[2771] = poly[10] * poly[281];
    poly[2772] = poly[11] * poly[281];
    poly[2773] = poly[12] * poly[281];
    poly[2774] = poly[13] * poly[281];
    poly[2775] = poly[14] * poly[281];
    poly[2776] = poly[15] * poly[281];
    poly[2777] = poly[16] * poly[281];
    poly[2778] = poly[17] * poly[281];
    poly[2779] = poly[217] * poly[21];
    poly[2780] = poly[18] * poly[278] - poly[2725];
    poly[2781] = poly[18] * poly[279] - poly[2743];
    poly[2782] = poly[280] * poly[18];
    poly[2783] = poly[1] * poly[282];
    poly[2784] = poly[2] * poly[282];
    poly[2785] = poly[3] * poly[282];
    poly[2786] = poly[4] * poly[282];
    poly[2787] = poly[5] * poly[282];
    poly[2788] = poly[6] * poly[282];
    poly[2789] = poly[7] * poly[282];
    poly[2790] = poly[8] * poly[282];
    poly[2791] = poly[9] * poly[282];
    poly[2792] = poly[10] * poly[282];
    poly[2793] = poly[11] * poly[282];
    poly[2794] = poly[12] * poly[282];
    poly[2795] = poly[13] * poly[282];
    poly[2796] = poly[14] * poly[282];
    poly[2797] = poly[15] * poly[282];
    poly[2798] = poly[16] * poly[282];
    poly[2799] = poly[17] * poly[282];
}

fn f_polynomials56(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2800] = poly[19] * poly[277] - poly[2725];
    poly[2801] = poly[236] * poly[21];
    poly[2802] = poly[19] * poly[279] - poly[2744];
    poly[2803] = poly[280] * poly[19];
    poly[2804] = poly[18] * poly[282] - poly[2800];
    poly[2805] = poly[1] * poly[283];
    poly[2806] = poly[2] * poly[283];
    poly[2807] = poly[3] * poly[283];
    poly[2808] = poly[4] * poly[283];
    poly[2809] = poly[5] * poly[283];
    poly[2810] = poly[6] * poly[283];
    poly[2811] = poly[7] * poly[283];
    poly[2812] = poly[8] * poly[283];
    poly[2813] = poly[9] * poly[283];
    poly[2814] = poly[10] * poly[283];
    poly[2815] = poly[11] * poly[283];
    poly[2816] = poly[12] * poly[283];
    poly[2817] = poly[13] * poly[283];
    poly[2818] = poly[14] * poly[283];
    poly[2819] = poly[15] * poly[283];
    poly[2820] = poly[16] * poly[283];
    poly[2821] = poly[17] * poly[283];
    poly[2822] = poly[20] * poly[277] - poly[2743];
    poly[2823] = poly[20] * poly[278] - poly[2744];
    poly[2824] = poly[257] * poly[21];
    poly[2825] = poly[280] * poly[20];
    poly[2826] = poly[18] * poly[283] - poly[2822];
    poly[2827] = poly[19] * poly[283] - poly[2823];
    poly[2828] = poly[1] * poly[285];
    poly[2829] = poly[1] * poly[286];
    poly[2830] = poly[2] * poly[286];
    poly[2831] = poly[1] * poly[287];
    poly[2832] = poly[2] * poly[287];
    poly[2833] = poly[3] * poly[287];
    poly[2834] = poly[1] * poly[288];
    poly[2835] = poly[2] * poly[288];
    poly[2836] = poly[3] * poly[288];
    poly[2837] = poly[4] * poly[288];
    poly[2838] = poly[1] * poly[289];
    poly[2839] = poly[2] * poly[289];
    poly[2840] = poly[3] * poly[289];
    poly[2841] = poly[4] * poly[289];
    poly[2842] = poly[5] * poly[289];
    poly[2843] = poly[1] * poly[290];
    poly[2844] = poly[2] * poly[290];
    poly[2845] = poly[3] * poly[290];
    poly[2846] = poly[4] * poly[290];
    poly[2847] = poly[5] * poly[290];
    poly[2848] = poly[6] * poly[290];
    poly[2849] = poly[1] * poly[291];
}

fn f_polynomials57(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2850] = poly[2] * poly[291];
    poly[2851] = poly[3] * poly[291];
    poly[2852] = poly[4] * poly[291];
    poly[2853] = poly[5] * poly[291];
    poly[2854] = poly[6] * poly[291];
    poly[2855] = mono[611] + mono[612] + mono[613] + mono[614];
    poly[2856] = poly[1] * poly[292];
    poly[2857] = poly[2] * poly[292];
    poly[2858] = poly[3] * poly[292];
    poly[2859] = poly[4] * poly[292];
    poly[2860] = poly[5] * poly[292];
    poly[2861] = poly[6] * poly[292];
    poly[2862] = mono[615] + mono[616] + mono[617] + mono[618];
    poly[2863] = mono[619] + mono[620] + mono[621] + mono[622];
    poly[2864] = poly[1] * poly[293];
    poly[2865] = poly[2] * poly[293];
    poly[2866] = poly[3] * poly[293];
    poly[2867] = poly[4] * poly[293];
    poly[2868] = poly[5] * poly[293];
    poly[2869] = poly[6] * poly[293];
    poly[2870] = mono[623] + mono[624] + mono[625] + mono[626];
    poly[2871] = mono[627] + mono[628] + mono[629] + mono[630];
    poly[2872] = mono[631] + mono[632] + mono[633] + mono[634];
    poly[2873] = poly[1] * poly[294];
    poly[2874] = poly[2] * poly[294];
    poly[2875] = poly[3] * poly[294];
    poly[2876] = poly[4] * poly[294];
    poly[2877] = poly[5] * poly[294];
    poly[2878] = poly[6] * poly[294];
    poly[2879] = poly[54] * poly[22];
    poly[2880] = poly[7] * poly[291] - poly[2855];
    poly[2881] = poly[7] * poly[292] - poly[2862];
    poly[2882] = poly[7] * poly[293] - poly[2870];
    poly[2883] = poly[1] * poly[295];
    poly[2884] = poly[2] * poly[295];
    poly[2885] = poly[3] * poly[295];
    poly[2886] = poly[4] * poly[295];
    poly[2887] = poly[5] * poly[295];
    poly[2888] = poly[6] * poly[295];
    poly[2889] = poly[8] * poly[290] - poly[2855];
    poly[2890] = poly[62] * poly[22];
    poly[2891] = poly[8] * poly[292] - poly[2863];
    poly[2892] = poly[8] * poly[293] - poly[2871];
    poly[2893] = poly[7] * poly[295] - poly[2889];
    poly[2894] = poly[1] * poly[296];
    poly[2895] = poly[2] * poly[296];
    poly[2896] = poly[3] * poly[296];
    poly[2897] = poly[4] * poly[296];
    poly[2898] = poly[5] * poly[296];
    poly[2899] = poly[6] * poly[296];
}

fn f_polynomials58(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2900] = poly[9] * poly[290] - poly[2862];
    poly[2901] = poly[9] * poly[291] - poly[2863];
    poly[2902] = poly[72] * poly[22];
    poly[2903] = poly[9] * poly[293] - poly[2872];
    poly[2904] = poly[7] * poly[296] - poly[2900];
    poly[2905] = poly[8] * poly[296] - poly[2901];
    poly[2906] = poly[1] * poly[297];
    poly[2907] = poly[2] * poly[297];
    poly[2908] = poly[3] * poly[297];
    poly[2909] = poly[4] * poly[297];
    poly[2910] = poly[5] * poly[297];
    poly[2911] = poly[6] * poly[297];
    poly[2912] = poly[10] * poly[290] - poly[2870];
    poly[2913] = poly[10] * poly[291] - poly[2871];
    poly[2914] = poly[10] * poly[292] - poly[2872];
    poly[2915] = poly[84] * poly[22];
    poly[2916] = poly[7] * poly[297] - poly[2912];
    poly[2917] = poly[8] * poly[297] - poly[2913];
    poly[2918] = poly[9] * poly[297] - poly[2914];
    poly[2919] = poly[1] * poly[298];
    poly[2920] = poly[2] * poly[298];
    poly[2921] = poly[3] * poly[298];
    poly[2922] = poly[4] * poly[298];
    poly[2923] = poly[5] * poly[298];
    poly[2924] = poly[6] * poly[298];
    poly[2925] = poly[11] * poly[290];
    poly[2926] = poly[11] * poly[291];
    poly[2927] = poly[11] * poly[292];
    poly[2928] = poly[11] * poly[293];
    poly[2929] = poly[11] * poly[294];
    poly[2930] = poly[11] * poly[295];
    poly[2931] = poly[11] * poly[296];
    poly[2932] = poly[11] * poly[297];
    poly[2933] = poly[1] * poly[299];
    poly[2934] = poly[2] * poly[299];
    poly[2935] = poly[3] * poly[299];
    poly[2936] = poly[4] * poly[299];
    poly[2937] = poly[5] * poly[299];
    poly[2938] = poly[6] * poly[299];
    poly[2939] = poly[12] * poly[290];
    poly[2940] = poly[12] * poly[291];
    poly[2941] = poly[12] * poly[292];
    poly[2942] = poly[12] * poly[293];
    poly[2943] = poly[12] * poly[294];
    poly[2944] = poly[12] * poly[295];
    poly[2945] = poly[12] * poly[296];
    poly[2946] = poly[12] * poly[297];
    poly[2947] = poly[11] * poly[299];
    poly[2948] = poly[109] * poly[22];
    poly[2949] = poly[1] * poly[300];
}

fn f_polynomials59(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[2950] = poly[2] * poly[300];
    poly[2951] = poly[3] * poly[300];
    poly[2952] = poly[4] * poly[300];
    poly[2953] = poly[5] * poly[300];
    poly[2954] = poly[6] * poly[300];
    poly[2955] = poly[13] * poly[290];
    poly[2956] = poly[13] * poly[291];
    poly[2957] = poly[13] * poly[292];
    poly[2958] = poly[13] * poly[293];
    poly[2959] = poly[13] * poly[294];
    poly[2960] = poly[13] * poly[295];
    poly[2961] = poly[13] * poly[296];
    poly[2962] = poly[13] * poly[297];
    poly[2963] = poly[11] * poly[300];
    poly[2964] = poly[22] * poly[121];
    poly[2965] = poly[122] * poly[22];
    poly[2966] = poly[22] * poly[123];
    poly[2967] = poly[1] * poly[301];
    poly[2968] = poly[2] * poly[301];
    poly[2969] = poly[3] * poly[301];
    poly[2970] = poly[4] * poly[301];
    poly[2971] = poly[5] * poly[301];
    poly[2972] = poly[6] * poly[301];
    poly[2973] = poly[14] * poly[290];
    poly[2974] = poly[14] * poly[291];
    poly[2975] = poly[14] * poly[292];
    poly[2976] = poly[14] * poly[293];
    poly[2977] = poly[14] * poly[294];
    poly[2978] = poly[14] * poly[295];
    poly[2979] = poly[14] * poly[296];
    poly[2980] = poly[14] * poly[297];
    poly[2981] = poly[11] * poly[301];
    poly[2982] = poly[22] * poly[135];
    poly[2983] = poly[22] * poly[136];
    poly[2984] = poly[137] * poly[22];
    poly[2985] = poly[22] * poly[138];
    poly[2986] = poly[22] * poly[139];
    poly[2987] = poly[1] * poly[302];
    poly[2988] = poly[2] * poly[302];
    poly[2989] = poly[3] * poly[302];
    poly[2990] = poly[4] * poly[302];
    poly[2991] = poly[5] * poly[302];
    poly[2992] = poly[6] * poly[302];
    poly[2993] = poly[15] * poly[290];
    poly[2994] = poly[15] * poly[291];
    poly[2995] = poly[15] * poly[292];
    poly[2996] = poly[15] * poly[293];
    poly[2997] = poly[15] * poly[294];
    poly[2998] = poly[15] * poly[295];
    poly[2999] = poly[15] * poly[296];
}

fn f_polynomials60(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3000] = poly[15] * poly[297];
    poly[3001] = poly[11] * poly[302];
    poly[3002] = poly[22] * poly[151];
    poly[3003] = poly[22] * poly[152];
    poly[3004] = poly[22] * poly[153];
    poly[3005] = poly[154] * poly[22];
    poly[3006] = poly[22] * poly[155];
    poly[3007] = poly[22] * poly[156];
    poly[3008] = poly[22] * poly[157];
    poly[3009] = poly[1] * poly[303];
    poly[3010] = poly[2] * poly[303];
    poly[3011] = poly[3] * poly[303];
    poly[3012] = poly[4] * poly[303];
    poly[3013] = poly[5] * poly[303];
    poly[3014] = poly[6] * poly[303];
    poly[3015] = mono[635]
        + mono[636]
        + mono[637]
        + mono[638]
        + mono[639]
        + mono[640]
        + mono[641]
        + mono[642];
    poly[3016] = mono[643]
        + mono[644]
        + mono[645]
        + mono[646]
        + mono[647]
        + mono[648]
        + mono[649]
        + mono[650];
    poly[3017] = mono[651]
        + mono[652]
        + mono[653]
        + mono[654]
        + mono[655]
        + mono[656]
        + mono[657]
        + mono[658];
    poly[3018] = mono[659]
        + mono[660]
        + mono[661]
        + mono[662]
        + mono[663]
        + mono[664]
        + mono[665]
        + mono[666];
    poly[3019] = poly[7] * poly[303] - poly[3015];
    poly[3020] = poly[8] * poly[303] - poly[3016];
    poly[3021] = poly[9] * poly[303] - poly[3017];
    poly[3022] = poly[10] * poly[303] - poly[3018];
    poly[3023] = poly[11] * poly[303];
    poly[3024] = mono[667]
        + mono[668]
        + mono[669]
        + mono[670]
        + mono[671]
        + mono[672]
        + mono[673]
        + mono[674];
    poly[3025] = mono[675]
        + mono[676]
        + mono[677]
        + mono[678]
        + mono[679]
        + mono[680]
        + mono[681]
        + mono[682];
    poly[3026] = mono[683]
        + mono[684]
        + mono[685]
        + mono[686]
        + mono[687]
        + mono[688]
        + mono[689]
        + mono[690];
    poly[3027] = mono[691]
        + mono[692]
        + mono[693]
        + mono[694]
        + mono[695]
        + mono[696]
        + mono[697]
        + mono[698];
    poly[3028] = mono[699] + mono[700] + mono[701] + mono[702];
    poly[3029] = poly[12] * poly[303] - poly[3024];
    poly[3030] = poly[13] * poly[303] - poly[3025];
    poly[3031] = poly[14] * poly[303] - poly[3026];
    poly[3032] = poly[15] * poly[303] - poly[3027];
    poly[3033] = poly[1] * poly[304];
    poly[3034] = poly[2] * poly[304];
    poly[3035] = poly[3] * poly[304];
    poly[3036] = poly[4] * poly[304];
    poly[3037] = poly[5] * poly[304];
    poly[3038] = poly[6] * poly[304];
    poly[3039] = poly[16] * poly[290] - poly[3015];
    poly[3040] = poly[16] * poly[291] - poly[3016];
    poly[3041] = poly[16] * poly[292] - poly[3017];
    poly[3042] = poly[16] * poly[293] - poly[3018];
    poly[3043] = poly[7] * poly[304] - poly[3039];
    poly[3044] = poly[8] * poly[304] - poly[3040];
    poly[3045] = poly[9] * poly[304] - poly[3041];
    poly[3046] = poly[10] * poly[304] - poly[3042];
    poly[3047] = poly[11] * poly[304];
    poly[3048] = poly[22] * poly[173] - poly[3024];
    poly[3049] = poly[22] * poly[174] - poly[3025];
}

fn f_polynomials61(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3050] = poly[22] * poly[175] - poly[3026];
    poly[3051] = poly[22] * poly[176] - poly[3027];
    poly[3052] = poly[22] * poly[177];
    poly[3053] = poly[22] * poly[178] - poly[3028];
    poly[3054] = poly[12] * poly[304] - poly[3048];
    poly[3055] = poly[13] * poly[304] - poly[3049];
    poly[3056] = poly[14] * poly[304] - poly[3050];
    poly[3057] = poly[15] * poly[304] - poly[3051];
    poly[3058] = poly[22] * poly[183];
    poly[3059] = poly[1] * poly[305];
    poly[3060] = poly[2] * poly[305];
    poly[3061] = poly[3] * poly[305];
    poly[3062] = poly[4] * poly[305];
    poly[3063] = poly[5] * poly[305];
    poly[3064] = poly[6] * poly[305];
    poly[3065] = poly[17] * poly[290];
    poly[3066] = poly[17] * poly[291];
    poly[3067] = poly[17] * poly[292];
    poly[3068] = poly[17] * poly[293];
    poly[3069] = poly[17] * poly[294];
    poly[3070] = poly[17] * poly[295];
    poly[3071] = poly[17] * poly[296];
    poly[3072] = poly[17] * poly[297];
    poly[3073] = poly[11] * poly[305];
    poly[3074] = poly[17] * poly[299];
    poly[3075] = poly[17] * poly[300];
    poly[3076] = poly[17] * poly[301];
    poly[3077] = poly[17] * poly[302];
    poly[3078] = poly[17] * poly[303];
    poly[3079] = poly[17] * poly[304];
    poly[3080] = poly[1] * poly[306];
    poly[3081] = poly[2] * poly[306];
    poly[3082] = poly[3] * poly[306];
    poly[3083] = poly[4] * poly[306];
    poly[3084] = poly[5] * poly[306];
    poly[3085] = poly[6] * poly[306];
    poly[3086] = mono[703] + mono[704] + mono[705] + mono[706];
    poly[3087] = mono[707] + mono[708] + mono[709] + mono[710];
    poly[3088] = mono[711] + mono[712] + mono[713] + mono[714];
    poly[3089] = mono[715] + mono[716] + mono[717] + mono[718];
    poly[3090] = poly[7] * poly[306] - poly[3086];
    poly[3091] = poly[8] * poly[306] - poly[3087];
    poly[3092] = poly[9] * poly[306] - poly[3088];
    poly[3093] = poly[10] * poly[306] - poly[3089];
    poly[3094] = poly[11] * poly[306];
    poly[3095] = poly[12] * poly[306];
    poly[3096] = poly[13] * poly[306];
    poly[3097] = poly[14] * poly[306];
    poly[3098] = poly[15] * poly[306];
    poly[3099] = mono[719]
        + mono[720]
        + mono[721]
        + mono[722]
        + mono[723]
        + mono[724]
        + mono[725]
        + mono[726];
}

fn f_polynomials62(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3100] = poly[16] * poly[306] - poly[3099];
    poly[3101] = poly[17] * poly[306];
    poly[3102] = poly[1] * poly[307];
    poly[3103] = poly[2] * poly[307];
    poly[3104] = poly[3] * poly[307];
    poly[3105] = poly[4] * poly[307];
    poly[3106] = poly[5] * poly[307];
    poly[3107] = poly[6] * poly[307];
    poly[3108] = mono[727] + mono[728] + mono[729] + mono[730];
    poly[3109] = mono[731] + mono[732] + mono[733] + mono[734];
    poly[3110] = mono[735] + mono[736] + mono[737] + mono[738];
    poly[3111] = mono[739] + mono[740] + mono[741] + mono[742];
    poly[3112] = poly[7] * poly[307] - poly[3108];
    poly[3113] = poly[8] * poly[307] - poly[3109];
    poly[3114] = poly[9] * poly[307] - poly[3110];
    poly[3115] = poly[10] * poly[307] - poly[3111];
    poly[3116] = poly[11] * poly[307];
    poly[3117] = poly[12] * poly[307];
    poly[3118] = poly[13] * poly[307];
    poly[3119] = poly[14] * poly[307];
    poly[3120] = poly[15] * poly[307];
    poly[3121] = mono[743]
        + mono[744]
        + mono[745]
        + mono[746]
        + mono[747]
        + mono[748]
        + mono[749]
        + mono[750];
    poly[3122] = poly[16] * poly[307] - poly[3121];
    poly[3123] = poly[17] * poly[307];
    poly[3124] = mono[751] + mono[752] + mono[753] + mono[754];
    poly[3125] = poly[1] * poly[308];
    poly[3126] = poly[2] * poly[308];
    poly[3127] = poly[3] * poly[308];
    poly[3128] = poly[4] * poly[308];
    poly[3129] = poly[5] * poly[308];
    poly[3130] = poly[6] * poly[308];
    poly[3131] = mono[755] + mono[756] + mono[757] + mono[758];
    poly[3132] = mono[759] + mono[760] + mono[761] + mono[762];
    poly[3133] = mono[763] + mono[764] + mono[765] + mono[766];
    poly[3134] = mono[767] + mono[768] + mono[769] + mono[770];
    poly[3135] = poly[7] * poly[308] - poly[3131];
    poly[3136] = poly[8] * poly[308] - poly[3132];
    poly[3137] = poly[9] * poly[308] - poly[3133];
    poly[3138] = poly[10] * poly[308] - poly[3134];
    poly[3139] = poly[11] * poly[308];
    poly[3140] = poly[12] * poly[308];
    poly[3141] = poly[13] * poly[308];
    poly[3142] = poly[14] * poly[308];
    poly[3143] = poly[15] * poly[308];
    poly[3144] = mono[771]
        + mono[772]
        + mono[773]
        + mono[774]
        + mono[775]
        + mono[776]
        + mono[777]
        + mono[778];
    poly[3145] = poly[16] * poly[308] - poly[3144];
    poly[3146] = poly[17] * poly[308];
    poly[3147] = mono[779] + mono[780] + mono[781] + mono[782];
    poly[3148] = mono[783] + mono[784] + mono[785] + mono[786];
    poly[3149] = poly[1] * poly[309];
}

fn f_polynomials63(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3150] = poly[2] * poly[309];
    poly[3151] = poly[3] * poly[309];
    poly[3152] = poly[4] * poly[309];
    poly[3153] = poly[5] * poly[309];
    poly[3154] = poly[6] * poly[309];
    poly[3155] = mono[787] + mono[788] + mono[789] + mono[790];
    poly[3156] = mono[791] + mono[792] + mono[793] + mono[794];
    poly[3157] = mono[795] + mono[796] + mono[797] + mono[798];
    poly[3158] = mono[799] + mono[800] + mono[801] + mono[802];
    poly[3159] = poly[7] * poly[309] - poly[3155];
    poly[3160] = poly[8] * poly[309] - poly[3156];
    poly[3161] = poly[9] * poly[309] - poly[3157];
    poly[3162] = poly[10] * poly[309] - poly[3158];
    poly[3163] = poly[11] * poly[309];
    poly[3164] = poly[12] * poly[309];
    poly[3165] = poly[13] * poly[309];
    poly[3166] = poly[14] * poly[309];
    poly[3167] = poly[15] * poly[309];
    poly[3168] = mono[803]
        + mono[804]
        + mono[805]
        + mono[806]
        + mono[807]
        + mono[808]
        + mono[809]
        + mono[810];
    poly[3169] = poly[16] * poly[309] - poly[3168];
    poly[3170] = poly[17] * poly[309];
    poly[3171] = mono[811] + mono[812] + mono[813] + mono[814];
    poly[3172] = mono[815] + mono[816] + mono[817] + mono[818];
    poly[3173] = mono[819] + mono[820] + mono[821] + mono[822];
    poly[3174] = poly[1] * poly[310];
    poly[3175] = poly[2] * poly[310];
    poly[3176] = poly[3] * poly[310];
    poly[3177] = poly[4] * poly[310];
    poly[3178] = poly[5] * poly[310];
    poly[3179] = poly[6] * poly[310];
    poly[3180] = poly[7] * poly[310];
    poly[3181] = poly[8] * poly[310];
    poly[3182] = poly[9] * poly[310];
    poly[3183] = poly[10] * poly[310];
    poly[3184] = poly[11] * poly[310];
    poly[3185] = poly[12] * poly[310];
    poly[3186] = poly[13] * poly[310];
    poly[3187] = poly[14] * poly[310];
    poly[3188] = poly[15] * poly[310];
    poly[3189] = poly[16] * poly[310];
    poly[3190] = poly[17] * poly[310];
    poly[3191] = poly[1] * poly[311];
    poly[3192] = poly[2] * poly[311];
    poly[3193] = poly[3] * poly[311];
    poly[3194] = poly[4] * poly[311];
    poly[3195] = poly[5] * poly[311];
    poly[3196] = poly[6] * poly[311];
    poly[3197] = mono[823] + mono[824];
    poly[3198] = mono[825] + mono[826];
    poly[3199] = mono[827] + mono[828];
}

fn f_polynomials64(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3200] = mono[829] + mono[830];
    poly[3201] = poly[7] * poly[311] - poly[3197];
    poly[3202] = poly[8] * poly[311] - poly[3198];
    poly[3203] = poly[9] * poly[311] - poly[3199];
    poly[3204] = poly[10] * poly[311] - poly[3200];
    poly[3205] = poly[11] * poly[311];
    poly[3206] = poly[12] * poly[311];
    poly[3207] = poly[13] * poly[311];
    poly[3208] = poly[14] * poly[311];
    poly[3209] = poly[15] * poly[311];
    poly[3210] = mono[831] + mono[832] + mono[833] + mono[834];
    poly[3211] = poly[16] * poly[311] - poly[3210];
    poly[3212] = poly[17] * poly[311];
    poly[3213] = poly[1] * poly[312];
    poly[3214] = poly[2] * poly[312];
    poly[3215] = poly[3] * poly[312];
    poly[3216] = poly[4] * poly[312];
    poly[3217] = poly[5] * poly[312];
    poly[3218] = poly[6] * poly[312];
    poly[3219] = poly[18] * poly[290] - poly[3086];
    poly[3220] = poly[18] * poly[291] - poly[3087];
    poly[3221] = poly[18] * poly[292] - poly[3088];
    poly[3222] = poly[18] * poly[293] - poly[3089];
    poly[3223] = poly[7] * poly[312] - poly[3219];
    poly[3224] = poly[8] * poly[312] - poly[3220];
    poly[3225] = poly[9] * poly[312] - poly[3221];
    poly[3226] = poly[10] * poly[312] - poly[3222];
    poly[3227] = poly[11] * poly[312];
    poly[3228] = poly[12] * poly[312];
    poly[3229] = poly[13] * poly[312];
    poly[3230] = poly[14] * poly[312];
    poly[3231] = poly[15] * poly[312];
    poly[3232] = poly[18] * poly[303] - poly[3099];
    poly[3233] = poly[16] * poly[312] - poly[3232];
    poly[3234] = poly[17] * poly[312];
    poly[3235] = poly[217] * poly[22];
    poly[3236] = poly[18] * poly[307] - poly[3124];
    poly[3237] = poly[18] * poly[308] - poly[3147];
    poly[3238] = poly[18] * poly[309] - poly[3171];
    poly[3239] = poly[18] * poly[310];
    poly[3240] = poly[18] * poly[311];
    poly[3241] = poly[1] * poly[313];
    poly[3242] = poly[2] * poly[313];
    poly[3243] = poly[3] * poly[313];
    poly[3244] = poly[4] * poly[313];
    poly[3245] = poly[5] * poly[313];
    poly[3246] = poly[6] * poly[313];
    poly[3247] = poly[19] * poly[290] - poly[3108];
    poly[3248] = poly[19] * poly[291] - poly[3109];
    poly[3249] = poly[19] * poly[292] - poly[3110];
}

fn f_polynomials65(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3250] = poly[19] * poly[293] - poly[3111];
    poly[3251] = poly[7] * poly[313] - poly[3247];
    poly[3252] = poly[8] * poly[313] - poly[3248];
    poly[3253] = poly[9] * poly[313] - poly[3249];
    poly[3254] = poly[10] * poly[313] - poly[3250];
    poly[3255] = poly[11] * poly[313];
    poly[3256] = poly[12] * poly[313];
    poly[3257] = poly[13] * poly[313];
    poly[3258] = poly[14] * poly[313];
    poly[3259] = poly[15] * poly[313];
    poly[3260] = poly[19] * poly[303] - poly[3121];
    poly[3261] = poly[16] * poly[313] - poly[3260];
    poly[3262] = poly[17] * poly[313];
    poly[3263] = poly[19] * poly[306] - poly[3124];
    poly[3264] = poly[236] * poly[22];
    poly[3265] = poly[19] * poly[308] - poly[3148];
    poly[3266] = poly[19] * poly[309] - poly[3172];
    poly[3267] = poly[19] * poly[310];
    poly[3268] = poly[19] * poly[311];
    poly[3269] = poly[18] * poly[313] - poly[3263];
    poly[3270] = poly[1] * poly[314];
    poly[3271] = poly[2] * poly[314];
    poly[3272] = poly[3] * poly[314];
    poly[3273] = poly[4] * poly[314];
    poly[3274] = poly[5] * poly[314];
    poly[3275] = poly[6] * poly[314];
    poly[3276] = poly[20] * poly[290] - poly[3131];
    poly[3277] = poly[20] * poly[291] - poly[3132];
    poly[3278] = poly[20] * poly[292] - poly[3133];
    poly[3279] = poly[20] * poly[293] - poly[3134];
    poly[3280] = poly[7] * poly[314] - poly[3276];
    poly[3281] = poly[8] * poly[314] - poly[3277];
    poly[3282] = poly[9] * poly[314] - poly[3278];
    poly[3283] = poly[10] * poly[314] - poly[3279];
    poly[3284] = poly[11] * poly[314];
    poly[3285] = poly[12] * poly[314];
    poly[3286] = poly[13] * poly[314];
    poly[3287] = poly[14] * poly[314];
    poly[3288] = poly[15] * poly[314];
    poly[3289] = poly[20] * poly[303] - poly[3144];
    poly[3290] = poly[16] * poly[314] - poly[3289];
    poly[3291] = poly[17] * poly[314];
    poly[3292] = poly[20] * poly[306] - poly[3147];
    poly[3293] = poly[20] * poly[307] - poly[3148];
    poly[3294] = poly[257] * poly[22];
    poly[3295] = poly[20] * poly[309] - poly[3173];
    poly[3296] = poly[20] * poly[310];
    poly[3297] = poly[20] * poly[311];
    poly[3298] = poly[18] * poly[314] - poly[3292];
    poly[3299] = poly[19] * poly[314] - poly[3293];
}

fn f_polynomials66(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3300] = poly[1] * poly[315];
    poly[3301] = poly[2] * poly[315];
    poly[3302] = poly[3] * poly[315];
    poly[3303] = poly[4] * poly[315];
    poly[3304] = poly[5] * poly[315];
    poly[3305] = poly[6] * poly[315];
    poly[3306] = poly[21] * poly[290] - poly[3155];
    poly[3307] = poly[21] * poly[291] - poly[3156];
    poly[3308] = poly[21] * poly[292] - poly[3157];
    poly[3309] = poly[21] * poly[293] - poly[3158];
    poly[3310] = poly[7] * poly[315] - poly[3306];
    poly[3311] = poly[8] * poly[315] - poly[3307];
    poly[3312] = poly[9] * poly[315] - poly[3308];
    poly[3313] = poly[10] * poly[315] - poly[3309];
    poly[3314] = poly[11] * poly[315];
    poly[3315] = poly[12] * poly[315];
    poly[3316] = poly[13] * poly[315];
    poly[3317] = poly[14] * poly[315];
    poly[3318] = poly[15] * poly[315];
    poly[3319] = poly[21] * poly[303] - poly[3168];
    poly[3320] = poly[16] * poly[315] - poly[3319];
    poly[3321] = poly[17] * poly[315];
    poly[3322] = poly[21] * poly[306] - poly[3171];
    poly[3323] = poly[21] * poly[307] - poly[3172];
    poly[3324] = poly[21] * poly[308] - poly[3173];
    poly[3325] = poly[280] * poly[22];
    poly[3326] = poly[21] * poly[310];
    poly[3327] = poly[21] * poly[311];
    poly[3328] = poly[18] * poly[315] - poly[3322];
    poly[3329] = poly[19] * poly[315] - poly[3323];
    poly[3330] = poly[20] * poly[315] - poly[3324];
    poly[3331] = poly[1] * poly[316];
    poly[3332] = poly[2] * poly[316];
    poly[3333] = poly[3] * poly[316];
    poly[3334] = poly[4] * poly[316];
    poly[3335] = poly[5] * poly[316];
    poly[3336] = poly[6] * poly[316];
    poly[3337] = poly[7] * poly[316];
    poly[3338] = poly[8] * poly[316];
    poly[3339] = poly[9] * poly[316];
    poly[3340] = poly[10] * poly[316];
    poly[3341] = poly[11] * poly[316];
    poly[3342] = poly[12] * poly[316];
    poly[3343] = poly[13] * poly[316];
    poly[3344] = poly[14] * poly[316];
    poly[3345] = poly[15] * poly[316];
    poly[3346] = poly[16] * poly[316];
    poly[3347] = poly[17] * poly[316];
    poly[3348] = mono[835] + mono[836];
    poly[3349] = mono[837] + mono[838];
}

fn f_polynomials67(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3350] = mono[839] + mono[840];
    poly[3351] = mono[841] + mono[842];
    poly[3352] = mono[843] + mono[844] + mono[845] + mono[846];
    poly[3353] = poly[18] * poly[316] - poly[3348];
    poly[3354] = poly[19] * poly[316] - poly[3349];
    poly[3355] = poly[20] * poly[316] - poly[3350];
    poly[3356] = poly[21] * poly[316] - poly[3351];
    poly[3357] = poly[1] * poly[318];
    poly[3358] = poly[1] * poly[319];
    poly[3359] = poly[2] * poly[319];
    poly[3360] = poly[1] * poly[320];
    poly[3361] = poly[2] * poly[320];
    poly[3362] = poly[3] * poly[320];
    poly[3363] = poly[1] * poly[321];
    poly[3364] = poly[2] * poly[321];
    poly[3365] = poly[3] * poly[321];
    poly[3366] = poly[4] * poly[321];
    poly[3367] = poly[1] * poly[322];
    poly[3368] = poly[2] * poly[322];
    poly[3369] = poly[3] * poly[322];
    poly[3370] = poly[4] * poly[322];
    poly[3371] = poly[5] * poly[322];
    poly[3372] = poly[1] * poly[323];
    poly[3373] = poly[2] * poly[323];
    poly[3374] = poly[3] * poly[323];
    poly[3375] = poly[4] * poly[323];
    poly[3376] = poly[5] * poly[323];
    poly[3377] = poly[6] * poly[323];
    poly[3378] = poly[54] * poly[23];
    poly[3379] = poly[1] * poly[324];
    poly[3380] = poly[2] * poly[324];
    poly[3381] = poly[3] * poly[324];
    poly[3382] = poly[4] * poly[324];
    poly[3383] = poly[5] * poly[324];
    poly[3384] = poly[6] * poly[324];
    poly[3385] = poly[23] * poly[61];
    poly[3386] = poly[62] * poly[23];
    poly[3387] = poly[23] * poly[63];
    poly[3388] = poly[1] * poly[325];
    poly[3389] = poly[2] * poly[325];
    poly[3390] = poly[3] * poly[325];
    poly[3391] = poly[4] * poly[325];
    poly[3392] = poly[5] * poly[325];
    poly[3393] = poly[6] * poly[325];
    poly[3394] = poly[23] * poly[70];
    poly[3395] = poly[23] * poly[71];
    poly[3396] = poly[72] * poly[23];
    poly[3397] = poly[23] * poly[73];
    poly[3398] = poly[23] * poly[74];
    poly[3399] = poly[1] * poly[326];
}

fn f_polynomials68(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3400] = poly[2] * poly[326];
    poly[3401] = poly[3] * poly[326];
    poly[3402] = poly[4] * poly[326];
    poly[3403] = poly[5] * poly[326];
    poly[3404] = poly[6] * poly[326];
    poly[3405] = poly[23] * poly[81];
    poly[3406] = poly[23] * poly[82];
    poly[3407] = poly[23] * poly[83];
    poly[3408] = poly[84] * poly[23];
    poly[3409] = poly[23] * poly[85];
    poly[3410] = poly[23] * poly[86];
    poly[3411] = poly[23] * poly[87];
    poly[3412] = poly[1] * poly[327];
    poly[3413] = poly[2] * poly[327];
    poly[3414] = poly[3] * poly[327];
    poly[3415] = poly[4] * poly[327];
    poly[3416] = poly[5] * poly[327];
    poly[3417] = poly[6] * poly[327];
    poly[3418] = poly[11] * poly[323];
    poly[3419] = poly[11] * poly[324];
    poly[3420] = poly[11] * poly[325];
    poly[3421] = poly[11] * poly[326];
    poly[3422] = poly[1] * poly[328];
    poly[3423] = poly[2] * poly[328];
    poly[3424] = poly[3] * poly[328];
    poly[3425] = poly[4] * poly[328];
    poly[3426] = poly[5] * poly[328];
    poly[3427] = poly[6] * poly[328];
    poly[3428] = poly[7] * poly[328];
    poly[3429] = poly[8] * poly[328];
    poly[3430] = poly[9] * poly[328];
    poly[3431] = poly[10] * poly[328];
    poly[3432] = poly[11] * poly[328];
    poly[3433] = poly[1] * poly[329];
    poly[3434] = poly[2] * poly[329];
    poly[3435] = poly[3] * poly[329];
    poly[3436] = poly[4] * poly[329];
    poly[3437] = poly[5] * poly[329];
    poly[3438] = poly[6] * poly[329];
    poly[3439] = poly[7] * poly[329];
    poly[3440] = poly[8] * poly[329];
    poly[3441] = poly[9] * poly[329];
    poly[3442] = poly[10] * poly[329];
    poly[3443] = poly[11] * poly[329];
    poly[3444] = mono[847] + mono[848] + mono[849] + mono[850];
    poly[3445] = poly[1] * poly[330];
    poly[3446] = poly[2] * poly[330];
    poly[3447] = poly[3] * poly[330];
    poly[3448] = poly[4] * poly[330];
    poly[3449] = poly[5] * poly[330];
}

fn f_polynomials69(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3450] = poly[6] * poly[330];
    poly[3451] = poly[7] * poly[330];
    poly[3452] = poly[8] * poly[330];
    poly[3453] = poly[9] * poly[330];
    poly[3454] = poly[10] * poly[330];
    poly[3455] = poly[11] * poly[330];
    poly[3456] = mono[851] + mono[852] + mono[853] + mono[854];
    poly[3457] = mono[855] + mono[856] + mono[857] + mono[858];
    poly[3458] = poly[1] * poly[331];
    poly[3459] = poly[2] * poly[331];
    poly[3460] = poly[3] * poly[331];
    poly[3461] = poly[4] * poly[331];
    poly[3462] = poly[5] * poly[331];
    poly[3463] = poly[6] * poly[331];
    poly[3464] = poly[7] * poly[331];
    poly[3465] = poly[8] * poly[331];
    poly[3466] = poly[9] * poly[331];
    poly[3467] = poly[10] * poly[331];
    poly[3468] = poly[11] * poly[331];
    poly[3469] = mono[859] + mono[860] + mono[861] + mono[862];
    poly[3470] = mono[863] + mono[864] + mono[865] + mono[866];
    poly[3471] = mono[867] + mono[868] + mono[869] + mono[870];
    poly[3472] = poly[1] * poly[332];
    poly[3473] = poly[2] * poly[332];
    poly[3474] = poly[3] * poly[332];
    poly[3475] = poly[4] * poly[332];
    poly[3476] = poly[5] * poly[332];
    poly[3477] = poly[6] * poly[332];
    poly[3478] = mono[871]
        + mono[872]
        + mono[873]
        + mono[874]
        + mono[875]
        + mono[876]
        + mono[877]
        + mono[878];
    poly[3479] = mono[879]
        + mono[880]
        + mono[881]
        + mono[882]
        + mono[883]
        + mono[884]
        + mono[885]
        + mono[886];
    poly[3480] = mono[887]
        + mono[888]
        + mono[889]
        + mono[890]
        + mono[891]
        + mono[892]
        + mono[893]
        + mono[894];
    poly[3481] = mono[895]
        + mono[896]
        + mono[897]
        + mono[898]
        + mono[899]
        + mono[900]
        + mono[901]
        + mono[902];
    poly[3482] = poly[7] * poly[332] - poly[3478];
    poly[3483] = poly[8] * poly[332] - poly[3479];
    poly[3484] = poly[9] * poly[332] - poly[3480];
    poly[3485] = poly[10] * poly[332] - poly[3481];
    poly[3486] = poly[11] * poly[332];
    poly[3487] = mono[903]
        + mono[904]
        + mono[905]
        + mono[906]
        + mono[907]
        + mono[908]
        + mono[909]
        + mono[910];
    poly[3488] = mono[911]
        + mono[912]
        + mono[913]
        + mono[914]
        + mono[915]
        + mono[916]
        + mono[917]
        + mono[918];
    poly[3489] = mono[919]
        + mono[920]
        + mono[921]
        + mono[922]
        + mono[923]
        + mono[924]
        + mono[925]
        + mono[926];
    poly[3490] = mono[927]
        + mono[928]
        + mono[929]
        + mono[930]
        + mono[931]
        + mono[932]
        + mono[933]
        + mono[934];
    poly[3491] = mono[935] + mono[936] + mono[937] + mono[938];
    poly[3492] = poly[1] * poly[333];
    poly[3493] = poly[2] * poly[333];
    poly[3494] = poly[3] * poly[333];
    poly[3495] = poly[4] * poly[333];
    poly[3496] = poly[5] * poly[333];
    poly[3497] = poly[6] * poly[333];
    poly[3498] = poly[7] * poly[333];
    poly[3499] = poly[8] * poly[333];
}

fn f_polynomials70(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3500] = poly[9] * poly[333];
    poly[3501] = poly[10] * poly[333];
    poly[3502] = poly[11] * poly[333];
    poly[3503] = poly[109] * poly[23];
    poly[3504] = poly[12] * poly[329] - poly[3444];
    poly[3505] = poly[12] * poly[330] - poly[3456];
    poly[3506] = poly[12] * poly[331] - poly[3469];
    poly[3507] = poly[12] * poly[332] - poly[3487];
    poly[3508] = poly[1] * poly[334];
    poly[3509] = poly[2] * poly[334];
    poly[3510] = poly[3] * poly[334];
    poly[3511] = poly[4] * poly[334];
    poly[3512] = poly[5] * poly[334];
    poly[3513] = poly[6] * poly[334];
    poly[3514] = poly[7] * poly[334];
    poly[3515] = poly[8] * poly[334];
    poly[3516] = poly[9] * poly[334];
    poly[3517] = poly[10] * poly[334];
    poly[3518] = poly[11] * poly[334];
    poly[3519] = poly[13] * poly[328] - poly[3444];
    poly[3520] = poly[122] * poly[23];
    poly[3521] = poly[13] * poly[330] - poly[3457];
    poly[3522] = poly[13] * poly[331] - poly[3470];
    poly[3523] = poly[13] * poly[332] - poly[3488];
    poly[3524] = poly[12] * poly[334] - poly[3519];
    poly[3525] = poly[1] * poly[335];
    poly[3526] = poly[2] * poly[335];
    poly[3527] = poly[3] * poly[335];
    poly[3528] = poly[4] * poly[335];
    poly[3529] = poly[5] * poly[335];
    poly[3530] = poly[6] * poly[335];
    poly[3531] = poly[7] * poly[335];
    poly[3532] = poly[8] * poly[335];
    poly[3533] = poly[9] * poly[335];
    poly[3534] = poly[10] * poly[335];
    poly[3535] = poly[11] * poly[335];
    poly[3536] = poly[14] * poly[328] - poly[3456];
    poly[3537] = poly[14] * poly[329] - poly[3457];
    poly[3538] = poly[137] * poly[23];
    poly[3539] = poly[14] * poly[331] - poly[3471];
    poly[3540] = poly[14] * poly[332] - poly[3489];
    poly[3541] = poly[12] * poly[335] - poly[3536];
    poly[3542] = poly[13] * poly[335] - poly[3537];
    poly[3543] = poly[1] * poly[336];
    poly[3544] = poly[2] * poly[336];
    poly[3545] = poly[3] * poly[336];
    poly[3546] = poly[4] * poly[336];
    poly[3547] = poly[5] * poly[336];
    poly[3548] = poly[6] * poly[336];
    poly[3549] = poly[7] * poly[336];
}

fn f_polynomials71(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3550] = poly[8] * poly[336];
    poly[3551] = poly[9] * poly[336];
    poly[3552] = poly[10] * poly[336];
    poly[3553] = poly[11] * poly[336];
    poly[3554] = poly[15] * poly[328] - poly[3469];
    poly[3555] = poly[15] * poly[329] - poly[3470];
    poly[3556] = poly[15] * poly[330] - poly[3471];
    poly[3557] = poly[154] * poly[23];
    poly[3558] = poly[15] * poly[332] - poly[3490];
    poly[3559] = poly[12] * poly[336] - poly[3554];
    poly[3560] = poly[13] * poly[336] - poly[3555];
    poly[3561] = poly[14] * poly[336] - poly[3556];
    poly[3562] = poly[1] * poly[337];
    poly[3563] = poly[2] * poly[337];
    poly[3564] = poly[3] * poly[337];
    poly[3565] = poly[4] * poly[337];
    poly[3566] = poly[5] * poly[337];
    poly[3567] = poly[6] * poly[337];
    poly[3568] = poly[23] * poly[164] - poly[3478];
    poly[3569] = poly[23] * poly[165] - poly[3479];
    poly[3570] = poly[23] * poly[166] - poly[3480];
    poly[3571] = poly[23] * poly[167] - poly[3481];
    poly[3572] = poly[7] * poly[337] - poly[3568];
    poly[3573] = poly[8] * poly[337] - poly[3569];
    poly[3574] = poly[9] * poly[337] - poly[3570];
    poly[3575] = poly[10] * poly[337] - poly[3571];
    poly[3576] = poly[11] * poly[337];
    poly[3577] = poly[16] * poly[328] - poly[3487];
    poly[3578] = poly[16] * poly[329] - poly[3488];
    poly[3579] = poly[16] * poly[330] - poly[3489];
    poly[3580] = poly[16] * poly[331] - poly[3490];
    poly[3581] = poly[23] * poly[177];
    poly[3582] = poly[23] * poly[178];
    poly[3583] = poly[12] * poly[337] - poly[3577];
    poly[3584] = poly[13] * poly[337] - poly[3578];
    poly[3585] = poly[14] * poly[337] - poly[3579];
    poly[3586] = poly[15] * poly[337] - poly[3580];
    poly[3587] = poly[23] * poly[183] - poly[3491];
    poly[3588] = poly[1] * poly[338];
    poly[3589] = poly[2] * poly[338];
    poly[3590] = poly[3] * poly[338];
    poly[3591] = poly[4] * poly[338];
    poly[3592] = poly[5] * poly[338];
    poly[3593] = poly[6] * poly[338];
    poly[3594] = poly[17] * poly[323];
    poly[3595] = poly[17] * poly[324];
    poly[3596] = poly[17] * poly[325];
    poly[3597] = poly[17] * poly[326];
    poly[3598] = poly[11] * poly[338];
    poly[3599] = poly[17] * poly[328];
}

fn f_polynomials72(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3600] = poly[17] * poly[329];
    poly[3601] = poly[17] * poly[330];
    poly[3602] = poly[17] * poly[331];
    poly[3603] = poly[17] * poly[332];
    poly[3604] = poly[17] * poly[333];
    poly[3605] = poly[17] * poly[334];
    poly[3606] = poly[17] * poly[335];
    poly[3607] = poly[17] * poly[336];
    poly[3608] = poly[17] * poly[337];
    poly[3609] = poly[1] * poly[339];
    poly[3610] = poly[2] * poly[339];
    poly[3611] = poly[3] * poly[339];
    poly[3612] = poly[4] * poly[339];
    poly[3613] = poly[5] * poly[339];
    poly[3614] = poly[6] * poly[339];
    poly[3615] = poly[7] * poly[339];
    poly[3616] = poly[8] * poly[339];
    poly[3617] = poly[9] * poly[339];
    poly[3618] = poly[10] * poly[339];
    poly[3619] = poly[11] * poly[339];
    poly[3620] = mono[939] + mono[940] + mono[941] + mono[942];
    poly[3621] = mono[943] + mono[944] + mono[945] + mono[946];
    poly[3622] = mono[947] + mono[948] + mono[949] + mono[950];
    poly[3623] = mono[951] + mono[952] + mono[953] + mono[954];
    poly[3624] = mono[955]
        + mono[956]
        + mono[957]
        + mono[958]
        + mono[959]
        + mono[960]
        + mono[961]
        + mono[962];
    poly[3625] = poly[12] * poly[339] - poly[3620];
    poly[3626] = poly[13] * poly[339] - poly[3621];
    poly[3627] = poly[14] * poly[339] - poly[3622];
    poly[3628] = poly[15] * poly[339] - poly[3623];
    poly[3629] = poly[16] * poly[339] - poly[3624];
    poly[3630] = poly[17] * poly[339];
    poly[3631] = poly[1] * poly[340];
    poly[3632] = poly[2] * poly[340];
    poly[3633] = poly[3] * poly[340];
    poly[3634] = poly[4] * poly[340];
    poly[3635] = poly[5] * poly[340];
    poly[3636] = poly[6] * poly[340];
    poly[3637] = poly[7] * poly[340];
    poly[3638] = poly[8] * poly[340];
    poly[3639] = poly[9] * poly[340];
    poly[3640] = poly[10] * poly[340];
    poly[3641] = poly[11] * poly[340];
    poly[3642] = mono[963] + mono[964] + mono[965] + mono[966];
    poly[3643] = mono[967] + mono[968] + mono[969] + mono[970];
    poly[3644] = mono[971] + mono[972] + mono[973] + mono[974];
    poly[3645] = mono[975] + mono[976] + mono[977] + mono[978];
    poly[3646] = mono[979]
        + mono[980]
        + mono[981]
        + mono[982]
        + mono[983]
        + mono[984]
        + mono[985]
        + mono[986];
    poly[3647] = poly[12] * poly[340] - poly[3642];
    poly[3648] = poly[13] * poly[340] - poly[3643];
    poly[3649] = poly[14] * poly[340] - poly[3644];
}

fn f_polynomials73(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3650] = poly[15] * poly[340] - poly[3645];
    poly[3651] = poly[16] * poly[340] - poly[3646];
    poly[3652] = poly[17] * poly[340];
    poly[3653] = mono[987] + mono[988] + mono[989] + mono[990];
    poly[3654] = poly[1] * poly[341];
    poly[3655] = poly[2] * poly[341];
    poly[3656] = poly[3] * poly[341];
    poly[3657] = poly[4] * poly[341];
    poly[3658] = poly[5] * poly[341];
    poly[3659] = poly[6] * poly[341];
    poly[3660] = poly[7] * poly[341];
    poly[3661] = poly[8] * poly[341];
    poly[3662] = poly[9] * poly[341];
    poly[3663] = poly[10] * poly[341];
    poly[3664] = poly[11] * poly[341];
    poly[3665] = mono[991] + mono[992] + mono[993] + mono[994];
    poly[3666] = mono[995] + mono[996] + mono[997] + mono[998];
    poly[3667] = mono[999] + mono[1000] + mono[1001] + mono[1002];
    poly[3668] = mono[1003] + mono[1004] + mono[1005] + mono[1006];
    poly[3669] = mono[1007]
        + mono[1008]
        + mono[1009]
        + mono[1010]
        + mono[1011]
        + mono[1012]
        + mono[1013]
        + mono[1014];
    poly[3670] = poly[12] * poly[341] - poly[3665];
    poly[3671] = poly[13] * poly[341] - poly[3666];
    poly[3672] = poly[14] * poly[341] - poly[3667];
    poly[3673] = poly[15] * poly[341] - poly[3668];
    poly[3674] = poly[16] * poly[341] - poly[3669];
    poly[3675] = poly[17] * poly[341];
    poly[3676] = mono[1015] + mono[1016] + mono[1017] + mono[1018];
    poly[3677] = mono[1019] + mono[1020] + mono[1021] + mono[1022];
    poly[3678] = poly[1] * poly[342];
    poly[3679] = poly[2] * poly[342];
    poly[3680] = poly[3] * poly[342];
    poly[3681] = poly[4] * poly[342];
    poly[3682] = poly[5] * poly[342];
    poly[3683] = poly[6] * poly[342];
    poly[3684] = poly[7] * poly[342];
    poly[3685] = poly[8] * poly[342];
    poly[3686] = poly[9] * poly[342];
    poly[3687] = poly[10] * poly[342];
    poly[3688] = poly[11] * poly[342];
    poly[3689] = mono[1023] + mono[1024] + mono[1025] + mono[1026];
    poly[3690] = mono[1027] + mono[1028] + mono[1029] + mono[1030];
    poly[3691] = mono[1031] + mono[1032] + mono[1033] + mono[1034];
    poly[3692] = mono[1035] + mono[1036] + mono[1037] + mono[1038];
    poly[3693] = mono[1039]
        + mono[1040]
        + mono[1041]
        + mono[1042]
        + mono[1043]
        + mono[1044]
        + mono[1045]
        + mono[1046];
    poly[3694] = poly[12] * poly[342] - poly[3689];
    poly[3695] = poly[13] * poly[342] - poly[3690];
    poly[3696] = poly[14] * poly[342] - poly[3691];
    poly[3697] = poly[15] * poly[342] - poly[3692];
    poly[3698] = poly[16] * poly[342] - poly[3693];
    poly[3699] = poly[17] * poly[342];
}

fn f_polynomials74(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3700] = mono[1047] + mono[1048] + mono[1049] + mono[1050];
    poly[3701] = mono[1051] + mono[1052] + mono[1053] + mono[1054];
    poly[3702] = mono[1055] + mono[1056] + mono[1057] + mono[1058];
    poly[3703] = poly[1] * poly[343];
    poly[3704] = poly[2] * poly[343];
    poly[3705] = poly[3] * poly[343];
    poly[3706] = poly[4] * poly[343];
    poly[3707] = poly[5] * poly[343];
    poly[3708] = poly[6] * poly[343];
    poly[3709] = mono[1059]
        + mono[1060]
        + mono[1061]
        + mono[1062]
        + mono[1063]
        + mono[1064]
        + mono[1065]
        + mono[1066];
    poly[3710] = mono[1067]
        + mono[1068]
        + mono[1069]
        + mono[1070]
        + mono[1071]
        + mono[1072]
        + mono[1073]
        + mono[1074];
    poly[3711] = mono[1075]
        + mono[1076]
        + mono[1077]
        + mono[1078]
        + mono[1079]
        + mono[1080]
        + mono[1081]
        + mono[1082];
    poly[3712] = mono[1083]
        + mono[1084]
        + mono[1085]
        + mono[1086]
        + mono[1087]
        + mono[1088]
        + mono[1089]
        + mono[1090];
    poly[3713] = poly[7] * poly[343] - poly[3709];
    poly[3714] = poly[8] * poly[343] - poly[3710];
    poly[3715] = poly[9] * poly[343] - poly[3711];
    poly[3716] = poly[10] * poly[343] - poly[3712];
    poly[3717] = poly[11] * poly[343];
    poly[3718] = mono[1091]
        + mono[1092]
        + mono[1093]
        + mono[1094]
        + mono[1095]
        + mono[1096]
        + mono[1097]
        + mono[1098];
    poly[3719] = mono[1099]
        + mono[1100]
        + mono[1101]
        + mono[1102]
        + mono[1103]
        + mono[1104]
        + mono[1105]
        + mono[1106];
    poly[3720] = mono[1107]
        + mono[1108]
        + mono[1109]
        + mono[1110]
        + mono[1111]
        + mono[1112]
        + mono[1113]
        + mono[1114];
    poly[3721] = mono[1115]
        + mono[1116]
        + mono[1117]
        + mono[1118]
        + mono[1119]
        + mono[1120]
        + mono[1121]
        + mono[1122];
    poly[3722] = mono[1123]
        + mono[1124]
        + mono[1125]
        + mono[1126]
        + mono[1127]
        + mono[1128]
        + mono[1129]
        + mono[1130];
    poly[3723] = mono[1131]
        + mono[1132]
        + mono[1133]
        + mono[1134]
        + mono[1135]
        + mono[1136]
        + mono[1137]
        + mono[1138];
    poly[3724] = poly[12] * poly[343] - poly[3718];
    poly[3725] = poly[13] * poly[343] - poly[3719];
    poly[3726] = poly[14] * poly[343] - poly[3720];
    poly[3727] = poly[15] * poly[343] - poly[3721];
    poly[3728] = mono[1139]
        + mono[1140]
        + mono[1141]
        + mono[1142]
        + mono[1143]
        + mono[1144]
        + mono[1145]
        + mono[1146];
    poly[3729] = poly[16] * poly[343] - poly[3728] - poly[3723] - poly[3722];
    poly[3730] = poly[17] * poly[343];
    poly[3731] = mono[1147]
        + mono[1148]
        + mono[1149]
        + mono[1150]
        + mono[1151]
        + mono[1152]
        + mono[1153]
        + mono[1154];
    poly[3732] = mono[1155]
        + mono[1156]
        + mono[1157]
        + mono[1158]
        + mono[1159]
        + mono[1160]
        + mono[1161]
        + mono[1162];
    poly[3733] = mono[1163]
        + mono[1164]
        + mono[1165]
        + mono[1166]
        + mono[1167]
        + mono[1168]
        + mono[1169]
        + mono[1170];
    poly[3734] = mono[1171]
        + mono[1172]
        + mono[1173]
        + mono[1174]
        + mono[1175]
        + mono[1176]
        + mono[1177]
        + mono[1178];
    poly[3735] = mono[1179] + mono[1180] + mono[1181] + mono[1182];
    poly[3736] = poly[1] * poly[344];
    poly[3737] = poly[2] * poly[344];
    poly[3738] = poly[3] * poly[344];
    poly[3739] = poly[4] * poly[344];
    poly[3740] = poly[5] * poly[344];
    poly[3741] = poly[6] * poly[344];
    poly[3742] = poly[7] * poly[344];
    poly[3743] = poly[8] * poly[344];
    poly[3744] = poly[9] * poly[344];
    poly[3745] = poly[10] * poly[344];
    poly[3746] = poly[11] * poly[344];
    poly[3747] = poly[12] * poly[344];
    poly[3748] = poly[13] * poly[344];
    poly[3749] = poly[14] * poly[344];
}

fn f_polynomials75(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3750] = poly[15] * poly[344];
    poly[3751] = poly[16] * poly[344];
    poly[3752] = poly[17] * poly[344];
    poly[3753] = poly[1] * poly[345];
    poly[3754] = poly[2] * poly[345];
    poly[3755] = poly[3] * poly[345];
    poly[3756] = poly[4] * poly[345];
    poly[3757] = poly[5] * poly[345];
    poly[3758] = poly[6] * poly[345];
    poly[3759] = poly[7] * poly[345];
    poly[3760] = poly[8] * poly[345];
    poly[3761] = poly[9] * poly[345];
    poly[3762] = poly[10] * poly[345];
    poly[3763] = poly[11] * poly[345];
    poly[3764] = mono[1183] + mono[1184];
    poly[3765] = mono[1185] + mono[1186];
    poly[3766] = mono[1187] + mono[1188];
    poly[3767] = mono[1189] + mono[1190];
    poly[3768] = mono[1191] + mono[1192] + mono[1193] + mono[1194];
    poly[3769] = poly[12] * poly[345] - poly[3764];
    poly[3770] = poly[13] * poly[345] - poly[3765];
    poly[3771] = poly[14] * poly[345] - poly[3766];
    poly[3772] = poly[15] * poly[345] - poly[3767];
    poly[3773] = poly[16] * poly[345] - poly[3768];
    poly[3774] = poly[17] * poly[345];
    poly[3775] = poly[1] * poly[346];
    poly[3776] = poly[2] * poly[346];
    poly[3777] = poly[3] * poly[346];
    poly[3778] = poly[4] * poly[346];
    poly[3779] = poly[5] * poly[346];
    poly[3780] = poly[6] * poly[346];
    poly[3781] = poly[7] * poly[346];
    poly[3782] = poly[8] * poly[346];
    poly[3783] = poly[9] * poly[346];
    poly[3784] = poly[10] * poly[346];
    poly[3785] = poly[11] * poly[346];
    poly[3786] = poly[18] * poly[328] - poly[3620];
    poly[3787] = poly[18] * poly[329] - poly[3621];
    poly[3788] = poly[18] * poly[330] - poly[3622];
    poly[3789] = poly[18] * poly[331] - poly[3623];
    poly[3790] = poly[18] * poly[332] - poly[3624];
    poly[3791] = poly[12] * poly[346] - poly[3786];
    poly[3792] = poly[13] * poly[346] - poly[3787];
    poly[3793] = poly[14] * poly[346] - poly[3788];
    poly[3794] = poly[15] * poly[346] - poly[3789];
    poly[3795] = poly[16] * poly[346] - poly[3790];
    poly[3796] = poly[17] * poly[346];
    poly[3797] = poly[217] * poly[23];
    poly[3798] = poly[18] * poly[340] - poly[3653];
    poly[3799] = poly[18] * poly[341] - poly[3676];
}

fn f_polynomials76(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3800] = poly[18] * poly[342] - poly[3700];
    poly[3801] = poly[18] * poly[343] - poly[3731];
    poly[3802] = poly[18] * poly[344];
    poly[3803] = poly[18] * poly[345];
    poly[3804] = poly[1] * poly[347];
    poly[3805] = poly[2] * poly[347];
    poly[3806] = poly[3] * poly[347];
    poly[3807] = poly[4] * poly[347];
    poly[3808] = poly[5] * poly[347];
    poly[3809] = poly[6] * poly[347];
    poly[3810] = poly[7] * poly[347];
    poly[3811] = poly[8] * poly[347];
    poly[3812] = poly[9] * poly[347];
    poly[3813] = poly[10] * poly[347];
    poly[3814] = poly[11] * poly[347];
    poly[3815] = poly[19] * poly[328] - poly[3642];
    poly[3816] = poly[19] * poly[329] - poly[3643];
    poly[3817] = poly[19] * poly[330] - poly[3644];
    poly[3818] = poly[19] * poly[331] - poly[3645];
    poly[3819] = poly[19] * poly[332] - poly[3646];
    poly[3820] = poly[12] * poly[347] - poly[3815];
    poly[3821] = poly[13] * poly[347] - poly[3816];
    poly[3822] = poly[14] * poly[347] - poly[3817];
    poly[3823] = poly[15] * poly[347] - poly[3818];
    poly[3824] = poly[16] * poly[347] - poly[3819];
    poly[3825] = poly[17] * poly[347];
    poly[3826] = poly[19] * poly[339] - poly[3653];
    poly[3827] = poly[236] * poly[23];
    poly[3828] = poly[19] * poly[341] - poly[3677];
    poly[3829] = poly[19] * poly[342] - poly[3701];
    poly[3830] = poly[19] * poly[343] - poly[3732];
    poly[3831] = poly[19] * poly[344];
    poly[3832] = poly[19] * poly[345];
    poly[3833] = poly[18] * poly[347] - poly[3826];
    poly[3834] = poly[1] * poly[348];
    poly[3835] = poly[2] * poly[348];
    poly[3836] = poly[3] * poly[348];
    poly[3837] = poly[4] * poly[348];
    poly[3838] = poly[5] * poly[348];
    poly[3839] = poly[6] * poly[348];
    poly[3840] = poly[7] * poly[348];
    poly[3841] = poly[8] * poly[348];
    poly[3842] = poly[9] * poly[348];
    poly[3843] = poly[10] * poly[348];
    poly[3844] = poly[11] * poly[348];
    poly[3845] = poly[20] * poly[328] - poly[3665];
    poly[3846] = poly[20] * poly[329] - poly[3666];
    poly[3847] = poly[20] * poly[330] - poly[3667];
    poly[3848] = poly[20] * poly[331] - poly[3668];
    poly[3849] = poly[20] * poly[332] - poly[3669];
}

fn f_polynomials77(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3850] = poly[12] * poly[348] - poly[3845];
    poly[3851] = poly[13] * poly[348] - poly[3846];
    poly[3852] = poly[14] * poly[348] - poly[3847];
    poly[3853] = poly[15] * poly[348] - poly[3848];
    poly[3854] = poly[16] * poly[348] - poly[3849];
    poly[3855] = poly[17] * poly[348];
    poly[3856] = poly[20] * poly[339] - poly[3676];
    poly[3857] = poly[20] * poly[340] - poly[3677];
    poly[3858] = poly[257] * poly[23];
    poly[3859] = poly[20] * poly[342] - poly[3702];
    poly[3860] = poly[20] * poly[343] - poly[3733];
    poly[3861] = poly[20] * poly[344];
    poly[3862] = poly[20] * poly[345];
    poly[3863] = poly[18] * poly[348] - poly[3856];
    poly[3864] = poly[19] * poly[348] - poly[3857];
    poly[3865] = poly[1] * poly[349];
    poly[3866] = poly[2] * poly[349];
    poly[3867] = poly[3] * poly[349];
    poly[3868] = poly[4] * poly[349];
    poly[3869] = poly[5] * poly[349];
    poly[3870] = poly[6] * poly[349];
    poly[3871] = poly[7] * poly[349];
    poly[3872] = poly[8] * poly[349];
    poly[3873] = poly[9] * poly[349];
    poly[3874] = poly[10] * poly[349];
    poly[3875] = poly[11] * poly[349];
    poly[3876] = poly[21] * poly[328] - poly[3689];
    poly[3877] = poly[21] * poly[329] - poly[3690];
    poly[3878] = poly[21] * poly[330] - poly[3691];
    poly[3879] = poly[21] * poly[331] - poly[3692];
    poly[3880] = poly[21] * poly[332] - poly[3693];
    poly[3881] = poly[12] * poly[349] - poly[3876];
    poly[3882] = poly[13] * poly[349] - poly[3877];
    poly[3883] = poly[14] * poly[349] - poly[3878];
    poly[3884] = poly[15] * poly[349] - poly[3879];
    poly[3885] = poly[16] * poly[349] - poly[3880];
    poly[3886] = poly[17] * poly[349];
    poly[3887] = poly[21] * poly[339] - poly[3700];
    poly[3888] = poly[21] * poly[340] - poly[3701];
    poly[3889] = poly[21] * poly[341] - poly[3702];
    poly[3890] = poly[280] * poly[23];
    poly[3891] = poly[21] * poly[343] - poly[3734];
    poly[3892] = poly[21] * poly[344];
    poly[3893] = poly[21] * poly[345];
    poly[3894] = poly[18] * poly[349] - poly[3887];
    poly[3895] = poly[19] * poly[349] - poly[3888];
    poly[3896] = poly[20] * poly[349] - poly[3889];
    poly[3897] = poly[1] * poly[350];
    poly[3898] = poly[2] * poly[350];
    poly[3899] = poly[3] * poly[350];
}

fn f_polynomials78(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3900] = poly[4] * poly[350];
    poly[3901] = poly[5] * poly[350];
    poly[3902] = poly[6] * poly[350];
    poly[3903] = poly[23] * poly[290] - poly[3709];
    poly[3904] = poly[23] * poly[291] - poly[3710];
    poly[3905] = poly[23] * poly[292] - poly[3711];
    poly[3906] = poly[23] * poly[293] - poly[3712];
    poly[3907] = poly[7] * poly[350] - poly[3903];
    poly[3908] = poly[8] * poly[350] - poly[3904];
    poly[3909] = poly[9] * poly[350] - poly[3905];
    poly[3910] = poly[10] * poly[350] - poly[3906];
    poly[3911] = poly[11] * poly[350];
    poly[3912] = poly[22] * poly[328] - poly[3718];
    poly[3913] = poly[22] * poly[329] - poly[3719];
    poly[3914] = poly[22] * poly[330] - poly[3720];
    poly[3915] = poly[22] * poly[331] - poly[3721];
    poly[3916] = mono[1195]
        + mono[1196]
        + mono[1197]
        + mono[1198]
        + mono[1199]
        + mono[1200]
        + mono[1201]
        + mono[1202];
    poly[3917] = poly[22] * poly[332] - poly[3916] - poly[3723] - poly[3722];
    poly[3918] = poly[12] * poly[350] - poly[3912];
    poly[3919] = poly[13] * poly[350] - poly[3913];
    poly[3920] = poly[14] * poly[350] - poly[3914];
    poly[3921] = poly[15] * poly[350] - poly[3915];
    poly[3922] = poly[23] * poly[303] - poly[3916] - poly[3728] - poly[3722];
    poly[3923] = poly[16] * poly[350] - poly[3922] - poly[3917] - poly[3916];
    poly[3924] = poly[17] * poly[350];
    poly[3925] = poly[22] * poly[339] - poly[3731];
    poly[3926] = poly[22] * poly[340] - poly[3732];
    poly[3927] = poly[22] * poly[341] - poly[3733];
    poly[3928] = poly[22] * poly[342] - poly[3734];
    poly[3929] = poly[23] * poly[310];
    poly[3930] = poly[23] * poly[311];
    poly[3931] = poly[22] * poly[344];
    poly[3932] = poly[22] * poly[345];
    poly[3933] = poly[18] * poly[350] - poly[3925];
    poly[3934] = poly[19] * poly[350] - poly[3926];
    poly[3935] = poly[20] * poly[350] - poly[3927];
    poly[3936] = poly[21] * poly[350] - poly[3928];
    poly[3937] = poly[23] * poly[316] - poly[3735];
    poly[3938] = poly[1] * poly[351];
    poly[3939] = poly[2] * poly[351];
    poly[3940] = poly[3] * poly[351];
    poly[3941] = poly[4] * poly[351];
    poly[3942] = poly[5] * poly[351];
    poly[3943] = poly[6] * poly[351];
    poly[3944] = poly[7] * poly[351];
    poly[3945] = poly[8] * poly[351];
    poly[3946] = poly[9] * poly[351];
    poly[3947] = poly[10] * poly[351];
    poly[3948] = poly[11] * poly[351];
    poly[3949] = poly[12] * poly[351];
}

fn f_polynomials79(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[3950] = poly[13] * poly[351];
    poly[3951] = poly[14] * poly[351];
    poly[3952] = poly[15] * poly[351];
    poly[3953] = poly[16] * poly[351];
    poly[3954] = poly[17] * poly[351];
    poly[3955] = mono[1203] + mono[1204];
    poly[3956] = mono[1205] + mono[1206];
    poly[3957] = mono[1207] + mono[1208];
    poly[3958] = mono[1209] + mono[1210];
    poly[3959] = mono[1211] + mono[1212] + mono[1213] + mono[1214];
    poly[3960] = mono[1215] + mono[1216] + mono[1217] + mono[1218];
    poly[3961] = poly[18] * poly[351] - poly[3955];
    poly[3962] = poly[19] * poly[351] - poly[3956];
    poly[3963] = poly[20] * poly[351] - poly[3957];
    poly[3964] = poly[21] * poly[351] - poly[3958];
    poly[3965] = poly[22] * poly[351] - poly[3959];
    poly[3966] = poly[1] * poly[353];
    poly[3967] = poly[1] * poly[354];
    poly[3968] = poly[2] * poly[354];
    poly[3969] = poly[1] * poly[355];
    poly[3970] = poly[2] * poly[355];
    poly[3971] = poly[3] * poly[355];
    poly[3972] = poly[1] * poly[356];
    poly[3973] = poly[2] * poly[356];
    poly[3974] = poly[3] * poly[356];
    poly[3975] = poly[4] * poly[356];
    poly[3976] = poly[1] * poly[357];
    poly[3977] = poly[2] * poly[357];
    poly[3978] = poly[3] * poly[357];
    poly[3979] = poly[4] * poly[357];
    poly[3980] = poly[5] * poly[357];
    poly[3981] = poly[1] * poly[358];
    poly[3982] = poly[2] * poly[358];
    poly[3983] = poly[3] * poly[358];
    poly[3984] = poly[4] * poly[358];
    poly[3985] = poly[5] * poly[358];
    poly[3986] = poly[6] * poly[358];
    poly[3987] = poly[24] * poly[54];
    poly[3988] = poly[1] * poly[359];
    poly[3989] = poly[2] * poly[359];
    poly[3990] = poly[3] * poly[359];
    poly[3991] = poly[4] * poly[359];
    poly[3992] = poly[5] * poly[359];
    poly[3993] = poly[6] * poly[359];
    poly[3994] = poly[24] * poly[61];
    poly[3995] = poly[24] * poly[62];
    poly[3996] = poly[24] * poly[63];
    poly[3997] = poly[1] * poly[360];
    poly[3998] = poly[2] * poly[360];
    poly[3999] = poly[3] * poly[360];
}

fn f_polynomials80(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4000] = poly[4] * poly[360];
    poly[4001] = poly[5] * poly[360];
    poly[4002] = poly[6] * poly[360];
    poly[4003] = poly[24] * poly[70];
    poly[4004] = poly[24] * poly[71];
    poly[4005] = poly[24] * poly[72];
    poly[4006] = poly[24] * poly[73];
    poly[4007] = poly[24] * poly[74];
    poly[4008] = poly[1] * poly[361];
    poly[4009] = poly[2] * poly[361];
    poly[4010] = poly[3] * poly[361];
    poly[4011] = poly[4] * poly[361];
    poly[4012] = poly[5] * poly[361];
    poly[4013] = poly[6] * poly[361];
    poly[4014] = poly[24] * poly[81];
    poly[4015] = poly[24] * poly[82];
    poly[4016] = poly[24] * poly[83];
    poly[4017] = poly[24] * poly[84];
    poly[4018] = poly[24] * poly[85];
    poly[4019] = poly[24] * poly[86];
    poly[4020] = poly[24] * poly[87];
    poly[4021] = poly[1] * poly[362];
    poly[4022] = poly[2] * poly[362];
    poly[4023] = poly[3] * poly[362];
    poly[4024] = poly[4] * poly[362];
    poly[4025] = poly[5] * poly[362];
    poly[4026] = poly[6] * poly[362];
    poly[4027] = poly[11] * poly[358];
    poly[4028] = poly[11] * poly[359];
    poly[4029] = poly[11] * poly[360];
    poly[4030] = poly[11] * poly[361];
    poly[4031] = poly[1] * poly[363];
    poly[4032] = poly[2] * poly[363];
    poly[4033] = poly[3] * poly[363];
    poly[4034] = poly[4] * poly[363];
    poly[4035] = poly[5] * poly[363];
    poly[4036] = poly[6] * poly[363];
    poly[4037] = poly[24] * poly[104];
    poly[4038] = poly[24] * poly[105];
    poly[4039] = poly[24] * poly[106];
    poly[4040] = poly[24] * poly[107];
    poly[4041] = poly[11] * poly[363];
    poly[4042] = poly[24] * poly[109];
    poly[4043] = poly[1] * poly[364];
    poly[4044] = poly[2] * poly[364];
    poly[4045] = poly[3] * poly[364];
    poly[4046] = poly[4] * poly[364];
    poly[4047] = poly[5] * poly[364];
    poly[4048] = poly[6] * poly[364];
    poly[4049] = poly[24] * poly[116];
}

fn f_polynomials81(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4050] = poly[24] * poly[117];
    poly[4051] = poly[24] * poly[118];
    poly[4052] = poly[24] * poly[119];
    poly[4053] = poly[11] * poly[364];
    poly[4054] = poly[24] * poly[121];
    poly[4055] = poly[24] * poly[122];
    poly[4056] = poly[24] * poly[123];
    poly[4057] = poly[1] * poly[365];
    poly[4058] = poly[2] * poly[365];
    poly[4059] = poly[3] * poly[365];
    poly[4060] = poly[4] * poly[365];
    poly[4061] = poly[5] * poly[365];
    poly[4062] = poly[6] * poly[365];
    poly[4063] = poly[24] * poly[130];
    poly[4064] = poly[24] * poly[131];
    poly[4065] = poly[24] * poly[132];
    poly[4066] = poly[24] * poly[133];
    poly[4067] = poly[11] * poly[365];
    poly[4068] = poly[24] * poly[135];
    poly[4069] = poly[24] * poly[136];
    poly[4070] = poly[24] * poly[137];
    poly[4071] = poly[24] * poly[138];
    poly[4072] = poly[24] * poly[139];
    poly[4073] = poly[1] * poly[366];
    poly[4074] = poly[2] * poly[366];
    poly[4075] = poly[3] * poly[366];
    poly[4076] = poly[4] * poly[366];
    poly[4077] = poly[5] * poly[366];
    poly[4078] = poly[6] * poly[366];
    poly[4079] = poly[24] * poly[146];
    poly[4080] = poly[24] * poly[147];
    poly[4081] = poly[24] * poly[148];
    poly[4082] = poly[24] * poly[149];
    poly[4083] = poly[11] * poly[366];
    poly[4084] = poly[24] * poly[151];
    poly[4085] = poly[24] * poly[152];
    poly[4086] = poly[24] * poly[153];
    poly[4087] = poly[24] * poly[154];
    poly[4088] = poly[24] * poly[155];
    poly[4089] = poly[24] * poly[156];
    poly[4090] = poly[24] * poly[157];
    poly[4091] = poly[1] * poly[367];
    poly[4092] = poly[2] * poly[367];
    poly[4093] = poly[3] * poly[367];
    poly[4094] = poly[4] * poly[367];
    poly[4095] = poly[5] * poly[367];
    poly[4096] = poly[6] * poly[367];
    poly[4097] = poly[24] * poly[164];
    poly[4098] = poly[24] * poly[165];
    poly[4099] = poly[24] * poly[166];
}

fn f_polynomials82(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4100] = poly[24] * poly[167];
    poly[4101] = poly[24] * poly[168];
    poly[4102] = poly[24] * poly[169];
    poly[4103] = poly[24] * poly[170];
    poly[4104] = poly[24] * poly[171];
    poly[4105] = poly[11] * poly[367];
    poly[4106] = poly[24] * poly[173];
    poly[4107] = poly[24] * poly[174];
    poly[4108] = poly[24] * poly[175];
    poly[4109] = poly[24] * poly[176];
    poly[4110] = poly[24] * poly[177];
    poly[4111] = poly[24] * poly[178];
    poly[4112] = poly[24] * poly[179];
    poly[4113] = poly[24] * poly[180];
    poly[4114] = poly[24] * poly[181];
    poly[4115] = poly[24] * poly[182];
    poly[4116] = poly[24] * poly[183];
    poly[4117] = poly[1] * poly[368];
    poly[4118] = poly[2] * poly[368];
    poly[4119] = poly[3] * poly[368];
    poly[4120] = poly[4] * poly[368];
    poly[4121] = poly[5] * poly[368];
    poly[4122] = poly[6] * poly[368];
    poly[4123] = poly[17] * poly[358];
    poly[4124] = poly[17] * poly[359];
    poly[4125] = poly[17] * poly[360];
    poly[4126] = poly[17] * poly[361];
    poly[4127] = poly[11] * poly[368];
    poly[4128] = poly[17] * poly[363];
    poly[4129] = poly[17] * poly[364];
    poly[4130] = poly[17] * poly[365];
    poly[4131] = poly[17] * poly[366];
    poly[4132] = poly[17] * poly[367];
    poly[4133] = poly[1] * poly[369];
    poly[4134] = poly[2] * poly[369];
    poly[4135] = poly[3] * poly[369];
    poly[4136] = poly[4] * poly[369];
    poly[4137] = poly[5] * poly[369];
    poly[4138] = poly[6] * poly[369];
    poly[4139] = poly[24] * poly[206];
    poly[4140] = poly[24] * poly[207];
    poly[4141] = poly[24] * poly[208];
    poly[4142] = poly[24] * poly[209];
    poly[4143] = poly[11] * poly[369];
    poly[4144] = poly[24] * poly[211];
    poly[4145] = poly[24] * poly[212];
    poly[4146] = poly[24] * poly[213];
    poly[4147] = poly[24] * poly[214];
    poly[4148] = poly[24] * poly[215];
    poly[4149] = poly[17] * poly[369];
}

fn f_polynomials83(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4150] = poly[24] * poly[217];
    poly[4151] = poly[1] * poly[370];
    poly[4152] = poly[2] * poly[370];
    poly[4153] = poly[3] * poly[370];
    poly[4154] = poly[4] * poly[370];
    poly[4155] = poly[5] * poly[370];
    poly[4156] = poly[6] * poly[370];
    poly[4157] = poly[24] * poly[224];
    poly[4158] = poly[24] * poly[225];
    poly[4159] = poly[24] * poly[226];
    poly[4160] = poly[24] * poly[227];
    poly[4161] = poly[11] * poly[370];
    poly[4162] = poly[24] * poly[229];
    poly[4163] = poly[24] * poly[230];
    poly[4164] = poly[24] * poly[231];
    poly[4165] = poly[24] * poly[232];
    poly[4166] = poly[24] * poly[233];
    poly[4167] = poly[17] * poly[370];
    poly[4168] = poly[24] * poly[235];
    poly[4169] = poly[24] * poly[236];
    poly[4170] = poly[24] * poly[237];
    poly[4171] = poly[1] * poly[371];
    poly[4172] = poly[2] * poly[371];
    poly[4173] = poly[3] * poly[371];
    poly[4174] = poly[4] * poly[371];
    poly[4175] = poly[5] * poly[371];
    poly[4176] = poly[6] * poly[371];
    poly[4177] = poly[24] * poly[244];
    poly[4178] = poly[24] * poly[245];
    poly[4179] = poly[24] * poly[246];
    poly[4180] = poly[24] * poly[247];
    poly[4181] = poly[11] * poly[371];
    poly[4182] = poly[24] * poly[249];
    poly[4183] = poly[24] * poly[250];
    poly[4184] = poly[24] * poly[251];
    poly[4185] = poly[24] * poly[252];
    poly[4186] = poly[24] * poly[253];
    poly[4187] = poly[17] * poly[371];
    poly[4188] = poly[24] * poly[255];
    poly[4189] = poly[24] * poly[256];
    poly[4190] = poly[24] * poly[257];
    poly[4191] = poly[24] * poly[258];
    poly[4192] = poly[24] * poly[259];
    poly[4193] = poly[1] * poly[372];
    poly[4194] = poly[2] * poly[372];
    poly[4195] = poly[3] * poly[372];
    poly[4196] = poly[4] * poly[372];
    poly[4197] = poly[5] * poly[372];
    poly[4198] = poly[6] * poly[372];
    poly[4199] = poly[24] * poly[266];
}

fn f_polynomials84(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4200] = poly[24] * poly[267];
    poly[4201] = poly[24] * poly[268];
    poly[4202] = poly[24] * poly[269];
    poly[4203] = poly[11] * poly[372];
    poly[4204] = poly[24] * poly[271];
    poly[4205] = poly[24] * poly[272];
    poly[4206] = poly[24] * poly[273];
    poly[4207] = poly[24] * poly[274];
    poly[4208] = poly[24] * poly[275];
    poly[4209] = poly[17] * poly[372];
    poly[4210] = poly[24] * poly[277];
    poly[4211] = poly[24] * poly[278];
    poly[4212] = poly[24] * poly[279];
    poly[4213] = poly[24] * poly[280];
    poly[4214] = poly[24] * poly[281];
    poly[4215] = poly[24] * poly[282];
    poly[4216] = poly[24] * poly[283];
    poly[4217] = poly[1] * poly[373];
    poly[4218] = poly[2] * poly[373];
    poly[4219] = poly[3] * poly[373];
    poly[4220] = poly[4] * poly[373];
    poly[4221] = poly[5] * poly[373];
    poly[4222] = poly[6] * poly[373];
    poly[4223] = poly[24] * poly[290];
    poly[4224] = poly[24] * poly[291];
    poly[4225] = poly[24] * poly[292];
    poly[4226] = poly[24] * poly[293];
    poly[4227] = poly[24] * poly[294];
    poly[4228] = poly[24] * poly[295];
    poly[4229] = poly[24] * poly[296];
    poly[4230] = poly[24] * poly[297];
    poly[4231] = poly[11] * poly[373];
    poly[4232] = poly[24] * poly[299];
    poly[4233] = poly[24] * poly[300];
    poly[4234] = poly[24] * poly[301];
    poly[4235] = poly[24] * poly[302];
    poly[4236] = poly[24] * poly[303];
    poly[4237] = poly[24] * poly[304];
    poly[4238] = poly[17] * poly[373];
    poly[4239] = poly[24] * poly[306];
    poly[4240] = poly[24] * poly[307];
    poly[4241] = poly[24] * poly[308];
    poly[4242] = poly[24] * poly[309];
    poly[4243] = poly[24] * poly[310];
    poly[4244] = poly[24] * poly[311];
    poly[4245] = poly[24] * poly[312];
    poly[4246] = poly[24] * poly[313];
    poly[4247] = poly[24] * poly[314];
    poly[4248] = poly[24] * poly[315];
    poly[4249] = poly[24] * poly[316];
}

fn f_polynomials85(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4250] = poly[1] * poly[374];
    poly[4251] = poly[2] * poly[374];
    poly[4252] = poly[3] * poly[374];
    poly[4253] = poly[4] * poly[374];
    poly[4254] = poly[5] * poly[374];
    poly[4255] = poly[6] * poly[374];
    poly[4256] = poly[24] * poly[323];
    poly[4257] = poly[24] * poly[324];
    poly[4258] = poly[24] * poly[325];
    poly[4259] = poly[24] * poly[326];
    poly[4260] = poly[11] * poly[374];
    poly[4261] = poly[24] * poly[328];
    poly[4262] = poly[24] * poly[329];
    poly[4263] = poly[24] * poly[330];
    poly[4264] = poly[24] * poly[331];
    poly[4265] = poly[24] * poly[332];
    poly[4266] = poly[24] * poly[333];
    poly[4267] = poly[24] * poly[334];
    poly[4268] = poly[24] * poly[335];
    poly[4269] = poly[24] * poly[336];
    poly[4270] = poly[24] * poly[337];
    poly[4271] = poly[17] * poly[374];
    poly[4272] = poly[24] * poly[339];
    poly[4273] = poly[24] * poly[340];
    poly[4274] = poly[24] * poly[341];
    poly[4275] = poly[24] * poly[342];
    poly[4276] = poly[24] * poly[343];
    poly[4277] = poly[24] * poly[344];
    poly[4278] = poly[24] * poly[345];
    poly[4279] = poly[24] * poly[346];
    poly[4280] = poly[24] * poly[347];
    poly[4281] = poly[24] * poly[348];
    poly[4282] = poly[24] * poly[349];
    poly[4283] = poly[24] * poly[350];
    poly[4284] = poly[24] * poly[351];
    poly[4285] = poly[1] * poly[376];
    poly[4286] = poly[1] * poly[377];
    poly[4287] = poly[2] * poly[377];
    poly[4288] = poly[1] * poly[378];
    poly[4289] = poly[2] * poly[378];
    poly[4290] = poly[3] * poly[378];
    poly[4291] = poly[1] * poly[379];
    poly[4292] = poly[2] * poly[379];
    poly[4293] = poly[3] * poly[379];
    poly[4294] = poly[4] * poly[379];
    poly[4295] = poly[1] * poly[380];
    poly[4296] = poly[2] * poly[380];
    poly[4297] = poly[3] * poly[380];
    poly[4298] = poly[4] * poly[380];
    poly[4299] = poly[5] * poly[380];
}

fn f_polynomials86(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4300] = poly[1] * poly[381];
    poly[4301] = poly[2] * poly[381];
    poly[4302] = poly[3] * poly[381];
    poly[4303] = poly[4] * poly[381];
    poly[4304] = poly[5] * poly[381];
    poly[4305] = poly[6] * poly[381];
    poly[4306] = poly[54] * poly[25];
    poly[4307] = poly[1] * poly[382];
    poly[4308] = poly[2] * poly[382];
    poly[4309] = poly[3] * poly[382];
    poly[4310] = poly[4] * poly[382];
    poly[4311] = poly[5] * poly[382];
    poly[4312] = poly[6] * poly[382];
    poly[4313] = poly[25] * poly[61];
    poly[4314] = poly[62] * poly[25];
    poly[4315] = poly[25] * poly[63];
    poly[4316] = poly[1] * poly[383];
    poly[4317] = poly[2] * poly[383];
    poly[4318] = poly[3] * poly[383];
    poly[4319] = poly[4] * poly[383];
    poly[4320] = poly[5] * poly[383];
    poly[4321] = poly[6] * poly[383];
    poly[4322] = poly[25] * poly[70];
    poly[4323] = poly[25] * poly[71];
    poly[4324] = poly[72] * poly[25];
    poly[4325] = poly[25] * poly[73];
    poly[4326] = poly[25] * poly[74];
    poly[4327] = poly[1] * poly[384];
    poly[4328] = poly[2] * poly[384];
    poly[4329] = poly[3] * poly[384];
    poly[4330] = poly[4] * poly[384];
    poly[4331] = poly[5] * poly[384];
    poly[4332] = poly[6] * poly[384];
    poly[4333] = poly[25] * poly[81];
    poly[4334] = poly[25] * poly[82];
    poly[4335] = poly[25] * poly[83];
    poly[4336] = poly[84] * poly[25];
    poly[4337] = poly[25] * poly[85];
    poly[4338] = poly[25] * poly[86];
    poly[4339] = poly[25] * poly[87];
    poly[4340] = poly[1] * poly[385];
    poly[4341] = poly[2] * poly[385];
    poly[4342] = poly[3] * poly[385];
    poly[4343] = poly[4] * poly[385];
    poly[4344] = poly[5] * poly[385];
    poly[4345] = poly[6] * poly[385];
    poly[4346] = poly[11] * poly[381];
    poly[4347] = poly[11] * poly[382];
    poly[4348] = poly[11] * poly[383];
    poly[4349] = poly[11] * poly[384];
}

fn f_polynomials87(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4350] = poly[1] * poly[386];
    poly[4351] = poly[2] * poly[386];
    poly[4352] = poly[3] * poly[386];
    poly[4353] = poly[4] * poly[386];
    poly[4354] = poly[5] * poly[386];
    poly[4355] = poly[6] * poly[386];
    poly[4356] = poly[7] * poly[386];
    poly[4357] = poly[8] * poly[386];
    poly[4358] = poly[9] * poly[386];
    poly[4359] = poly[10] * poly[386];
    poly[4360] = poly[11] * poly[386];
    poly[4361] = poly[109] * poly[25];
    poly[4362] = poly[1] * poly[387];
    poly[4363] = poly[2] * poly[387];
    poly[4364] = poly[3] * poly[387];
    poly[4365] = poly[4] * poly[387];
    poly[4366] = poly[5] * poly[387];
    poly[4367] = poly[6] * poly[387];
    poly[4368] = poly[7] * poly[387];
    poly[4369] = poly[8] * poly[387];
    poly[4370] = poly[9] * poly[387];
    poly[4371] = poly[10] * poly[387];
    poly[4372] = poly[11] * poly[387];
    poly[4373] = poly[25] * poly[121];
    poly[4374] = poly[122] * poly[25];
    poly[4375] = poly[25] * poly[123];
    poly[4376] = poly[1] * poly[388];
    poly[4377] = poly[2] * poly[388];
    poly[4378] = poly[3] * poly[388];
    poly[4379] = poly[4] * poly[388];
    poly[4380] = poly[5] * poly[388];
    poly[4381] = poly[6] * poly[388];
    poly[4382] = poly[7] * poly[388];
    poly[4383] = poly[8] * poly[388];
    poly[4384] = poly[9] * poly[388];
    poly[4385] = poly[10] * poly[388];
    poly[4386] = poly[11] * poly[388];
    poly[4387] = poly[25] * poly[135];
    poly[4388] = poly[25] * poly[136];
    poly[4389] = poly[137] * poly[25];
    poly[4390] = poly[25] * poly[138];
    poly[4391] = poly[25] * poly[139];
    poly[4392] = poly[1] * poly[389];
    poly[4393] = poly[2] * poly[389];
    poly[4394] = poly[3] * poly[389];
    poly[4395] = poly[4] * poly[389];
    poly[4396] = poly[5] * poly[389];
    poly[4397] = poly[6] * poly[389];
    poly[4398] = poly[7] * poly[389];
    poly[4399] = poly[8] * poly[389];
}

fn f_polynomials88(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4400] = poly[9] * poly[389];
    poly[4401] = poly[10] * poly[389];
    poly[4402] = poly[11] * poly[389];
    poly[4403] = poly[25] * poly[151];
    poly[4404] = poly[25] * poly[152];
    poly[4405] = poly[25] * poly[153];
    poly[4406] = poly[154] * poly[25];
    poly[4407] = poly[25] * poly[155];
    poly[4408] = poly[25] * poly[156];
    poly[4409] = poly[25] * poly[157];
    poly[4410] = poly[1] * poly[390];
    poly[4411] = poly[2] * poly[390];
    poly[4412] = poly[3] * poly[390];
    poly[4413] = poly[4] * poly[390];
    poly[4414] = poly[5] * poly[390];
    poly[4415] = poly[6] * poly[390];
    poly[4416] = poly[25] * poly[164];
    poly[4417] = poly[25] * poly[165];
    poly[4418] = poly[25] * poly[166];
    poly[4419] = poly[25] * poly[167];
    poly[4420] = poly[25] * poly[168];
    poly[4421] = poly[25] * poly[169];
    poly[4422] = poly[25] * poly[170];
    poly[4423] = poly[25] * poly[171];
    poly[4424] = poly[11] * poly[390];
    poly[4425] = poly[25] * poly[173];
    poly[4426] = poly[25] * poly[174];
    poly[4427] = poly[25] * poly[175];
    poly[4428] = poly[25] * poly[176];
    poly[4429] = poly[25] * poly[177];
    poly[4430] = poly[25] * poly[178];
    poly[4431] = poly[25] * poly[179];
    poly[4432] = poly[25] * poly[180];
    poly[4433] = poly[25] * poly[181];
    poly[4434] = poly[25] * poly[182];
    poly[4435] = poly[25] * poly[183];
    poly[4436] = poly[1] * poly[391];
    poly[4437] = poly[2] * poly[391];
    poly[4438] = poly[3] * poly[391];
    poly[4439] = poly[4] * poly[391];
    poly[4440] = poly[5] * poly[391];
    poly[4441] = poly[6] * poly[391];
    poly[4442] = poly[17] * poly[381];
    poly[4443] = poly[17] * poly[382];
    poly[4444] = poly[17] * poly[383];
    poly[4445] = poly[17] * poly[384];
    poly[4446] = poly[11] * poly[391];
    poly[4447] = poly[17] * poly[386];
    poly[4448] = poly[17] * poly[387];
    poly[4449] = poly[17] * poly[388];
}

fn f_polynomials89(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4450] = poly[17] * poly[389];
    poly[4451] = poly[17] * poly[390];
    poly[4452] = poly[1] * poly[392];
    poly[4453] = poly[2] * poly[392];
    poly[4454] = poly[3] * poly[392];
    poly[4455] = poly[4] * poly[392];
    poly[4456] = poly[5] * poly[392];
    poly[4457] = poly[6] * poly[392];
    poly[4458] = poly[7] * poly[392];
    poly[4459] = poly[8] * poly[392];
    poly[4460] = poly[9] * poly[392];
    poly[4461] = poly[10] * poly[392];
    poly[4462] = poly[11] * poly[392];
    poly[4463] = poly[12] * poly[392];
    poly[4464] = poly[13] * poly[392];
    poly[4465] = poly[14] * poly[392];
    poly[4466] = poly[15] * poly[392];
    poly[4467] = poly[16] * poly[392];
    poly[4468] = poly[17] * poly[392];
    poly[4469] = poly[217] * poly[25];
    poly[4470] = poly[1] * poly[393];
    poly[4471] = poly[2] * poly[393];
    poly[4472] = poly[3] * poly[393];
    poly[4473] = poly[4] * poly[393];
    poly[4474] = poly[5] * poly[393];
    poly[4475] = poly[6] * poly[393];
    poly[4476] = poly[7] * poly[393];
    poly[4477] = poly[8] * poly[393];
    poly[4478] = poly[9] * poly[393];
    poly[4479] = poly[10] * poly[393];
    poly[4480] = poly[11] * poly[393];
    poly[4481] = poly[12] * poly[393];
    poly[4482] = poly[13] * poly[393];
    poly[4483] = poly[14] * poly[393];
    poly[4484] = poly[15] * poly[393];
    poly[4485] = poly[16] * poly[393];
    poly[4486] = poly[17] * poly[393];
    poly[4487] = poly[25] * poly[235];
    poly[4488] = poly[236] * poly[25];
    poly[4489] = poly[25] * poly[237];
    poly[4490] = poly[1] * poly[394];
    poly[4491] = poly[2] * poly[394];
    poly[4492] = poly[3] * poly[394];
    poly[4493] = poly[4] * poly[394];
    poly[4494] = poly[5] * poly[394];
    poly[4495] = poly[6] * poly[394];
    poly[4496] = poly[7] * poly[394];
    poly[4497] = poly[8] * poly[394];
    poly[4498] = poly[9] * poly[394];
    poly[4499] = poly[10] * poly[394];
}

fn f_polynomials90(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4500] = poly[11] * poly[394];
    poly[4501] = poly[12] * poly[394];
    poly[4502] = poly[13] * poly[394];
    poly[4503] = poly[14] * poly[394];
    poly[4504] = poly[15] * poly[394];
    poly[4505] = poly[16] * poly[394];
    poly[4506] = poly[17] * poly[394];
    poly[4507] = poly[25] * poly[255];
    poly[4508] = poly[25] * poly[256];
    poly[4509] = poly[257] * poly[25];
    poly[4510] = poly[25] * poly[258];
    poly[4511] = poly[25] * poly[259];
    poly[4512] = poly[1] * poly[395];
    poly[4513] = poly[2] * poly[395];
    poly[4514] = poly[3] * poly[395];
    poly[4515] = poly[4] * poly[395];
    poly[4516] = poly[5] * poly[395];
    poly[4517] = poly[6] * poly[395];
    poly[4518] = poly[7] * poly[395];
    poly[4519] = poly[8] * poly[395];
    poly[4520] = poly[9] * poly[395];
    poly[4521] = poly[10] * poly[395];
    poly[4522] = poly[11] * poly[395];
    poly[4523] = poly[12] * poly[395];
    poly[4524] = poly[13] * poly[395];
    poly[4525] = poly[14] * poly[395];
    poly[4526] = poly[15] * poly[395];
    poly[4527] = poly[16] * poly[395];
    poly[4528] = poly[17] * poly[395];
    poly[4529] = poly[25] * poly[277];
    poly[4530] = poly[25] * poly[278];
    poly[4531] = poly[25] * poly[279];
    poly[4532] = poly[280] * poly[25];
    poly[4533] = poly[25] * poly[281];
    poly[4534] = poly[25] * poly[282];
    poly[4535] = poly[25] * poly[283];
    poly[4536] = poly[1] * poly[396];
    poly[4537] = poly[2] * poly[396];
    poly[4538] = poly[3] * poly[396];
    poly[4539] = poly[4] * poly[396];
    poly[4540] = poly[5] * poly[396];
    poly[4541] = poly[6] * poly[396];
    poly[4542] = poly[25] * poly[290];
    poly[4543] = poly[25] * poly[291];
    poly[4544] = poly[25] * poly[292];
    poly[4545] = poly[25] * poly[293];
    poly[4546] = poly[25] * poly[294];
    poly[4547] = poly[25] * poly[295];
    poly[4548] = poly[25] * poly[296];
    poly[4549] = poly[25] * poly[297];
}

fn f_polynomials91(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4550] = poly[11] * poly[396];
    poly[4551] = poly[12] * poly[396];
    poly[4552] = poly[13] * poly[396];
    poly[4553] = poly[14] * poly[396];
    poly[4554] = poly[15] * poly[396];
    poly[4555] = poly[25] * poly[303];
    poly[4556] = poly[25] * poly[304];
    poly[4557] = poly[17] * poly[396];
    poly[4558] = poly[25] * poly[306];
    poly[4559] = poly[25] * poly[307];
    poly[4560] = poly[25] * poly[308];
    poly[4561] = poly[25] * poly[309];
    poly[4562] = poly[25] * poly[310];
    poly[4563] = poly[25] * poly[311];
    poly[4564] = poly[25] * poly[312];
    poly[4565] = poly[25] * poly[313];
    poly[4566] = poly[25] * poly[314];
    poly[4567] = poly[25] * poly[315];
    poly[4568] = poly[25] * poly[316];
    poly[4569] = poly[1] * poly[397];
    poly[4570] = poly[2] * poly[397];
    poly[4571] = poly[3] * poly[397];
    poly[4572] = poly[4] * poly[397];
    poly[4573] = poly[5] * poly[397];
    poly[4574] = poly[6] * poly[397];
    poly[4575] = poly[7] * poly[397];
    poly[4576] = poly[8] * poly[397];
    poly[4577] = poly[9] * poly[397];
    poly[4578] = poly[10] * poly[397];
    poly[4579] = poly[11] * poly[397];
    poly[4580] = poly[25] * poly[328];
    poly[4581] = poly[25] * poly[329];
    poly[4582] = poly[25] * poly[330];
    poly[4583] = poly[25] * poly[331];
    poly[4584] = poly[25] * poly[332];
    poly[4585] = poly[25] * poly[333];
    poly[4586] = poly[25] * poly[334];
    poly[4587] = poly[25] * poly[335];
    poly[4588] = poly[25] * poly[336];
    poly[4589] = poly[25] * poly[337];
    poly[4590] = poly[17] * poly[397];
    poly[4591] = poly[25] * poly[339];
    poly[4592] = poly[25] * poly[340];
    poly[4593] = poly[25] * poly[341];
    poly[4594] = poly[25] * poly[342];
    poly[4595] = poly[25] * poly[343];
    poly[4596] = poly[25] * poly[344];
    poly[4597] = poly[25] * poly[345];
    poly[4598] = poly[25] * poly[346];
    poly[4599] = poly[25] * poly[347];
}

fn f_polynomials92(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4600] = poly[25] * poly[348];
    poly[4601] = poly[25] * poly[349];
    poly[4602] = poly[25] * poly[350];
    poly[4603] = poly[25] * poly[351];
    poly[4604] = poly[1] * poly[398];
    poly[4605] = poly[2] * poly[398];
    poly[4606] = poly[3] * poly[398];
    poly[4607] = poly[4] * poly[398];
    poly[4608] = poly[5] * poly[398];
    poly[4609] = poly[6] * poly[398];
    poly[4610] = poly[24] * poly[381];
    poly[4611] = poly[24] * poly[382];
    poly[4612] = poly[24] * poly[383];
    poly[4613] = poly[24] * poly[384];
    poly[4614] = poly[11] * poly[398];
    poly[4615] = poly[24] * poly[386];
    poly[4616] = poly[24] * poly[387];
    poly[4617] = poly[24] * poly[388];
    poly[4618] = poly[24] * poly[389];
    poly[4619] = poly[24] * poly[390];
    poly[4620] = poly[17] * poly[398];
    poly[4621] = poly[24] * poly[392];
    poly[4622] = poly[24] * poly[393];
    poly[4623] = poly[24] * poly[394];
    poly[4624] = poly[24] * poly[395];
    poly[4625] = poly[24] * poly[396];
    poly[4626] = poly[24] * poly[397];
    poly[4627] = poly[1] * poly[399];
    poly[4628] = poly[2] * poly[399];
    poly[4629] = poly[3] * poly[399];
    poly[4630] = poly[4] * poly[399];
    poly[4631] = poly[5] * poly[399];
    poly[4632] = poly[6] * poly[399];
    poly[4633] = poly[399] * poly[7];
    poly[4634] = poly[399] * poly[8];
    poly[4635] = poly[399] * poly[9];
    poly[4636] = poly[399] * poly[10];
    poly[4637] = poly[11] * poly[399];
    poly[4638] = poly[399] * poly[12];
    poly[4639] = poly[399] * poly[13];
    poly[4640] = poly[399] * poly[14];
    poly[4641] = poly[399] * poly[15];
    poly[4642] = poly[399] * poly[16];
    poly[4643] = poly[17] * poly[399];
    poly[4644] = poly[399] * poly[18];
    poly[4645] = poly[399] * poly[19];
    poly[4646] = poly[399] * poly[20];
    poly[4647] = poly[399] * poly[21];
    poly[4648] = poly[399] * poly[22];
    poly[4649] = poly[399] * poly[23];
}

fn f_polynomials93(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4650] = poly[24] * poly[399];
    poly[4651] = poly[1] * poly[401];
    poly[4652] = poly[1] * poly[402];
    poly[4653] = poly[2] * poly[402];
    poly[4654] = poly[1] * poly[403];
    poly[4655] = poly[2] * poly[403];
    poly[4656] = poly[3] * poly[403];
    poly[4657] = poly[1] * poly[404];
    poly[4658] = poly[2] * poly[404];
    poly[4659] = poly[3] * poly[404];
    poly[4660] = poly[4] * poly[404];
    poly[4661] = poly[1] * poly[405];
    poly[4662] = poly[2] * poly[405];
    poly[4663] = poly[3] * poly[405];
    poly[4664] = poly[4] * poly[405];
    poly[4665] = poly[5] * poly[405];
    poly[4666] = poly[1] * poly[406];
    poly[4667] = poly[2] * poly[406];
    poly[4668] = poly[3] * poly[406];
    poly[4669] = poly[4] * poly[406];
    poly[4670] = poly[5] * poly[406];
    poly[4671] = poly[6] * poly[406];
    poly[4672] = poly[54] * poly[26];
    poly[4673] = poly[1] * poly[407];
    poly[4674] = poly[2] * poly[407];
    poly[4675] = poly[3] * poly[407];
    poly[4676] = poly[4] * poly[407];
    poly[4677] = poly[5] * poly[407];
    poly[4678] = poly[6] * poly[407];
    poly[4679] = poly[26] * poly[61];
    poly[4680] = poly[62] * poly[26];
    poly[4681] = poly[26] * poly[63];
    poly[4682] = poly[1] * poly[408];
    poly[4683] = poly[2] * poly[408];
    poly[4684] = poly[3] * poly[408];
    poly[4685] = poly[4] * poly[408];
    poly[4686] = poly[5] * poly[408];
    poly[4687] = poly[6] * poly[408];
    poly[4688] = poly[26] * poly[70];
    poly[4689] = poly[26] * poly[71];
    poly[4690] = poly[72] * poly[26];
    poly[4691] = poly[26] * poly[73];
    poly[4692] = poly[26] * poly[74];
    poly[4693] = poly[1] * poly[409];
    poly[4694] = poly[2] * poly[409];
    poly[4695] = poly[3] * poly[409];
    poly[4696] = poly[4] * poly[409];
    poly[4697] = poly[5] * poly[409];
    poly[4698] = poly[6] * poly[409];
    poly[4699] = poly[26] * poly[81];
}

fn f_polynomials94(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4700] = poly[26] * poly[82];
    poly[4701] = poly[26] * poly[83];
    poly[4702] = poly[84] * poly[26];
    poly[4703] = poly[26] * poly[85];
    poly[4704] = poly[26] * poly[86];
    poly[4705] = poly[26] * poly[87];
    poly[4706] = poly[1] * poly[410];
    poly[4707] = poly[2] * poly[410];
    poly[4708] = poly[3] * poly[410];
    poly[4709] = poly[4] * poly[410];
    poly[4710] = poly[5] * poly[410];
    poly[4711] = poly[6] * poly[410];
    poly[4712] = poly[11] * poly[406];
    poly[4713] = poly[11] * poly[407];
    poly[4714] = poly[11] * poly[408];
    poly[4715] = poly[11] * poly[409];
    poly[4716] = poly[1] * poly[411];
    poly[4717] = poly[2] * poly[411];
    poly[4718] = poly[3] * poly[411];
    poly[4719] = poly[4] * poly[411];
    poly[4720] = poly[5] * poly[411];
    poly[4721] = poly[6] * poly[411];
    poly[4722] = poly[7] * poly[411];
    poly[4723] = poly[8] * poly[411];
    poly[4724] = poly[9] * poly[411];
    poly[4725] = poly[10] * poly[411];
    poly[4726] = poly[11] * poly[411];
    poly[4727] = poly[109] * poly[26];
    poly[4728] = poly[1] * poly[412];
    poly[4729] = poly[2] * poly[412];
    poly[4730] = poly[3] * poly[412];
    poly[4731] = poly[4] * poly[412];
    poly[4732] = poly[5] * poly[412];
    poly[4733] = poly[6] * poly[412];
    poly[4734] = poly[7] * poly[412];
    poly[4735] = poly[8] * poly[412];
    poly[4736] = poly[9] * poly[412];
    poly[4737] = poly[10] * poly[412];
    poly[4738] = poly[11] * poly[412];
    poly[4739] = poly[26] * poly[121];
    poly[4740] = poly[122] * poly[26];
    poly[4741] = poly[26] * poly[123];
    poly[4742] = poly[1] * poly[413];
    poly[4743] = poly[2] * poly[413];
    poly[4744] = poly[3] * poly[413];
    poly[4745] = poly[4] * poly[413];
    poly[4746] = poly[5] * poly[413];
    poly[4747] = poly[6] * poly[413];
    poly[4748] = poly[7] * poly[413];
    poly[4749] = poly[8] * poly[413];
}

fn f_polynomials95(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4750] = poly[9] * poly[413];
    poly[4751] = poly[10] * poly[413];
    poly[4752] = poly[11] * poly[413];
    poly[4753] = poly[26] * poly[135];
    poly[4754] = poly[26] * poly[136];
    poly[4755] = poly[137] * poly[26];
    poly[4756] = poly[26] * poly[138];
    poly[4757] = poly[26] * poly[139];
    poly[4758] = poly[1] * poly[414];
    poly[4759] = poly[2] * poly[414];
    poly[4760] = poly[3] * poly[414];
    poly[4761] = poly[4] * poly[414];
    poly[4762] = poly[5] * poly[414];
    poly[4763] = poly[6] * poly[414];
    poly[4764] = poly[7] * poly[414];
    poly[4765] = poly[8] * poly[414];
    poly[4766] = poly[9] * poly[414];
    poly[4767] = poly[10] * poly[414];
    poly[4768] = poly[11] * poly[414];
    poly[4769] = poly[26] * poly[151];
    poly[4770] = poly[26] * poly[152];
    poly[4771] = poly[26] * poly[153];
    poly[4772] = poly[154] * poly[26];
    poly[4773] = poly[26] * poly[155];
    poly[4774] = poly[26] * poly[156];
    poly[4775] = poly[26] * poly[157];
    poly[4776] = poly[1] * poly[415];
    poly[4777] = poly[2] * poly[415];
    poly[4778] = poly[3] * poly[415];
    poly[4779] = poly[4] * poly[415];
    poly[4780] = poly[5] * poly[415];
    poly[4781] = poly[6] * poly[415];
    poly[4782] = poly[26] * poly[164];
    poly[4783] = poly[26] * poly[165];
    poly[4784] = poly[26] * poly[166];
    poly[4785] = poly[26] * poly[167];
    poly[4786] = poly[26] * poly[168];
    poly[4787] = poly[26] * poly[169];
    poly[4788] = poly[26] * poly[170];
    poly[4789] = poly[26] * poly[171];
    poly[4790] = poly[11] * poly[415];
    poly[4791] = poly[26] * poly[173];
    poly[4792] = poly[26] * poly[174];
    poly[4793] = poly[26] * poly[175];
    poly[4794] = poly[26] * poly[176];
    poly[4795] = poly[26] * poly[177];
    poly[4796] = poly[26] * poly[178];
    poly[4797] = poly[26] * poly[179];
    poly[4798] = poly[26] * poly[180];
    poly[4799] = poly[26] * poly[181];
}

fn f_polynomials96(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4800] = poly[26] * poly[182];
    poly[4801] = poly[26] * poly[183];
    poly[4802] = poly[1] * poly[416];
    poly[4803] = poly[2] * poly[416];
    poly[4804] = poly[3] * poly[416];
    poly[4805] = poly[4] * poly[416];
    poly[4806] = poly[5] * poly[416];
    poly[4807] = poly[6] * poly[416];
    poly[4808] = poly[17] * poly[406];
    poly[4809] = poly[17] * poly[407];
    poly[4810] = poly[17] * poly[408];
    poly[4811] = poly[17] * poly[409];
    poly[4812] = poly[11] * poly[416];
    poly[4813] = poly[17] * poly[411];
    poly[4814] = poly[17] * poly[412];
    poly[4815] = poly[17] * poly[413];
    poly[4816] = poly[17] * poly[414];
    poly[4817] = poly[17] * poly[415];
    poly[4818] = poly[1] * poly[417];
    poly[4819] = poly[2] * poly[417];
    poly[4820] = poly[3] * poly[417];
    poly[4821] = poly[4] * poly[417];
    poly[4822] = poly[5] * poly[417];
    poly[4823] = poly[6] * poly[417];
    poly[4824] = poly[7] * poly[417];
    poly[4825] = poly[8] * poly[417];
    poly[4826] = poly[9] * poly[417];
    poly[4827] = poly[10] * poly[417];
    poly[4828] = poly[11] * poly[417];
    poly[4829] = poly[12] * poly[417];
    poly[4830] = poly[13] * poly[417];
    poly[4831] = poly[14] * poly[417];
    poly[4832] = poly[15] * poly[417];
    poly[4833] = poly[16] * poly[417];
    poly[4834] = poly[17] * poly[417];
    poly[4835] = poly[217] * poly[26];
    poly[4836] = poly[1] * poly[418];
    poly[4837] = poly[2] * poly[418];
    poly[4838] = poly[3] * poly[418];
    poly[4839] = poly[4] * poly[418];
    poly[4840] = poly[5] * poly[418];
    poly[4841] = poly[6] * poly[418];
    poly[4842] = poly[7] * poly[418];
    poly[4843] = poly[8] * poly[418];
    poly[4844] = poly[9] * poly[418];
    poly[4845] = poly[10] * poly[418];
    poly[4846] = poly[11] * poly[418];
    poly[4847] = poly[12] * poly[418];
    poly[4848] = poly[13] * poly[418];
    poly[4849] = poly[14] * poly[418];
}

fn f_polynomials97(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4850] = poly[15] * poly[418];
    poly[4851] = poly[16] * poly[418];
    poly[4852] = poly[17] * poly[418];
    poly[4853] = poly[26] * poly[235];
    poly[4854] = poly[236] * poly[26];
    poly[4855] = poly[26] * poly[237];
    poly[4856] = poly[1] * poly[419];
    poly[4857] = poly[2] * poly[419];
    poly[4858] = poly[3] * poly[419];
    poly[4859] = poly[4] * poly[419];
    poly[4860] = poly[5] * poly[419];
    poly[4861] = poly[6] * poly[419];
    poly[4862] = poly[7] * poly[419];
    poly[4863] = poly[8] * poly[419];
    poly[4864] = poly[9] * poly[419];
    poly[4865] = poly[10] * poly[419];
    poly[4866] = poly[11] * poly[419];
    poly[4867] = poly[12] * poly[419];
    poly[4868] = poly[13] * poly[419];
    poly[4869] = poly[14] * poly[419];
    poly[4870] = poly[15] * poly[419];
    poly[4871] = poly[16] * poly[419];
    poly[4872] = poly[17] * poly[419];
    poly[4873] = poly[26] * poly[255];
    poly[4874] = poly[26] * poly[256];
    poly[4875] = poly[257] * poly[26];
    poly[4876] = poly[26] * poly[258];
    poly[4877] = poly[26] * poly[259];
    poly[4878] = poly[1] * poly[420];
    poly[4879] = poly[2] * poly[420];
    poly[4880] = poly[3] * poly[420];
    poly[4881] = poly[4] * poly[420];
    poly[4882] = poly[5] * poly[420];
    poly[4883] = poly[6] * poly[420];
    poly[4884] = poly[7] * poly[420];
    poly[4885] = poly[8] * poly[420];
    poly[4886] = poly[9] * poly[420];
    poly[4887] = poly[10] * poly[420];
    poly[4888] = poly[11] * poly[420];
    poly[4889] = poly[12] * poly[420];
    poly[4890] = poly[13] * poly[420];
    poly[4891] = poly[14] * poly[420];
    poly[4892] = poly[15] * poly[420];
    poly[4893] = poly[16] * poly[420];
    poly[4894] = poly[17] * poly[420];
    poly[4895] = poly[26] * poly[277];
    poly[4896] = poly[26] * poly[278];
    poly[4897] = poly[26] * poly[279];
    poly[4898] = poly[280] * poly[26];
    poly[4899] = poly[26] * poly[281];
}

fn f_polynomials98(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4900] = poly[26] * poly[282];
    poly[4901] = poly[26] * poly[283];
    poly[4902] = poly[1] * poly[421];
    poly[4903] = poly[2] * poly[421];
    poly[4904] = poly[3] * poly[421];
    poly[4905] = poly[4] * poly[421];
    poly[4906] = poly[5] * poly[421];
    poly[4907] = poly[6] * poly[421];
    poly[4908] = poly[26] * poly[290];
    poly[4909] = poly[26] * poly[291];
    poly[4910] = poly[26] * poly[292];
    poly[4911] = poly[26] * poly[293];
    poly[4912] = poly[26] * poly[294];
    poly[4913] = poly[26] * poly[295];
    poly[4914] = poly[26] * poly[296];
    poly[4915] = poly[26] * poly[297];
    poly[4916] = poly[11] * poly[421];
    poly[4917] = poly[12] * poly[421];
    poly[4918] = poly[13] * poly[421];
    poly[4919] = poly[14] * poly[421];
    poly[4920] = poly[15] * poly[421];
    poly[4921] = poly[26] * poly[303];
    poly[4922] = poly[26] * poly[304];
    poly[4923] = poly[17] * poly[421];
    poly[4924] = poly[26] * poly[306];
    poly[4925] = poly[26] * poly[307];
    poly[4926] = poly[26] * poly[308];
    poly[4927] = poly[26] * poly[309];
    poly[4928] = poly[26] * poly[310];
    poly[4929] = poly[26] * poly[311];
    poly[4930] = poly[26] * poly[312];
    poly[4931] = poly[26] * poly[313];
    poly[4932] = poly[26] * poly[314];
    poly[4933] = poly[26] * poly[315];
    poly[4934] = poly[26] * poly[316];
    poly[4935] = poly[1] * poly[422];
    poly[4936] = poly[2] * poly[422];
    poly[4937] = poly[3] * poly[422];
    poly[4938] = poly[4] * poly[422];
    poly[4939] = poly[5] * poly[422];
    poly[4940] = poly[6] * poly[422];
    poly[4941] = poly[7] * poly[422];
    poly[4942] = poly[8] * poly[422];
    poly[4943] = poly[9] * poly[422];
    poly[4944] = poly[10] * poly[422];
    poly[4945] = poly[11] * poly[422];
    poly[4946] = poly[26] * poly[328];
    poly[4947] = poly[26] * poly[329];
    poly[4948] = poly[26] * poly[330];
    poly[4949] = poly[26] * poly[331];
}

fn f_polynomials99(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[4950] = poly[26] * poly[332];
    poly[4951] = poly[26] * poly[333];
    poly[4952] = poly[26] * poly[334];
    poly[4953] = poly[26] * poly[335];
    poly[4954] = poly[26] * poly[336];
    poly[4955] = poly[26] * poly[337];
    poly[4956] = poly[17] * poly[422];
    poly[4957] = poly[26] * poly[339];
    poly[4958] = poly[26] * poly[340];
    poly[4959] = poly[26] * poly[341];
    poly[4960] = poly[26] * poly[342];
    poly[4961] = poly[26] * poly[343];
    poly[4962] = poly[26] * poly[344];
    poly[4963] = poly[26] * poly[345];
    poly[4964] = poly[26] * poly[346];
    poly[4965] = poly[26] * poly[347];
    poly[4966] = poly[26] * poly[348];
    poly[4967] = poly[26] * poly[349];
    poly[4968] = poly[26] * poly[350];
    poly[4969] = poly[26] * poly[351];
    poly[4970] = poly[1] * poly[423];
    poly[4971] = poly[2] * poly[423];
    poly[4972] = poly[3] * poly[423];
    poly[4973] = poly[4] * poly[423];
    poly[4974] = poly[5] * poly[423];
    poly[4975] = poly[6] * poly[423];
    poly[4976] = poly[24] * poly[406];
    poly[4977] = poly[24] * poly[407];
    poly[4978] = poly[24] * poly[408];
    poly[4979] = poly[24] * poly[409];
    poly[4980] = poly[11] * poly[423];
    poly[4981] = poly[24] * poly[411];
    poly[4982] = poly[24] * poly[412];
    poly[4983] = poly[24] * poly[413];
    poly[4984] = poly[24] * poly[414];
    poly[4985] = poly[24] * poly[415];
    poly[4986] = poly[17] * poly[423];
    poly[4987] = poly[24] * poly[417];
    poly[4988] = poly[24] * poly[418];
    poly[4989] = poly[24] * poly[419];
    poly[4990] = poly[24] * poly[420];
    poly[4991] = poly[24] * poly[421];
    poly[4992] = poly[24] * poly[422];
    poly[4993] = poly[1] * poly[424];
    poly[4994] = poly[2] * poly[424];
    poly[4995] = poly[3] * poly[424];
    poly[4996] = poly[4] * poly[424];
    poly[4997] = poly[5] * poly[424];
    poly[4998] = poly[6] * poly[424];
    poly[4999] = poly[7] * poly[424];
}

fn f_polynomials100(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5000] = poly[8] * poly[424];
    poly[5001] = poly[9] * poly[424];
    poly[5002] = poly[10] * poly[424];
    poly[5003] = poly[11] * poly[424];
    poly[5004] = poly[12] * poly[424];
    poly[5005] = poly[13] * poly[424];
    poly[5006] = poly[14] * poly[424];
    poly[5007] = poly[15] * poly[424];
    poly[5008] = poly[16] * poly[424];
    poly[5009] = poly[17] * poly[424];
    poly[5010] = poly[18] * poly[424];
    poly[5011] = poly[19] * poly[424];
    poly[5012] = poly[20] * poly[424];
    poly[5013] = poly[21] * poly[424];
    poly[5014] = poly[22] * poly[424];
    poly[5015] = poly[23] * poly[424];
    poly[5016] = poly[24] * poly[424];
    poly[5017] = poly[1] * poly[425];
    poly[5018] = poly[2] * poly[425];
    poly[5019] = poly[3] * poly[425];
    poly[5020] = poly[4] * poly[425];
    poly[5021] = poly[5] * poly[425];
    poly[5022] = poly[6] * poly[425];
    poly[5023] = poly[425] * poly[7];
    poly[5024] = poly[425] * poly[8];
    poly[5025] = poly[425] * poly[9];
    poly[5026] = poly[425] * poly[10];
    poly[5027] = poly[11] * poly[425];
    poly[5028] = poly[425] * poly[12];
    poly[5029] = poly[425] * poly[13];
    poly[5030] = poly[425] * poly[14];
    poly[5031] = poly[425] * poly[15];
    poly[5032] = poly[425] * poly[16];
    poly[5033] = poly[17] * poly[425];
    poly[5034] = poly[425] * poly[18];
    poly[5035] = poly[425] * poly[19];
    poly[5036] = poly[425] * poly[20];
    poly[5037] = poly[425] * poly[21];
    poly[5038] = poly[425] * poly[22];
    poly[5039] = poly[425] * poly[23];
    poly[5040] = poly[24] * poly[425];
    poly[5041] = poly[1] * poly[426];
    poly[5042] = poly[2] * poly[426];
    poly[5043] = poly[3] * poly[426];
    poly[5044] = poly[4] * poly[426];
    poly[5045] = poly[5] * poly[426];
    poly[5046] = poly[6] * poly[426];
    poly[5047] = poly[7] * poly[426];
    poly[5048] = poly[8] * poly[426];
    poly[5049] = poly[9] * poly[426];
}

fn f_polynomials101(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5050] = poly[10] * poly[426];
    poly[5051] = poly[11] * poly[426];
    poly[5052] = poly[12] * poly[426];
    poly[5053] = poly[13] * poly[426];
    poly[5054] = poly[14] * poly[426];
    poly[5055] = poly[15] * poly[426];
    poly[5056] = poly[16] * poly[426];
    poly[5057] = poly[17] * poly[426];
    poly[5058] = poly[18] * poly[426];
    poly[5059] = poly[19] * poly[426];
    poly[5060] = poly[20] * poly[426];
    poly[5061] = poly[21] * poly[426];
    poly[5062] = poly[22] * poly[426];
    poly[5063] = poly[23] * poly[426];
    poly[5064] = poly[24] * poly[426];
    poly[5065] = poly[399] * poly[26];
    poly[5066] = poly[425] * poly[25];
    poly[5067] = poly[1] * poly[428];
    poly[5068] = poly[1] * poly[429];
    poly[5069] = poly[2] * poly[429];
    poly[5070] = poly[1] * poly[430];
    poly[5071] = poly[2] * poly[430];
    poly[5072] = poly[3] * poly[430];
    poly[5073] = poly[1] * poly[431];
    poly[5074] = poly[2] * poly[431];
    poly[5075] = poly[3] * poly[431];
    poly[5076] = poly[4] * poly[431];
    poly[5077] = poly[1] * poly[432];
    poly[5078] = poly[2] * poly[432];
    poly[5079] = poly[3] * poly[432];
    poly[5080] = poly[4] * poly[432];
    poly[5081] = poly[5] * poly[432];
    poly[5082] = poly[1] * poly[433];
    poly[5083] = poly[2] * poly[433];
    poly[5084] = poly[3] * poly[433];
    poly[5085] = poly[4] * poly[433];
    poly[5086] = poly[5] * poly[433];
    poly[5087] = poly[6] * poly[433];
    poly[5088] = poly[54] * poly[27];
    poly[5089] = poly[1] * poly[434];
    poly[5090] = poly[2] * poly[434];
    poly[5091] = poly[3] * poly[434];
    poly[5092] = poly[4] * poly[434];
    poly[5093] = poly[5] * poly[434];
    poly[5094] = poly[6] * poly[434];
    poly[5095] = poly[27] * poly[61];
    poly[5096] = poly[62] * poly[27];
    poly[5097] = poly[27] * poly[63];
    poly[5098] = poly[1] * poly[435];
    poly[5099] = poly[2] * poly[435];
}

fn f_polynomials102(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5100] = poly[3] * poly[435];
    poly[5101] = poly[4] * poly[435];
    poly[5102] = poly[5] * poly[435];
    poly[5103] = poly[6] * poly[435];
    poly[5104] = poly[27] * poly[70];
    poly[5105] = poly[27] * poly[71];
    poly[5106] = poly[72] * poly[27];
    poly[5107] = poly[27] * poly[73];
    poly[5108] = poly[27] * poly[74];
    poly[5109] = poly[1] * poly[436];
    poly[5110] = poly[2] * poly[436];
    poly[5111] = poly[3] * poly[436];
    poly[5112] = poly[4] * poly[436];
    poly[5113] = poly[5] * poly[436];
    poly[5114] = poly[6] * poly[436];
    poly[5115] = poly[27] * poly[81];
    poly[5116] = poly[27] * poly[82];
    poly[5117] = poly[27] * poly[83];
    poly[5118] = poly[84] * poly[27];
    poly[5119] = poly[27] * poly[85];
    poly[5120] = poly[27] * poly[86];
    poly[5121] = poly[27] * poly[87];
    poly[5122] = poly[1] * poly[437];
    poly[5123] = poly[2] * poly[437];
    poly[5124] = poly[3] * poly[437];
    poly[5125] = poly[4] * poly[437];
    poly[5126] = poly[5] * poly[437];
    poly[5127] = poly[6] * poly[437];
    poly[5128] = poly[11] * poly[433];
    poly[5129] = poly[11] * poly[434];
    poly[5130] = poly[11] * poly[435];
    poly[5131] = poly[11] * poly[436];
    poly[5132] = poly[1] * poly[438];
    poly[5133] = poly[2] * poly[438];
    poly[5134] = poly[3] * poly[438];
    poly[5135] = poly[4] * poly[438];
    poly[5136] = poly[5] * poly[438];
    poly[5137] = poly[6] * poly[438];
    poly[5138] = poly[7] * poly[438];
    poly[5139] = poly[8] * poly[438];
    poly[5140] = poly[9] * poly[438];
    poly[5141] = poly[10] * poly[438];
    poly[5142] = poly[11] * poly[438];
    poly[5143] = poly[109] * poly[27];
    poly[5144] = poly[1] * poly[439];
    poly[5145] = poly[2] * poly[439];
    poly[5146] = poly[3] * poly[439];
    poly[5147] = poly[4] * poly[439];
    poly[5148] = poly[5] * poly[439];
    poly[5149] = poly[6] * poly[439];
}

fn f_polynomials103(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5150] = poly[7] * poly[439];
    poly[5151] = poly[8] * poly[439];
    poly[5152] = poly[9] * poly[439];
    poly[5153] = poly[10] * poly[439];
    poly[5154] = poly[11] * poly[439];
    poly[5155] = poly[27] * poly[121];
    poly[5156] = poly[122] * poly[27];
    poly[5157] = poly[27] * poly[123];
    poly[5158] = poly[1] * poly[440];
    poly[5159] = poly[2] * poly[440];
    poly[5160] = poly[3] * poly[440];
    poly[5161] = poly[4] * poly[440];
    poly[5162] = poly[5] * poly[440];
    poly[5163] = poly[6] * poly[440];
    poly[5164] = poly[7] * poly[440];
    poly[5165] = poly[8] * poly[440];
    poly[5166] = poly[9] * poly[440];
    poly[5167] = poly[10] * poly[440];
    poly[5168] = poly[11] * poly[440];
    poly[5169] = poly[27] * poly[135];
    poly[5170] = poly[27] * poly[136];
    poly[5171] = poly[137] * poly[27];
    poly[5172] = poly[27] * poly[138];
    poly[5173] = poly[27] * poly[139];
    poly[5174] = poly[1] * poly[441];
    poly[5175] = poly[2] * poly[441];
    poly[5176] = poly[3] * poly[441];
    poly[5177] = poly[4] * poly[441];
    poly[5178] = poly[5] * poly[441];
    poly[5179] = poly[6] * poly[441];
    poly[5180] = poly[7] * poly[441];
    poly[5181] = poly[8] * poly[441];
    poly[5182] = poly[9] * poly[441];
    poly[5183] = poly[10] * poly[441];
    poly[5184] = poly[11] * poly[441];
    poly[5185] = poly[27] * poly[151];
    poly[5186] = poly[27] * poly[152];
    poly[5187] = poly[27] * poly[153];
    poly[5188] = poly[154] * poly[27];
    poly[5189] = poly[27] * poly[155];
    poly[5190] = poly[27] * poly[156];
    poly[5191] = poly[27] * poly[157];
    poly[5192] = poly[1] * poly[442];
    poly[5193] = poly[2] * poly[442];
    poly[5194] = poly[3] * poly[442];
    poly[5195] = poly[4] * poly[442];
    poly[5196] = poly[5] * poly[442];
    poly[5197] = poly[6] * poly[442];
    poly[5198] = poly[27] * poly[164];
    poly[5199] = poly[27] * poly[165];
}

fn f_polynomials104(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5200] = poly[27] * poly[166];
    poly[5201] = poly[27] * poly[167];
    poly[5202] = poly[27] * poly[168];
    poly[5203] = poly[27] * poly[169];
    poly[5204] = poly[27] * poly[170];
    poly[5205] = poly[27] * poly[171];
    poly[5206] = poly[11] * poly[442];
    poly[5207] = poly[27] * poly[173];
    poly[5208] = poly[27] * poly[174];
    poly[5209] = poly[27] * poly[175];
    poly[5210] = poly[27] * poly[176];
    poly[5211] = poly[27] * poly[177];
    poly[5212] = poly[27] * poly[178];
    poly[5213] = poly[27] * poly[179];
    poly[5214] = poly[27] * poly[180];
    poly[5215] = poly[27] * poly[181];
    poly[5216] = poly[27] * poly[182];
    poly[5217] = poly[27] * poly[183];
    poly[5218] = poly[1] * poly[443];
    poly[5219] = poly[2] * poly[443];
    poly[5220] = poly[3] * poly[443];
    poly[5221] = poly[4] * poly[443];
    poly[5222] = poly[5] * poly[443];
    poly[5223] = poly[6] * poly[443];
    poly[5224] = poly[17] * poly[433];
    poly[5225] = poly[17] * poly[434];
    poly[5226] = poly[17] * poly[435];
    poly[5227] = poly[17] * poly[436];
    poly[5228] = poly[11] * poly[443];
    poly[5229] = poly[17] * poly[438];
    poly[5230] = poly[17] * poly[439];
    poly[5231] = poly[17] * poly[440];
    poly[5232] = poly[17] * poly[441];
    poly[5233] = poly[17] * poly[442];
    poly[5234] = poly[1] * poly[444];
    poly[5235] = poly[2] * poly[444];
    poly[5236] = poly[3] * poly[444];
    poly[5237] = poly[4] * poly[444];
    poly[5238] = poly[5] * poly[444];
    poly[5239] = poly[6] * poly[444];
    poly[5240] = poly[7] * poly[444];
    poly[5241] = poly[8] * poly[444];
    poly[5242] = poly[9] * poly[444];
    poly[5243] = poly[10] * poly[444];
    poly[5244] = poly[11] * poly[444];
    poly[5245] = poly[12] * poly[444];
    poly[5246] = poly[13] * poly[444];
    poly[5247] = poly[14] * poly[444];
    poly[5248] = poly[15] * poly[444];
    poly[5249] = poly[16] * poly[444];
}

fn f_polynomials105(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5250] = poly[17] * poly[444];
    poly[5251] = poly[217] * poly[27];
    poly[5252] = poly[1] * poly[445];
    poly[5253] = poly[2] * poly[445];
    poly[5254] = poly[3] * poly[445];
    poly[5255] = poly[4] * poly[445];
    poly[5256] = poly[5] * poly[445];
    poly[5257] = poly[6] * poly[445];
    poly[5258] = poly[7] * poly[445];
    poly[5259] = poly[8] * poly[445];
    poly[5260] = poly[9] * poly[445];
    poly[5261] = poly[10] * poly[445];
    poly[5262] = poly[11] * poly[445];
    poly[5263] = poly[12] * poly[445];
    poly[5264] = poly[13] * poly[445];
    poly[5265] = poly[14] * poly[445];
    poly[5266] = poly[15] * poly[445];
    poly[5267] = poly[16] * poly[445];
    poly[5268] = poly[17] * poly[445];
    poly[5269] = poly[27] * poly[235];
    poly[5270] = poly[236] * poly[27];
    poly[5271] = poly[27] * poly[237];
    poly[5272] = poly[1] * poly[446];
    poly[5273] = poly[2] * poly[446];
    poly[5274] = poly[3] * poly[446];
    poly[5275] = poly[4] * poly[446];
    poly[5276] = poly[5] * poly[446];
    poly[5277] = poly[6] * poly[446];
    poly[5278] = poly[7] * poly[446];
    poly[5279] = poly[8] * poly[446];
    poly[5280] = poly[9] * poly[446];
    poly[5281] = poly[10] * poly[446];
    poly[5282] = poly[11] * poly[446];
    poly[5283] = poly[12] * poly[446];
    poly[5284] = poly[13] * poly[446];
    poly[5285] = poly[14] * poly[446];
    poly[5286] = poly[15] * poly[446];
    poly[5287] = poly[16] * poly[446];
    poly[5288] = poly[17] * poly[446];
    poly[5289] = poly[27] * poly[255];
    poly[5290] = poly[27] * poly[256];
    poly[5291] = poly[257] * poly[27];
    poly[5292] = poly[27] * poly[258];
    poly[5293] = poly[27] * poly[259];
    poly[5294] = poly[1] * poly[447];
    poly[5295] = poly[2] * poly[447];
    poly[5296] = poly[3] * poly[447];
    poly[5297] = poly[4] * poly[447];
    poly[5298] = poly[5] * poly[447];
    poly[5299] = poly[6] * poly[447];
}

fn f_polynomials106(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5300] = poly[7] * poly[447];
    poly[5301] = poly[8] * poly[447];
    poly[5302] = poly[9] * poly[447];
    poly[5303] = poly[10] * poly[447];
    poly[5304] = poly[11] * poly[447];
    poly[5305] = poly[12] * poly[447];
    poly[5306] = poly[13] * poly[447];
    poly[5307] = poly[14] * poly[447];
    poly[5308] = poly[15] * poly[447];
    poly[5309] = poly[16] * poly[447];
    poly[5310] = poly[17] * poly[447];
    poly[5311] = poly[27] * poly[277];
    poly[5312] = poly[27] * poly[278];
    poly[5313] = poly[27] * poly[279];
    poly[5314] = poly[280] * poly[27];
    poly[5315] = poly[27] * poly[281];
    poly[5316] = poly[27] * poly[282];
    poly[5317] = poly[27] * poly[283];
    poly[5318] = poly[1] * poly[448];
    poly[5319] = poly[2] * poly[448];
    poly[5320] = poly[3] * poly[448];
    poly[5321] = poly[4] * poly[448];
    poly[5322] = poly[5] * poly[448];
    poly[5323] = poly[6] * poly[448];
    poly[5324] = poly[27] * poly[290];
    poly[5325] = poly[27] * poly[291];
    poly[5326] = poly[27] * poly[292];
    poly[5327] = poly[27] * poly[293];
    poly[5328] = poly[27] * poly[294];
    poly[5329] = poly[27] * poly[295];
    poly[5330] = poly[27] * poly[296];
    poly[5331] = poly[27] * poly[297];
    poly[5332] = poly[11] * poly[448];
    poly[5333] = poly[12] * poly[448];
    poly[5334] = poly[13] * poly[448];
    poly[5335] = poly[14] * poly[448];
    poly[5336] = poly[15] * poly[448];
    poly[5337] = poly[27] * poly[303];
    poly[5338] = poly[27] * poly[304];
    poly[5339] = poly[17] * poly[448];
    poly[5340] = poly[27] * poly[306];
    poly[5341] = poly[27] * poly[307];
    poly[5342] = poly[27] * poly[308];
    poly[5343] = poly[27] * poly[309];
    poly[5344] = poly[27] * poly[310];
    poly[5345] = poly[27] * poly[311];
    poly[5346] = poly[27] * poly[312];
    poly[5347] = poly[27] * poly[313];
    poly[5348] = poly[27] * poly[314];
    poly[5349] = poly[27] * poly[315];
}

fn f_polynomials107(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5350] = poly[27] * poly[316];
    poly[5351] = poly[1] * poly[449];
    poly[5352] = poly[2] * poly[449];
    poly[5353] = poly[3] * poly[449];
    poly[5354] = poly[4] * poly[449];
    poly[5355] = poly[5] * poly[449];
    poly[5356] = poly[6] * poly[449];
    poly[5357] = poly[7] * poly[449];
    poly[5358] = poly[8] * poly[449];
    poly[5359] = poly[9] * poly[449];
    poly[5360] = poly[10] * poly[449];
    poly[5361] = poly[11] * poly[449];
    poly[5362] = poly[27] * poly[328];
    poly[5363] = poly[27] * poly[329];
    poly[5364] = poly[27] * poly[330];
    poly[5365] = poly[27] * poly[331];
    poly[5366] = poly[27] * poly[332];
    poly[5367] = poly[27] * poly[333];
    poly[5368] = poly[27] * poly[334];
    poly[5369] = poly[27] * poly[335];
    poly[5370] = poly[27] * poly[336];
    poly[5371] = poly[27] * poly[337];
    poly[5372] = poly[17] * poly[449];
    poly[5373] = poly[27] * poly[339];
    poly[5374] = poly[27] * poly[340];
    poly[5375] = poly[27] * poly[341];
    poly[5376] = poly[27] * poly[342];
    poly[5377] = poly[27] * poly[343];
    poly[5378] = poly[27] * poly[344];
    poly[5379] = poly[27] * poly[345];
    poly[5380] = poly[27] * poly[346];
    poly[5381] = poly[27] * poly[347];
    poly[5382] = poly[27] * poly[348];
    poly[5383] = poly[27] * poly[349];
    poly[5384] = poly[27] * poly[350];
    poly[5385] = poly[27] * poly[351];
    poly[5386] = poly[1] * poly[450];
    poly[5387] = poly[2] * poly[450];
    poly[5388] = poly[3] * poly[450];
    poly[5389] = poly[4] * poly[450];
    poly[5390] = poly[5] * poly[450];
    poly[5391] = poly[6] * poly[450];
    poly[5392] = poly[24] * poly[433];
    poly[5393] = poly[24] * poly[434];
    poly[5394] = poly[24] * poly[435];
    poly[5395] = poly[24] * poly[436];
    poly[5396] = poly[11] * poly[450];
    poly[5397] = poly[24] * poly[438];
    poly[5398] = poly[24] * poly[439];
    poly[5399] = poly[24] * poly[440];
}

fn f_polynomials108(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5400] = poly[24] * poly[441];
    poly[5401] = poly[24] * poly[442];
    poly[5402] = poly[17] * poly[450];
    poly[5403] = poly[24] * poly[444];
    poly[5404] = poly[24] * poly[445];
    poly[5405] = poly[24] * poly[446];
    poly[5406] = poly[24] * poly[447];
    poly[5407] = poly[24] * poly[448];
    poly[5408] = poly[24] * poly[449];
    poly[5409] = poly[1] * poly[451];
    poly[5410] = poly[2] * poly[451];
    poly[5411] = poly[3] * poly[451];
    poly[5412] = poly[4] * poly[451];
    poly[5413] = poly[5] * poly[451];
    poly[5414] = poly[6] * poly[451];
    poly[5415] = poly[7] * poly[451];
    poly[5416] = poly[8] * poly[451];
    poly[5417] = poly[9] * poly[451];
    poly[5418] = poly[10] * poly[451];
    poly[5419] = poly[11] * poly[451];
    poly[5420] = poly[12] * poly[451];
    poly[5421] = poly[13] * poly[451];
    poly[5422] = poly[14] * poly[451];
    poly[5423] = poly[15] * poly[451];
    poly[5424] = poly[16] * poly[451];
    poly[5425] = poly[17] * poly[451];
    poly[5426] = poly[18] * poly[451];
    poly[5427] = poly[19] * poly[451];
    poly[5428] = poly[20] * poly[451];
    poly[5429] = poly[21] * poly[451];
    poly[5430] = poly[22] * poly[451];
    poly[5431] = poly[23] * poly[451];
    poly[5432] = poly[24] * poly[451];
    poly[5433] = poly[1] * poly[452];
    poly[5434] = poly[2] * poly[452];
    poly[5435] = poly[3] * poly[452];
    poly[5436] = poly[4] * poly[452];
    poly[5437] = poly[5] * poly[452];
    poly[5438] = poly[6] * poly[452];
    poly[5439] = poly[7] * poly[452];
    poly[5440] = poly[8] * poly[452];
    poly[5441] = poly[9] * poly[452];
    poly[5442] = poly[10] * poly[452];
    poly[5443] = poly[11] * poly[452];
    poly[5444] = poly[12] * poly[452];
    poly[5445] = poly[13] * poly[452];
    poly[5446] = poly[14] * poly[452];
    poly[5447] = poly[15] * poly[452];
    poly[5448] = poly[16] * poly[452];
    poly[5449] = poly[17] * poly[452];
}

fn f_polynomials109(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5450] = poly[18] * poly[452];
    poly[5451] = poly[19] * poly[452];
    poly[5452] = poly[20] * poly[452];
    poly[5453] = poly[21] * poly[452];
    poly[5454] = poly[22] * poly[452];
    poly[5455] = poly[23] * poly[452];
    poly[5456] = poly[24] * poly[452];
    poly[5457] = mono[1219] + mono[1220];
    poly[5458] = poly[1] * poly[453];
    poly[5459] = poly[2] * poly[453];
    poly[5460] = poly[3] * poly[453];
    poly[5461] = poly[4] * poly[453];
    poly[5462] = poly[5] * poly[453];
    poly[5463] = poly[6] * poly[453];
    poly[5464] = poly[453] * poly[7];
    poly[5465] = poly[453] * poly[8];
    poly[5466] = poly[453] * poly[9];
    poly[5467] = poly[453] * poly[10];
    poly[5468] = poly[11] * poly[453];
    poly[5469] = poly[453] * poly[12];
    poly[5470] = poly[453] * poly[13];
    poly[5471] = poly[453] * poly[14];
    poly[5472] = poly[453] * poly[15];
    poly[5473] = poly[453] * poly[16];
    poly[5474] = poly[17] * poly[453];
    poly[5475] = poly[453] * poly[18];
    poly[5476] = poly[453] * poly[19];
    poly[5477] = poly[453] * poly[20];
    poly[5478] = poly[453] * poly[21];
    poly[5479] = poly[453] * poly[22];
    poly[5480] = poly[453] * poly[23];
    poly[5481] = poly[24] * poly[453];
    poly[5482] = poly[1] * poly[454];
    poly[5483] = poly[2] * poly[454];
    poly[5484] = poly[3] * poly[454];
    poly[5485] = poly[4] * poly[454];
    poly[5486] = poly[5] * poly[454];
    poly[5487] = poly[6] * poly[454];
    poly[5488] = poly[7] * poly[454];
    poly[5489] = poly[8] * poly[454];
    poly[5490] = poly[9] * poly[454];
    poly[5491] = poly[10] * poly[454];
    poly[5492] = poly[11] * poly[454];
    poly[5493] = poly[12] * poly[454];
    poly[5494] = poly[13] * poly[454];
    poly[5495] = poly[14] * poly[454];
    poly[5496] = poly[15] * poly[454];
    poly[5497] = poly[16] * poly[454];
    poly[5498] = poly[17] * poly[454];
    poly[5499] = poly[18] * poly[454];
}

fn f_polynomials110(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5500] = poly[19] * poly[454];
    poly[5501] = poly[20] * poly[454];
    poly[5502] = poly[21] * poly[454];
    poly[5503] = poly[22] * poly[454];
    poly[5504] = poly[23] * poly[454];
    poly[5505] = poly[24] * poly[454];
    poly[5506] = poly[399] * poly[27];
    poly[5507] = poly[25] * poly[452] - poly[5457];
    poly[5508] = poly[453] * poly[25];
    poly[5509] = poly[1] * poly[455];
    poly[5510] = poly[2] * poly[455];
    poly[5511] = poly[3] * poly[455];
    poly[5512] = poly[4] * poly[455];
    poly[5513] = poly[5] * poly[455];
    poly[5514] = poly[6] * poly[455];
    poly[5515] = poly[7] * poly[455];
    poly[5516] = poly[8] * poly[455];
    poly[5517] = poly[9] * poly[455];
    poly[5518] = poly[10] * poly[455];
    poly[5519] = poly[11] * poly[455];
    poly[5520] = poly[12] * poly[455];
    poly[5521] = poly[13] * poly[455];
    poly[5522] = poly[14] * poly[455];
    poly[5523] = poly[15] * poly[455];
    poly[5524] = poly[16] * poly[455];
    poly[5525] = poly[17] * poly[455];
    poly[5526] = poly[18] * poly[455];
    poly[5527] = poly[19] * poly[455];
    poly[5528] = poly[20] * poly[455];
    poly[5529] = poly[21] * poly[455];
    poly[5530] = poly[22] * poly[455];
    poly[5531] = poly[23] * poly[455];
    poly[5532] = poly[24] * poly[455];
    poly[5533] = poly[26] * poly[451] - poly[5457];
    poly[5534] = poly[425] * poly[27];
    poly[5535] = poly[453] * poly[26];
    poly[5536] = poly[25] * poly[455] - poly[5533];
    poly[5537] = poly[1] * poly[457];
    poly[5538] = poly[1] * poly[458];
    poly[5539] = poly[2] * poly[458];
    poly[5540] = poly[1] * poly[459];
    poly[5541] = poly[2] * poly[459];
    poly[5542] = poly[3] * poly[459];
    poly[5543] = poly[1] * poly[460];
    poly[5544] = poly[2] * poly[460];
    poly[5545] = poly[3] * poly[460];
    poly[5546] = poly[4] * poly[460];
    poly[5547] = poly[1] * poly[461];
    poly[5548] = poly[2] * poly[461];
    poly[5549] = poly[3] * poly[461];
}

fn f_polynomials111(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5550] = poly[4] * poly[461];
    poly[5551] = poly[5] * poly[461];
    poly[5552] = poly[1] * poly[462];
    poly[5553] = poly[2] * poly[462];
    poly[5554] = poly[3] * poly[462];
    poly[5555] = poly[4] * poly[462];
    poly[5556] = poly[5] * poly[462];
    poly[5557] = poly[6] * poly[462];
    poly[5558] = poly[54] * poly[28];
    poly[5559] = poly[1] * poly[463];
    poly[5560] = poly[2] * poly[463];
    poly[5561] = poly[3] * poly[463];
    poly[5562] = poly[4] * poly[463];
    poly[5563] = poly[5] * poly[463];
    poly[5564] = poly[6] * poly[463];
    poly[5565] = poly[28] * poly[61];
    poly[5566] = poly[62] * poly[28];
    poly[5567] = poly[28] * poly[63];
    poly[5568] = poly[1] * poly[464];
    poly[5569] = poly[2] * poly[464];
    poly[5570] = poly[3] * poly[464];
    poly[5571] = poly[4] * poly[464];
    poly[5572] = poly[5] * poly[464];
    poly[5573] = poly[6] * poly[464];
    poly[5574] = poly[28] * poly[70];
    poly[5575] = poly[28] * poly[71];
    poly[5576] = poly[72] * poly[28];
    poly[5577] = poly[28] * poly[73];
    poly[5578] = poly[28] * poly[74];
    poly[5579] = poly[1] * poly[465];
    poly[5580] = poly[2] * poly[465];
    poly[5581] = poly[3] * poly[465];
    poly[5582] = poly[4] * poly[465];
    poly[5583] = poly[5] * poly[465];
    poly[5584] = poly[6] * poly[465];
    poly[5585] = poly[28] * poly[81];
    poly[5586] = poly[28] * poly[82];
    poly[5587] = poly[28] * poly[83];
    poly[5588] = poly[84] * poly[28];
    poly[5589] = poly[28] * poly[85];
    poly[5590] = poly[28] * poly[86];
    poly[5591] = poly[28] * poly[87];
    poly[5592] = poly[1] * poly[466];
    poly[5593] = poly[2] * poly[466];
    poly[5594] = poly[3] * poly[466];
    poly[5595] = poly[4] * poly[466];
    poly[5596] = poly[5] * poly[466];
    poly[5597] = poly[6] * poly[466];
    poly[5598] = poly[11] * poly[462];
    poly[5599] = poly[11] * poly[463];
}

fn f_polynomials112(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5600] = poly[11] * poly[464];
    poly[5601] = poly[11] * poly[465];
    poly[5602] = poly[1] * poly[467];
    poly[5603] = poly[2] * poly[467];
    poly[5604] = poly[3] * poly[467];
    poly[5605] = poly[4] * poly[467];
    poly[5606] = poly[5] * poly[467];
    poly[5607] = poly[6] * poly[467];
    poly[5608] = poly[7] * poly[467];
    poly[5609] = poly[8] * poly[467];
    poly[5610] = poly[9] * poly[467];
    poly[5611] = poly[10] * poly[467];
    poly[5612] = poly[11] * poly[467];
    poly[5613] = poly[109] * poly[28];
    poly[5614] = poly[1] * poly[468];
    poly[5615] = poly[2] * poly[468];
    poly[5616] = poly[3] * poly[468];
    poly[5617] = poly[4] * poly[468];
    poly[5618] = poly[5] * poly[468];
    poly[5619] = poly[6] * poly[468];
    poly[5620] = poly[7] * poly[468];
    poly[5621] = poly[8] * poly[468];
    poly[5622] = poly[9] * poly[468];
    poly[5623] = poly[10] * poly[468];
    poly[5624] = poly[11] * poly[468];
    poly[5625] = poly[28] * poly[121];
    poly[5626] = poly[122] * poly[28];
    poly[5627] = poly[28] * poly[123];
    poly[5628] = poly[1] * poly[469];
    poly[5629] = poly[2] * poly[469];
    poly[5630] = poly[3] * poly[469];
    poly[5631] = poly[4] * poly[469];
    poly[5632] = poly[5] * poly[469];
    poly[5633] = poly[6] * poly[469];
    poly[5634] = poly[7] * poly[469];
    poly[5635] = poly[8] * poly[469];
    poly[5636] = poly[9] * poly[469];
    poly[5637] = poly[10] * poly[469];
    poly[5638] = poly[11] * poly[469];
    poly[5639] = poly[28] * poly[135];
    poly[5640] = poly[28] * poly[136];
    poly[5641] = poly[137] * poly[28];
    poly[5642] = poly[28] * poly[138];
    poly[5643] = poly[28] * poly[139];
    poly[5644] = poly[1] * poly[470];
    poly[5645] = poly[2] * poly[470];
    poly[5646] = poly[3] * poly[470];
    poly[5647] = poly[4] * poly[470];
    poly[5648] = poly[5] * poly[470];
    poly[5649] = poly[6] * poly[470];
}

fn f_polynomials113(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5650] = poly[7] * poly[470];
    poly[5651] = poly[8] * poly[470];
    poly[5652] = poly[9] * poly[470];
    poly[5653] = poly[10] * poly[470];
    poly[5654] = poly[11] * poly[470];
    poly[5655] = poly[28] * poly[151];
    poly[5656] = poly[28] * poly[152];
    poly[5657] = poly[28] * poly[153];
    poly[5658] = poly[154] * poly[28];
    poly[5659] = poly[28] * poly[155];
    poly[5660] = poly[28] * poly[156];
    poly[5661] = poly[28] * poly[157];
    poly[5662] = poly[1] * poly[471];
    poly[5663] = poly[2] * poly[471];
    poly[5664] = poly[3] * poly[471];
    poly[5665] = poly[4] * poly[471];
    poly[5666] = poly[5] * poly[471];
    poly[5667] = poly[6] * poly[471];
    poly[5668] = poly[28] * poly[164];
    poly[5669] = poly[28] * poly[165];
    poly[5670] = poly[28] * poly[166];
    poly[5671] = poly[28] * poly[167];
    poly[5672] = poly[28] * poly[168];
    poly[5673] = poly[28] * poly[169];
    poly[5674] = poly[28] * poly[170];
    poly[5675] = poly[28] * poly[171];
    poly[5676] = poly[11] * poly[471];
    poly[5677] = poly[28] * poly[173];
    poly[5678] = poly[28] * poly[174];
    poly[5679] = poly[28] * poly[175];
    poly[5680] = poly[28] * poly[176];
    poly[5681] = poly[28] * poly[177];
    poly[5682] = poly[28] * poly[178];
    poly[5683] = poly[28] * poly[179];
    poly[5684] = poly[28] * poly[180];
    poly[5685] = poly[28] * poly[181];
    poly[5686] = poly[28] * poly[182];
    poly[5687] = poly[28] * poly[183];
    poly[5688] = poly[1] * poly[472];
    poly[5689] = poly[2] * poly[472];
    poly[5690] = poly[3] * poly[472];
    poly[5691] = poly[4] * poly[472];
    poly[5692] = poly[5] * poly[472];
    poly[5693] = poly[6] * poly[472];
    poly[5694] = poly[17] * poly[462];
    poly[5695] = poly[17] * poly[463];
    poly[5696] = poly[17] * poly[464];
    poly[5697] = poly[17] * poly[465];
    poly[5698] = poly[11] * poly[472];
    poly[5699] = poly[17] * poly[467];
}

fn f_polynomials114(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5700] = poly[17] * poly[468];
    poly[5701] = poly[17] * poly[469];
    poly[5702] = poly[17] * poly[470];
    poly[5703] = poly[17] * poly[471];
    poly[5704] = poly[1] * poly[473];
    poly[5705] = poly[2] * poly[473];
    poly[5706] = poly[3] * poly[473];
    poly[5707] = poly[4] * poly[473];
    poly[5708] = poly[5] * poly[473];
    poly[5709] = poly[6] * poly[473];
    poly[5710] = poly[7] * poly[473];
    poly[5711] = poly[8] * poly[473];
    poly[5712] = poly[9] * poly[473];
    poly[5713] = poly[10] * poly[473];
    poly[5714] = poly[11] * poly[473];
    poly[5715] = poly[12] * poly[473];
    poly[5716] = poly[13] * poly[473];
    poly[5717] = poly[14] * poly[473];
    poly[5718] = poly[15] * poly[473];
    poly[5719] = poly[16] * poly[473];
    poly[5720] = poly[17] * poly[473];
    poly[5721] = poly[217] * poly[28];
    poly[5722] = poly[1] * poly[474];
    poly[5723] = poly[2] * poly[474];
    poly[5724] = poly[3] * poly[474];
    poly[5725] = poly[4] * poly[474];
    poly[5726] = poly[5] * poly[474];
    poly[5727] = poly[6] * poly[474];
    poly[5728] = poly[7] * poly[474];
    poly[5729] = poly[8] * poly[474];
    poly[5730] = poly[9] * poly[474];
    poly[5731] = poly[10] * poly[474];
    poly[5732] = poly[11] * poly[474];
    poly[5733] = poly[12] * poly[474];
    poly[5734] = poly[13] * poly[474];
    poly[5735] = poly[14] * poly[474];
    poly[5736] = poly[15] * poly[474];
    poly[5737] = poly[16] * poly[474];
    poly[5738] = poly[17] * poly[474];
    poly[5739] = poly[28] * poly[235];
    poly[5740] = poly[236] * poly[28];
    poly[5741] = poly[28] * poly[237];
    poly[5742] = poly[1] * poly[475];
    poly[5743] = poly[2] * poly[475];
    poly[5744] = poly[3] * poly[475];
    poly[5745] = poly[4] * poly[475];
    poly[5746] = poly[5] * poly[475];
    poly[5747] = poly[6] * poly[475];
    poly[5748] = poly[7] * poly[475];
    poly[5749] = poly[8] * poly[475];
}

fn f_polynomials115(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5750] = poly[9] * poly[475];
    poly[5751] = poly[10] * poly[475];
    poly[5752] = poly[11] * poly[475];
    poly[5753] = poly[12] * poly[475];
    poly[5754] = poly[13] * poly[475];
    poly[5755] = poly[14] * poly[475];
    poly[5756] = poly[15] * poly[475];
    poly[5757] = poly[16] * poly[475];
    poly[5758] = poly[17] * poly[475];
    poly[5759] = poly[28] * poly[255];
    poly[5760] = poly[28] * poly[256];
    poly[5761] = poly[257] * poly[28];
    poly[5762] = poly[28] * poly[258];
    poly[5763] = poly[28] * poly[259];
    poly[5764] = poly[1] * poly[476];
    poly[5765] = poly[2] * poly[476];
    poly[5766] = poly[3] * poly[476];
    poly[5767] = poly[4] * poly[476];
    poly[5768] = poly[5] * poly[476];
    poly[5769] = poly[6] * poly[476];
    poly[5770] = poly[7] * poly[476];
    poly[5771] = poly[8] * poly[476];
    poly[5772] = poly[9] * poly[476];
    poly[5773] = poly[10] * poly[476];
    poly[5774] = poly[11] * poly[476];
    poly[5775] = poly[12] * poly[476];
    poly[5776] = poly[13] * poly[476];
    poly[5777] = poly[14] * poly[476];
    poly[5778] = poly[15] * poly[476];
    poly[5779] = poly[16] * poly[476];
    poly[5780] = poly[17] * poly[476];
    poly[5781] = poly[28] * poly[277];
    poly[5782] = poly[28] * poly[278];
    poly[5783] = poly[28] * poly[279];
    poly[5784] = poly[280] * poly[28];
    poly[5785] = poly[28] * poly[281];
    poly[5786] = poly[28] * poly[282];
    poly[5787] = poly[28] * poly[283];
    poly[5788] = poly[1] * poly[477];
    poly[5789] = poly[2] * poly[477];
    poly[5790] = poly[3] * poly[477];
    poly[5791] = poly[4] * poly[477];
    poly[5792] = poly[5] * poly[477];
    poly[5793] = poly[6] * poly[477];
    poly[5794] = poly[28] * poly[290];
    poly[5795] = poly[28] * poly[291];
    poly[5796] = poly[28] * poly[292];
    poly[5797] = poly[28] * poly[293];
    poly[5798] = poly[28] * poly[294];
    poly[5799] = poly[28] * poly[295];
}

fn f_polynomials116(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5800] = poly[28] * poly[296];
    poly[5801] = poly[28] * poly[297];
    poly[5802] = poly[11] * poly[477];
    poly[5803] = poly[12] * poly[477];
    poly[5804] = poly[13] * poly[477];
    poly[5805] = poly[14] * poly[477];
    poly[5806] = poly[15] * poly[477];
    poly[5807] = poly[28] * poly[303];
    poly[5808] = poly[28] * poly[304];
    poly[5809] = poly[17] * poly[477];
    poly[5810] = poly[28] * poly[306];
    poly[5811] = poly[28] * poly[307];
    poly[5812] = poly[28] * poly[308];
    poly[5813] = poly[28] * poly[309];
    poly[5814] = poly[28] * poly[310];
    poly[5815] = poly[28] * poly[311];
    poly[5816] = poly[28] * poly[312];
    poly[5817] = poly[28] * poly[313];
    poly[5818] = poly[28] * poly[314];
    poly[5819] = poly[28] * poly[315];
    poly[5820] = poly[28] * poly[316];
    poly[5821] = poly[1] * poly[478];
    poly[5822] = poly[2] * poly[478];
    poly[5823] = poly[3] * poly[478];
    poly[5824] = poly[4] * poly[478];
    poly[5825] = poly[5] * poly[478];
    poly[5826] = poly[6] * poly[478];
    poly[5827] = poly[7] * poly[478];
    poly[5828] = poly[8] * poly[478];
    poly[5829] = poly[9] * poly[478];
    poly[5830] = poly[10] * poly[478];
    poly[5831] = poly[11] * poly[478];
    poly[5832] = poly[28] * poly[328];
    poly[5833] = poly[28] * poly[329];
    poly[5834] = poly[28] * poly[330];
    poly[5835] = poly[28] * poly[331];
    poly[5836] = poly[28] * poly[332];
    poly[5837] = poly[28] * poly[333];
    poly[5838] = poly[28] * poly[334];
    poly[5839] = poly[28] * poly[335];
    poly[5840] = poly[28] * poly[336];
    poly[5841] = poly[28] * poly[337];
    poly[5842] = poly[17] * poly[478];
    poly[5843] = poly[28] * poly[339];
    poly[5844] = poly[28] * poly[340];
    poly[5845] = poly[28] * poly[341];
    poly[5846] = poly[28] * poly[342];
    poly[5847] = poly[28] * poly[343];
    poly[5848] = poly[28] * poly[344];
    poly[5849] = poly[28] * poly[345];
}

fn f_polynomials117(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5850] = poly[28] * poly[346];
    poly[5851] = poly[28] * poly[347];
    poly[5852] = poly[28] * poly[348];
    poly[5853] = poly[28] * poly[349];
    poly[5854] = poly[28] * poly[350];
    poly[5855] = poly[28] * poly[351];
    poly[5856] = poly[1] * poly[479];
    poly[5857] = poly[2] * poly[479];
    poly[5858] = poly[3] * poly[479];
    poly[5859] = poly[4] * poly[479];
    poly[5860] = poly[5] * poly[479];
    poly[5861] = poly[6] * poly[479];
    poly[5862] = poly[24] * poly[462];
    poly[5863] = poly[24] * poly[463];
    poly[5864] = poly[24] * poly[464];
    poly[5865] = poly[24] * poly[465];
    poly[5866] = poly[11] * poly[479];
    poly[5867] = poly[24] * poly[467];
    poly[5868] = poly[24] * poly[468];
    poly[5869] = poly[24] * poly[469];
    poly[5870] = poly[24] * poly[470];
    poly[5871] = poly[24] * poly[471];
    poly[5872] = poly[17] * poly[479];
    poly[5873] = poly[24] * poly[473];
    poly[5874] = poly[24] * poly[474];
    poly[5875] = poly[24] * poly[475];
    poly[5876] = poly[24] * poly[476];
    poly[5877] = poly[24] * poly[477];
    poly[5878] = poly[24] * poly[478];
    poly[5879] = poly[1] * poly[480];
    poly[5880] = poly[2] * poly[480];
    poly[5881] = poly[3] * poly[480];
    poly[5882] = poly[4] * poly[480];
    poly[5883] = poly[5] * poly[480];
    poly[5884] = poly[6] * poly[480];
    poly[5885] = poly[7] * poly[480];
    poly[5886] = poly[8] * poly[480];
    poly[5887] = poly[9] * poly[480];
    poly[5888] = poly[10] * poly[480];
    poly[5889] = poly[11] * poly[480];
    poly[5890] = poly[12] * poly[480];
    poly[5891] = poly[13] * poly[480];
    poly[5892] = poly[14] * poly[480];
    poly[5893] = poly[15] * poly[480];
    poly[5894] = poly[16] * poly[480];
    poly[5895] = poly[17] * poly[480];
    poly[5896] = poly[18] * poly[480];
    poly[5897] = poly[19] * poly[480];
    poly[5898] = poly[20] * poly[480];
    poly[5899] = poly[21] * poly[480];
}

fn f_polynomials118(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5900] = poly[22] * poly[480];
    poly[5901] = poly[23] * poly[480];
    poly[5902] = poly[24] * poly[480];
    poly[5903] = poly[1] * poly[481];
    poly[5904] = poly[2] * poly[481];
    poly[5905] = poly[3] * poly[481];
    poly[5906] = poly[4] * poly[481];
    poly[5907] = poly[5] * poly[481];
    poly[5908] = poly[6] * poly[481];
    poly[5909] = poly[7] * poly[481];
    poly[5910] = poly[8] * poly[481];
    poly[5911] = poly[9] * poly[481];
    poly[5912] = poly[10] * poly[481];
    poly[5913] = poly[11] * poly[481];
    poly[5914] = poly[12] * poly[481];
    poly[5915] = poly[13] * poly[481];
    poly[5916] = poly[14] * poly[481];
    poly[5917] = poly[15] * poly[481];
    poly[5918] = poly[16] * poly[481];
    poly[5919] = poly[17] * poly[481];
    poly[5920] = poly[18] * poly[481];
    poly[5921] = poly[19] * poly[481];
    poly[5922] = poly[20] * poly[481];
    poly[5923] = poly[21] * poly[481];
    poly[5924] = poly[22] * poly[481];
    poly[5925] = poly[23] * poly[481];
    poly[5926] = poly[24] * poly[481];
    poly[5927] = mono[1221] + mono[1222];
    poly[5928] = poly[1] * poly[482];
    poly[5929] = poly[2] * poly[482];
    poly[5930] = poly[3] * poly[482];
    poly[5931] = poly[4] * poly[482];
    poly[5932] = poly[5] * poly[482];
    poly[5933] = poly[6] * poly[482];
    poly[5934] = poly[7] * poly[482];
    poly[5935] = poly[8] * poly[482];
    poly[5936] = poly[9] * poly[482];
    poly[5937] = poly[10] * poly[482];
    poly[5938] = poly[11] * poly[482];
    poly[5939] = poly[12] * poly[482];
    poly[5940] = poly[13] * poly[482];
    poly[5941] = poly[14] * poly[482];
    poly[5942] = poly[15] * poly[482];
    poly[5943] = poly[16] * poly[482];
    poly[5944] = poly[17] * poly[482];
    poly[5945] = poly[18] * poly[482];
    poly[5946] = poly[19] * poly[482];
    poly[5947] = poly[20] * poly[482];
    poly[5948] = poly[21] * poly[482];
    poly[5949] = poly[22] * poly[482];
}

fn f_polynomials119(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[5950] = poly[23] * poly[482];
    poly[5951] = poly[24] * poly[482];
    poly[5952] = mono[1223] + mono[1224];
    poly[5953] = mono[1225] + mono[1226];
    poly[5954] = poly[1] * poly[483];
    poly[5955] = poly[2] * poly[483];
    poly[5956] = poly[3] * poly[483];
    poly[5957] = poly[4] * poly[483];
    poly[5958] = poly[5] * poly[483];
    poly[5959] = poly[6] * poly[483];
    poly[5960] = poly[483] * poly[7];
    poly[5961] = poly[483] * poly[8];
    poly[5962] = poly[483] * poly[9];
    poly[5963] = poly[483] * poly[10];
    poly[5964] = poly[11] * poly[483];
    poly[5965] = poly[483] * poly[12];
    poly[5966] = poly[483] * poly[13];
    poly[5967] = poly[483] * poly[14];
    poly[5968] = poly[483] * poly[15];
    poly[5969] = poly[483] * poly[16];
    poly[5970] = poly[17] * poly[483];
    poly[5971] = poly[483] * poly[18];
    poly[5972] = poly[483] * poly[19];
    poly[5973] = poly[483] * poly[20];
    poly[5974] = poly[483] * poly[21];
    poly[5975] = poly[483] * poly[22];
    poly[5976] = poly[483] * poly[23];
    poly[5977] = poly[24] * poly[483];
    poly[5978] = poly[1] * poly[484];
    poly[5979] = poly[2] * poly[484];
    poly[5980] = poly[3] * poly[484];
    poly[5981] = poly[4] * poly[484];
    poly[5982] = poly[5] * poly[484];
    poly[5983] = poly[6] * poly[484];
    poly[5984] = poly[7] * poly[484];
    poly[5985] = poly[8] * poly[484];
    poly[5986] = poly[9] * poly[484];
    poly[5987] = poly[10] * poly[484];
    poly[5988] = poly[11] * poly[484];
    poly[5989] = poly[12] * poly[484];
    poly[5990] = poly[13] * poly[484];
    poly[5991] = poly[14] * poly[484];
    poly[5992] = poly[15] * poly[484];
    poly[5993] = poly[16] * poly[484];
    poly[5994] = poly[17] * poly[484];
    poly[5995] = poly[18] * poly[484];
    poly[5996] = poly[19] * poly[484];
    poly[5997] = poly[20] * poly[484];
    poly[5998] = poly[21] * poly[484];
    poly[5999] = poly[22] * poly[484];
}

fn f_polynomials120(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6000] = poly[23] * poly[484];
    poly[6001] = poly[24] * poly[484];
    poly[6002] = poly[399] * poly[28];
    poly[6003] = poly[25] * poly[481] - poly[5927];
    poly[6004] = poly[25] * poly[482] - poly[5952];
    poly[6005] = poly[483] * poly[25];
    poly[6006] = poly[1] * poly[485];
    poly[6007] = poly[2] * poly[485];
    poly[6008] = poly[3] * poly[485];
    poly[6009] = poly[4] * poly[485];
    poly[6010] = poly[5] * poly[485];
    poly[6011] = poly[6] * poly[485];
    poly[6012] = poly[7] * poly[485];
    poly[6013] = poly[8] * poly[485];
    poly[6014] = poly[9] * poly[485];
    poly[6015] = poly[10] * poly[485];
    poly[6016] = poly[11] * poly[485];
    poly[6017] = poly[12] * poly[485];
    poly[6018] = poly[13] * poly[485];
    poly[6019] = poly[14] * poly[485];
    poly[6020] = poly[15] * poly[485];
    poly[6021] = poly[16] * poly[485];
    poly[6022] = poly[17] * poly[485];
    poly[6023] = poly[18] * poly[485];
    poly[6024] = poly[19] * poly[485];
    poly[6025] = poly[20] * poly[485];
    poly[6026] = poly[21] * poly[485];
    poly[6027] = poly[22] * poly[485];
    poly[6028] = poly[23] * poly[485];
    poly[6029] = poly[24] * poly[485];
    poly[6030] = poly[26] * poly[480] - poly[5927];
    poly[6031] = poly[425] * poly[28];
    poly[6032] = poly[26] * poly[482] - poly[5953];
    poly[6033] = poly[483] * poly[26];
    poly[6034] = poly[25] * poly[485] - poly[6030];
    poly[6035] = poly[1] * poly[486];
    poly[6036] = poly[2] * poly[486];
    poly[6037] = poly[3] * poly[486];
    poly[6038] = poly[4] * poly[486];
    poly[6039] = poly[5] * poly[486];
    poly[6040] = poly[6] * poly[486];
    poly[6041] = poly[7] * poly[486];
    poly[6042] = poly[8] * poly[486];
    poly[6043] = poly[9] * poly[486];
    poly[6044] = poly[10] * poly[486];
    poly[6045] = poly[11] * poly[486];
    poly[6046] = poly[12] * poly[486];
    poly[6047] = poly[13] * poly[486];
    poly[6048] = poly[14] * poly[486];
    poly[6049] = poly[15] * poly[486];
}

fn f_polynomials121(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6050] = poly[16] * poly[486];
    poly[6051] = poly[17] * poly[486];
    poly[6052] = poly[18] * poly[486];
    poly[6053] = poly[19] * poly[486];
    poly[6054] = poly[20] * poly[486];
    poly[6055] = poly[21] * poly[486];
    poly[6056] = poly[22] * poly[486];
    poly[6057] = poly[23] * poly[486];
    poly[6058] = poly[24] * poly[486];
    poly[6059] = poly[27] * poly[480] - poly[5952];
    poly[6060] = poly[27] * poly[481] - poly[5953];
    poly[6061] = poly[453] * poly[28];
    poly[6062] = poly[483] * poly[27];
    poly[6063] = poly[25] * poly[486] - poly[6059];
    poly[6064] = poly[26] * poly[486] - poly[6060];
    poly[6065] = poly[1] * poly[488];
    poly[6066] = poly[1] * poly[489];
    poly[6067] = poly[2] * poly[489];
    poly[6068] = poly[1] * poly[490];
    poly[6069] = poly[2] * poly[490];
    poly[6070] = poly[3] * poly[490];
    poly[6071] = poly[1] * poly[491];
    poly[6072] = poly[2] * poly[491];
    poly[6073] = poly[3] * poly[491];
    poly[6074] = poly[4] * poly[491];
    poly[6075] = poly[1] * poly[492];
    poly[6076] = poly[2] * poly[492];
    poly[6077] = poly[3] * poly[492];
    poly[6078] = poly[4] * poly[492];
    poly[6079] = poly[5] * poly[492];
    poly[6080] = poly[1] * poly[493];
    poly[6081] = poly[2] * poly[493];
    poly[6082] = poly[3] * poly[493];
    poly[6083] = poly[4] * poly[493];
    poly[6084] = poly[5] * poly[493];
    poly[6085] = poly[6] * poly[493];
    poly[6086] = poly[1] * poly[494];
    poly[6087] = poly[2] * poly[494];
    poly[6088] = poly[3] * poly[494];
    poly[6089] = poly[4] * poly[494];
    poly[6090] = poly[5] * poly[494];
    poly[6091] = poly[6] * poly[494];
    poly[6092] = mono[1227] + mono[1228] + mono[1229] + mono[1230];
    poly[6093] = poly[1] * poly[495];
    poly[6094] = poly[2] * poly[495];
    poly[6095] = poly[3] * poly[495];
    poly[6096] = poly[4] * poly[495];
    poly[6097] = poly[5] * poly[495];
    poly[6098] = poly[6] * poly[495];
    poly[6099] = mono[1231] + mono[1232] + mono[1233] + mono[1234];
}

fn f_polynomials122(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6100] = mono[1235] + mono[1236] + mono[1237] + mono[1238];
    poly[6101] = poly[1] * poly[496];
    poly[6102] = poly[2] * poly[496];
    poly[6103] = poly[3] * poly[496];
    poly[6104] = poly[4] * poly[496];
    poly[6105] = poly[5] * poly[496];
    poly[6106] = poly[6] * poly[496];
    poly[6107] = mono[1239] + mono[1240] + mono[1241] + mono[1242];
    poly[6108] = mono[1243] + mono[1244] + mono[1245] + mono[1246];
    poly[6109] = mono[1247] + mono[1248] + mono[1249] + mono[1250];
    poly[6110] = poly[1] * poly[497];
    poly[6111] = poly[2] * poly[497];
    poly[6112] = poly[3] * poly[497];
    poly[6113] = poly[4] * poly[497];
    poly[6114] = poly[5] * poly[497];
    poly[6115] = poly[6] * poly[497];
    poly[6116] = poly[54] * poly[29];
    poly[6117] = poly[7] * poly[494] - poly[6092];
    poly[6118] = poly[7] * poly[495] - poly[6099];
    poly[6119] = poly[7] * poly[496] - poly[6107];
    poly[6120] = poly[1] * poly[498];
    poly[6121] = poly[2] * poly[498];
    poly[6122] = poly[3] * poly[498];
    poly[6123] = poly[4] * poly[498];
    poly[6124] = poly[5] * poly[498];
    poly[6125] = poly[6] * poly[498];
    poly[6126] = poly[8] * poly[493] - poly[6092];
    poly[6127] = poly[62] * poly[29];
    poly[6128] = poly[8] * poly[495] - poly[6100];
    poly[6129] = poly[8] * poly[496] - poly[6108];
    poly[6130] = poly[7] * poly[498] - poly[6126];
    poly[6131] = poly[1] * poly[499];
    poly[6132] = poly[2] * poly[499];
    poly[6133] = poly[3] * poly[499];
    poly[6134] = poly[4] * poly[499];
    poly[6135] = poly[5] * poly[499];
    poly[6136] = poly[6] * poly[499];
    poly[6137] = poly[9] * poly[493] - poly[6099];
    poly[6138] = poly[9] * poly[494] - poly[6100];
    poly[6139] = poly[72] * poly[29];
    poly[6140] = poly[9] * poly[496] - poly[6109];
    poly[6141] = poly[7] * poly[499] - poly[6137];
    poly[6142] = poly[8] * poly[499] - poly[6138];
    poly[6143] = poly[1] * poly[500];
    poly[6144] = poly[2] * poly[500];
    poly[6145] = poly[3] * poly[500];
    poly[6146] = poly[4] * poly[500];
    poly[6147] = poly[5] * poly[500];
    poly[6148] = poly[6] * poly[500];
    poly[6149] = poly[10] * poly[493] - poly[6107];
}

fn f_polynomials123(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6150] = poly[10] * poly[494] - poly[6108];
    poly[6151] = poly[10] * poly[495] - poly[6109];
    poly[6152] = poly[84] * poly[29];
    poly[6153] = poly[7] * poly[500] - poly[6149];
    poly[6154] = poly[8] * poly[500] - poly[6150];
    poly[6155] = poly[9] * poly[500] - poly[6151];
    poly[6156] = poly[1] * poly[501];
    poly[6157] = poly[2] * poly[501];
    poly[6158] = poly[3] * poly[501];
    poly[6159] = poly[4] * poly[501];
    poly[6160] = poly[5] * poly[501];
    poly[6161] = poly[6] * poly[501];
    poly[6162] = poly[11] * poly[493];
    poly[6163] = poly[11] * poly[494];
    poly[6164] = poly[11] * poly[495];
    poly[6165] = poly[11] * poly[496];
    poly[6166] = poly[11] * poly[497];
    poly[6167] = poly[11] * poly[498];
    poly[6168] = poly[11] * poly[499];
    poly[6169] = poly[11] * poly[500];
    poly[6170] = poly[1] * poly[502];
    poly[6171] = poly[2] * poly[502];
    poly[6172] = poly[3] * poly[502];
    poly[6173] = poly[4] * poly[502];
    poly[6174] = poly[5] * poly[502];
    poly[6175] = poly[6] * poly[502];
    poly[6176] = poly[12] * poly[493];
    poly[6177] = poly[12] * poly[494];
    poly[6178] = poly[12] * poly[495];
    poly[6179] = poly[12] * poly[496];
    poly[6180] = poly[12] * poly[497];
    poly[6181] = poly[12] * poly[498];
    poly[6182] = poly[12] * poly[499];
    poly[6183] = poly[12] * poly[500];
    poly[6184] = poly[11] * poly[502];
    poly[6185] = poly[109] * poly[29];
    poly[6186] = poly[1] * poly[503];
    poly[6187] = poly[2] * poly[503];
    poly[6188] = poly[3] * poly[503];
    poly[6189] = poly[4] * poly[503];
    poly[6190] = poly[5] * poly[503];
    poly[6191] = poly[6] * poly[503];
    poly[6192] = poly[13] * poly[493];
    poly[6193] = poly[13] * poly[494];
    poly[6194] = poly[13] * poly[495];
    poly[6195] = poly[13] * poly[496];
    poly[6196] = poly[13] * poly[497];
    poly[6197] = poly[13] * poly[498];
    poly[6198] = poly[13] * poly[499];
    poly[6199] = poly[13] * poly[500];
}

fn f_polynomials124(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6200] = poly[11] * poly[503];
    poly[6201] = poly[29] * poly[121];
    poly[6202] = poly[122] * poly[29];
    poly[6203] = poly[29] * poly[123];
    poly[6204] = poly[1] * poly[504];
    poly[6205] = poly[2] * poly[504];
    poly[6206] = poly[3] * poly[504];
    poly[6207] = poly[4] * poly[504];
    poly[6208] = poly[5] * poly[504];
    poly[6209] = poly[6] * poly[504];
    poly[6210] = poly[14] * poly[493];
    poly[6211] = poly[14] * poly[494];
    poly[6212] = poly[14] * poly[495];
    poly[6213] = poly[14] * poly[496];
    poly[6214] = poly[14] * poly[497];
    poly[6215] = poly[14] * poly[498];
    poly[6216] = poly[14] * poly[499];
    poly[6217] = poly[14] * poly[500];
    poly[6218] = poly[11] * poly[504];
    poly[6219] = poly[29] * poly[135];
    poly[6220] = poly[29] * poly[136];
    poly[6221] = poly[137] * poly[29];
    poly[6222] = poly[29] * poly[138];
    poly[6223] = poly[29] * poly[139];
    poly[6224] = poly[1] * poly[505];
    poly[6225] = poly[2] * poly[505];
    poly[6226] = poly[3] * poly[505];
    poly[6227] = poly[4] * poly[505];
    poly[6228] = poly[5] * poly[505];
    poly[6229] = poly[6] * poly[505];
    poly[6230] = poly[15] * poly[493];
    poly[6231] = poly[15] * poly[494];
    poly[6232] = poly[15] * poly[495];
    poly[6233] = poly[15] * poly[496];
    poly[6234] = poly[15] * poly[497];
    poly[6235] = poly[15] * poly[498];
    poly[6236] = poly[15] * poly[499];
    poly[6237] = poly[15] * poly[500];
    poly[6238] = poly[11] * poly[505];
    poly[6239] = poly[29] * poly[151];
    poly[6240] = poly[29] * poly[152];
    poly[6241] = poly[29] * poly[153];
    poly[6242] = poly[154] * poly[29];
    poly[6243] = poly[29] * poly[155];
    poly[6244] = poly[29] * poly[156];
    poly[6245] = poly[29] * poly[157];
    poly[6246] = poly[1] * poly[506];
    poly[6247] = poly[2] * poly[506];
    poly[6248] = poly[3] * poly[506];
    poly[6249] = poly[4] * poly[506];
}

fn f_polynomials125(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6250] = poly[5] * poly[506];
    poly[6251] = poly[6] * poly[506];
    poly[6252] = mono[1251]
        + mono[1252]
        + mono[1253]
        + mono[1254]
        + mono[1255]
        + mono[1256]
        + mono[1257]
        + mono[1258];
    poly[6253] = mono[1259]
        + mono[1260]
        + mono[1261]
        + mono[1262]
        + mono[1263]
        + mono[1264]
        + mono[1265]
        + mono[1266];
    poly[6254] = mono[1267]
        + mono[1268]
        + mono[1269]
        + mono[1270]
        + mono[1271]
        + mono[1272]
        + mono[1273]
        + mono[1274];
    poly[6255] = mono[1275]
        + mono[1276]
        + mono[1277]
        + mono[1278]
        + mono[1279]
        + mono[1280]
        + mono[1281]
        + mono[1282];
    poly[6256] = poly[7] * poly[506] - poly[6252];
    poly[6257] = poly[8] * poly[506] - poly[6253];
    poly[6258] = poly[9] * poly[506] - poly[6254];
    poly[6259] = poly[10] * poly[506] - poly[6255];
    poly[6260] = poly[11] * poly[506];
    poly[6261] = mono[1283]
        + mono[1284]
        + mono[1285]
        + mono[1286]
        + mono[1287]
        + mono[1288]
        + mono[1289]
        + mono[1290];
    poly[6262] = mono[1291]
        + mono[1292]
        + mono[1293]
        + mono[1294]
        + mono[1295]
        + mono[1296]
        + mono[1297]
        + mono[1298];
    poly[6263] = mono[1299]
        + mono[1300]
        + mono[1301]
        + mono[1302]
        + mono[1303]
        + mono[1304]
        + mono[1305]
        + mono[1306];
    poly[6264] = mono[1307]
        + mono[1308]
        + mono[1309]
        + mono[1310]
        + mono[1311]
        + mono[1312]
        + mono[1313]
        + mono[1314];
    poly[6265] = mono[1315] + mono[1316] + mono[1317] + mono[1318];
    poly[6266] = poly[12] * poly[506] - poly[6261];
    poly[6267] = poly[13] * poly[506] - poly[6262];
    poly[6268] = poly[14] * poly[506] - poly[6263];
    poly[6269] = poly[15] * poly[506] - poly[6264];
    poly[6270] = poly[1] * poly[507];
    poly[6271] = poly[2] * poly[507];
    poly[6272] = poly[3] * poly[507];
    poly[6273] = poly[4] * poly[507];
    poly[6274] = poly[5] * poly[507];
    poly[6275] = poly[6] * poly[507];
    poly[6276] = poly[16] * poly[493] - poly[6252];
    poly[6277] = poly[16] * poly[494] - poly[6253];
    poly[6278] = poly[16] * poly[495] - poly[6254];
    poly[6279] = poly[16] * poly[496] - poly[6255];
    poly[6280] = poly[7] * poly[507] - poly[6276];
    poly[6281] = poly[8] * poly[507] - poly[6277];
    poly[6282] = poly[9] * poly[507] - poly[6278];
    poly[6283] = poly[10] * poly[507] - poly[6279];
    poly[6284] = poly[11] * poly[507];
    poly[6285] = poly[29] * poly[173] - poly[6261];
    poly[6286] = poly[29] * poly[174] - poly[6262];
    poly[6287] = poly[29] * poly[175] - poly[6263];
    poly[6288] = poly[29] * poly[176] - poly[6264];
    poly[6289] = poly[29] * poly[177];
    poly[6290] = poly[29] * poly[178] - poly[6265];
    poly[6291] = poly[12] * poly[507] - poly[6285];
    poly[6292] = poly[13] * poly[507] - poly[6286];
    poly[6293] = poly[14] * poly[507] - poly[6287];
    poly[6294] = poly[15] * poly[507] - poly[6288];
    poly[6295] = poly[29] * poly[183];
    poly[6296] = poly[1] * poly[508];
    poly[6297] = poly[2] * poly[508];
    poly[6298] = poly[3] * poly[508];
    poly[6299] = poly[4] * poly[508];
}

fn f_polynomials126(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6300] = poly[5] * poly[508];
    poly[6301] = poly[6] * poly[508];
    poly[6302] = poly[17] * poly[493];
    poly[6303] = poly[17] * poly[494];
    poly[6304] = poly[17] * poly[495];
    poly[6305] = poly[17] * poly[496];
    poly[6306] = poly[17] * poly[497];
    poly[6307] = poly[17] * poly[498];
    poly[6308] = poly[17] * poly[499];
    poly[6309] = poly[17] * poly[500];
    poly[6310] = poly[11] * poly[508];
    poly[6311] = poly[17] * poly[502];
    poly[6312] = poly[17] * poly[503];
    poly[6313] = poly[17] * poly[504];
    poly[6314] = poly[17] * poly[505];
    poly[6315] = poly[17] * poly[506];
    poly[6316] = poly[17] * poly[507];
    poly[6317] = poly[1] * poly[509];
    poly[6318] = poly[2] * poly[509];
    poly[6319] = poly[3] * poly[509];
    poly[6320] = poly[4] * poly[509];
    poly[6321] = poly[5] * poly[509];
    poly[6322] = poly[6] * poly[509];
    poly[6323] = poly[18] * poly[493];
    poly[6324] = poly[18] * poly[494];
    poly[6325] = poly[18] * poly[495];
    poly[6326] = poly[18] * poly[496];
    poly[6327] = poly[18] * poly[497];
    poly[6328] = poly[18] * poly[498];
    poly[6329] = poly[18] * poly[499];
    poly[6330] = poly[18] * poly[500];
    poly[6331] = poly[11] * poly[509];
    poly[6332] = poly[12] * poly[509];
    poly[6333] = poly[13] * poly[509];
    poly[6334] = poly[14] * poly[509];
    poly[6335] = poly[15] * poly[509];
    poly[6336] = poly[18] * poly[506];
    poly[6337] = poly[18] * poly[507];
    poly[6338] = poly[17] * poly[509];
    poly[6339] = poly[217] * poly[29];
    poly[6340] = poly[1] * poly[510];
    poly[6341] = poly[2] * poly[510];
    poly[6342] = poly[3] * poly[510];
    poly[6343] = poly[4] * poly[510];
    poly[6344] = poly[5] * poly[510];
    poly[6345] = poly[6] * poly[510];
    poly[6346] = poly[19] * poly[493];
    poly[6347] = poly[19] * poly[494];
    poly[6348] = poly[19] * poly[495];
    poly[6349] = poly[19] * poly[496];
}

fn f_polynomials127(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6350] = poly[19] * poly[497];
    poly[6351] = poly[19] * poly[498];
    poly[6352] = poly[19] * poly[499];
    poly[6353] = poly[19] * poly[500];
    poly[6354] = poly[11] * poly[510];
    poly[6355] = poly[12] * poly[510];
    poly[6356] = poly[13] * poly[510];
    poly[6357] = poly[14] * poly[510];
    poly[6358] = poly[15] * poly[510];
    poly[6359] = poly[19] * poly[506];
    poly[6360] = poly[19] * poly[507];
    poly[6361] = poly[17] * poly[510];
    poly[6362] = poly[29] * poly[235];
    poly[6363] = poly[236] * poly[29];
    poly[6364] = poly[29] * poly[237];
    poly[6365] = poly[1] * poly[511];
    poly[6366] = poly[2] * poly[511];
    poly[6367] = poly[3] * poly[511];
    poly[6368] = poly[4] * poly[511];
    poly[6369] = poly[5] * poly[511];
    poly[6370] = poly[6] * poly[511];
    poly[6371] = poly[20] * poly[493];
    poly[6372] = poly[20] * poly[494];
    poly[6373] = poly[20] * poly[495];
    poly[6374] = poly[20] * poly[496];
    poly[6375] = poly[20] * poly[497];
    poly[6376] = poly[20] * poly[498];
    poly[6377] = poly[20] * poly[499];
    poly[6378] = poly[20] * poly[500];
    poly[6379] = poly[11] * poly[511];
    poly[6380] = poly[12] * poly[511];
    poly[6381] = poly[13] * poly[511];
    poly[6382] = poly[14] * poly[511];
    poly[6383] = poly[15] * poly[511];
    poly[6384] = poly[20] * poly[506];
    poly[6385] = poly[20] * poly[507];
    poly[6386] = poly[17] * poly[511];
    poly[6387] = poly[29] * poly[255];
    poly[6388] = poly[29] * poly[256];
    poly[6389] = poly[257] * poly[29];
    poly[6390] = poly[29] * poly[258];
    poly[6391] = poly[29] * poly[259];
    poly[6392] = poly[1] * poly[512];
    poly[6393] = poly[2] * poly[512];
    poly[6394] = poly[3] * poly[512];
    poly[6395] = poly[4] * poly[512];
    poly[6396] = poly[5] * poly[512];
    poly[6397] = poly[6] * poly[512];
    poly[6398] = poly[21] * poly[493];
    poly[6399] = poly[21] * poly[494];
}

fn f_polynomials128(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6400] = poly[21] * poly[495];
    poly[6401] = poly[21] * poly[496];
    poly[6402] = poly[21] * poly[497];
    poly[6403] = poly[21] * poly[498];
    poly[6404] = poly[21] * poly[499];
    poly[6405] = poly[21] * poly[500];
    poly[6406] = poly[11] * poly[512];
    poly[6407] = poly[12] * poly[512];
    poly[6408] = poly[13] * poly[512];
    poly[6409] = poly[14] * poly[512];
    poly[6410] = poly[15] * poly[512];
    poly[6411] = poly[21] * poly[506];
    poly[6412] = poly[21] * poly[507];
    poly[6413] = poly[17] * poly[512];
    poly[6414] = poly[29] * poly[277];
    poly[6415] = poly[29] * poly[278];
    poly[6416] = poly[29] * poly[279];
    poly[6417] = poly[280] * poly[29];
    poly[6418] = poly[29] * poly[281];
    poly[6419] = poly[29] * poly[282];
    poly[6420] = poly[29] * poly[283];
    poly[6421] = poly[1] * poly[513];
    poly[6422] = poly[2] * poly[513];
    poly[6423] = poly[3] * poly[513];
    poly[6424] = poly[4] * poly[513];
    poly[6425] = poly[5] * poly[513];
    poly[6426] = poly[6] * poly[513];
    poly[6427] = mono[1319]
        + mono[1320]
        + mono[1321]
        + mono[1322]
        + mono[1323]
        + mono[1324]
        + mono[1325]
        + mono[1326];
    poly[6428] = mono[1327]
        + mono[1328]
        + mono[1329]
        + mono[1330]
        + mono[1331]
        + mono[1332]
        + mono[1333]
        + mono[1334];
    poly[6429] = mono[1335]
        + mono[1336]
        + mono[1337]
        + mono[1338]
        + mono[1339]
        + mono[1340]
        + mono[1341]
        + mono[1342];
    poly[6430] = mono[1343]
        + mono[1344]
        + mono[1345]
        + mono[1346]
        + mono[1347]
        + mono[1348]
        + mono[1349]
        + mono[1350];
    poly[6431] = poly[7] * poly[513] - poly[6427];
    poly[6432] = poly[8] * poly[513] - poly[6428];
    poly[6433] = poly[9] * poly[513] - poly[6429];
    poly[6434] = poly[10] * poly[513] - poly[6430];
    poly[6435] = poly[11] * poly[513];
    poly[6436] = poly[12] * poly[513];
    poly[6437] = poly[13] * poly[513];
    poly[6438] = poly[14] * poly[513];
    poly[6439] = poly[15] * poly[513];
    poly[6440] = mono[1351]
        + mono[1352]
        + mono[1353]
        + mono[1354]
        + mono[1355]
        + mono[1356]
        + mono[1357]
        + mono[1358]
        + mono[1359]
        + mono[1360]
        + mono[1361]
        + mono[1362]
        + mono[1363]
        + mono[1364]
        + mono[1365]
        + mono[1366];
    poly[6441] = poly[16] * poly[513] - poly[6440];
    poly[6442] = poly[17] * poly[513];
    poly[6443] = mono[1367]
        + mono[1368]
        + mono[1369]
        + mono[1370]
        + mono[1371]
        + mono[1372]
        + mono[1373]
        + mono[1374];
    poly[6444] = mono[1375]
        + mono[1376]
        + mono[1377]
        + mono[1378]
        + mono[1379]
        + mono[1380]
        + mono[1381]
        + mono[1382];
    poly[6445] = mono[1383]
        + mono[1384]
        + mono[1385]
        + mono[1386]
        + mono[1387]
        + mono[1388]
        + mono[1389]
        + mono[1390];
    poly[6446] = mono[1391]
        + mono[1392]
        + mono[1393]
        + mono[1394]
        + mono[1395]
        + mono[1396]
        + mono[1397]
        + mono[1398];
    poly[6447] = mono[1399] + mono[1400] + mono[1401] + mono[1402];
    poly[6448] = poly[18] * poly[513] - poly[6443];
    poly[6449] = poly[19] * poly[513] - poly[6444];
}

fn f_polynomials129(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6450] = poly[20] * poly[513] - poly[6445];
    poly[6451] = poly[21] * poly[513] - poly[6446];
    poly[6452] = poly[1] * poly[514];
    poly[6453] = poly[2] * poly[514];
    poly[6454] = poly[3] * poly[514];
    poly[6455] = poly[4] * poly[514];
    poly[6456] = poly[5] * poly[514];
    poly[6457] = poly[6] * poly[514];
    poly[6458] = poly[22] * poly[493] - poly[6427];
    poly[6459] = poly[22] * poly[494] - poly[6428];
    poly[6460] = poly[22] * poly[495] - poly[6429];
    poly[6461] = poly[22] * poly[496] - poly[6430];
    poly[6462] = poly[7] * poly[514] - poly[6458];
    poly[6463] = poly[8] * poly[514] - poly[6459];
    poly[6464] = poly[9] * poly[514] - poly[6460];
    poly[6465] = poly[10] * poly[514] - poly[6461];
    poly[6466] = poly[11] * poly[514];
    poly[6467] = poly[12] * poly[514];
    poly[6468] = poly[13] * poly[514];
    poly[6469] = poly[14] * poly[514];
    poly[6470] = poly[15] * poly[514];
    poly[6471] = poly[22] * poly[506] - poly[6440];
    poly[6472] = poly[16] * poly[514] - poly[6471];
    poly[6473] = poly[17] * poly[514];
    poly[6474] = poly[29] * poly[306] - poly[6443];
    poly[6475] = poly[29] * poly[307] - poly[6444];
    poly[6476] = poly[29] * poly[308] - poly[6445];
    poly[6477] = poly[29] * poly[309] - poly[6446];
    poly[6478] = poly[29] * poly[310];
    poly[6479] = poly[29] * poly[311] - poly[6447];
    poly[6480] = poly[18] * poly[514] - poly[6474];
    poly[6481] = poly[19] * poly[514] - poly[6475];
    poly[6482] = poly[20] * poly[514] - poly[6476];
    poly[6483] = poly[21] * poly[514] - poly[6477];
    poly[6484] = poly[29] * poly[316];
    poly[6485] = poly[1] * poly[515];
    poly[6486] = poly[2] * poly[515];
    poly[6487] = poly[3] * poly[515];
    poly[6488] = poly[4] * poly[515];
    poly[6489] = poly[5] * poly[515];
    poly[6490] = poly[6] * poly[515];
    poly[6491] = poly[23] * poly[493];
    poly[6492] = poly[23] * poly[494];
    poly[6493] = poly[23] * poly[495];
    poly[6494] = poly[23] * poly[496];
    poly[6495] = poly[23] * poly[497];
    poly[6496] = poly[23] * poly[498];
    poly[6497] = poly[23] * poly[499];
    poly[6498] = poly[23] * poly[500];
    poly[6499] = poly[11] * poly[515];
}

fn f_polynomials130(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6500] = poly[29] * poly[328];
    poly[6501] = poly[29] * poly[329];
    poly[6502] = poly[29] * poly[330];
    poly[6503] = poly[29] * poly[331];
    poly[6504] = mono[1403]
        + mono[1404]
        + mono[1405]
        + mono[1406]
        + mono[1407]
        + mono[1408]
        + mono[1409]
        + mono[1410]
        + mono[1411]
        + mono[1412]
        + mono[1413]
        + mono[1414]
        + mono[1415]
        + mono[1416]
        + mono[1417]
        + mono[1418];
    poly[6505] = poly[29] * poly[332] - poly[6504];
    poly[6506] = poly[29] * poly[333];
    poly[6507] = poly[29] * poly[334];
    poly[6508] = poly[29] * poly[335];
    poly[6509] = poly[29] * poly[336];
    poly[6510] = poly[23] * poly[506] - poly[6504];
    poly[6511] = poly[23] * poly[507] - poly[6505];
    poly[6512] = poly[17] * poly[515];
    poly[6513] = poly[29] * poly[339];
    poly[6514] = poly[29] * poly[340];
    poly[6515] = poly[29] * poly[341];
    poly[6516] = poly[29] * poly[342];
    poly[6517] = mono[1419]
        + mono[1420]
        + mono[1421]
        + mono[1422]
        + mono[1423]
        + mono[1424]
        + mono[1425]
        + mono[1426]
        + mono[1427]
        + mono[1428]
        + mono[1429]
        + mono[1430]
        + mono[1431]
        + mono[1432]
        + mono[1433]
        + mono[1434];
    poly[6518] = poly[29] * poly[343] - poly[6517];
    poly[6519] = poly[29] * poly[344];
    poly[6520] = poly[29] * poly[345];
    poly[6521] = poly[29] * poly[346];
    poly[6522] = poly[29] * poly[347];
    poly[6523] = poly[29] * poly[348];
    poly[6524] = poly[29] * poly[349];
    poly[6525] = poly[23] * poly[513] - poly[6517];
    poly[6526] = poly[23] * poly[514] - poly[6518];
    poly[6527] = poly[29] * poly[351];
    poly[6528] = poly[1] * poly[516];
    poly[6529] = poly[2] * poly[516];
    poly[6530] = poly[3] * poly[516];
    poly[6531] = poly[4] * poly[516];
    poly[6532] = poly[5] * poly[516];
    poly[6533] = poly[6] * poly[516];
    poly[6534] = poly[24] * poly[493];
    poly[6535] = poly[24] * poly[494];
    poly[6536] = poly[24] * poly[495];
    poly[6537] = poly[24] * poly[496];
    poly[6538] = poly[24] * poly[497];
    poly[6539] = poly[24] * poly[498];
    poly[6540] = poly[24] * poly[499];
    poly[6541] = poly[24] * poly[500];
    poly[6542] = poly[11] * poly[516];
    poly[6543] = poly[24] * poly[502];
    poly[6544] = poly[24] * poly[503];
    poly[6545] = poly[24] * poly[504];
    poly[6546] = poly[24] * poly[505];
    poly[6547] = poly[24] * poly[506];
    poly[6548] = poly[24] * poly[507];
    poly[6549] = poly[17] * poly[516];
}

fn f_polynomials131(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6550] = poly[24] * poly[509];
    poly[6551] = poly[24] * poly[510];
    poly[6552] = poly[24] * poly[511];
    poly[6553] = poly[24] * poly[512];
    poly[6554] = poly[24] * poly[513];
    poly[6555] = poly[24] * poly[514];
    poly[6556] = poly[24] * poly[515];
    poly[6557] = poly[1] * poly[517];
    poly[6558] = poly[2] * poly[517];
    poly[6559] = poly[3] * poly[517];
    poly[6560] = poly[4] * poly[517];
    poly[6561] = poly[5] * poly[517];
    poly[6562] = poly[6] * poly[517];
    poly[6563] = mono[1435] + mono[1436] + mono[1437] + mono[1438];
    poly[6564] = mono[1439] + mono[1440] + mono[1441] + mono[1442];
    poly[6565] = mono[1443] + mono[1444] + mono[1445] + mono[1446];
    poly[6566] = mono[1447] + mono[1448] + mono[1449] + mono[1450];
    poly[6567] = poly[7] * poly[517] - poly[6563];
    poly[6568] = poly[8] * poly[517] - poly[6564];
    poly[6569] = poly[9] * poly[517] - poly[6565];
    poly[6570] = poly[10] * poly[517] - poly[6566];
    poly[6571] = poly[11] * poly[517];
    poly[6572] = poly[12] * poly[517];
    poly[6573] = poly[13] * poly[517];
    poly[6574] = poly[14] * poly[517];
    poly[6575] = poly[15] * poly[517];
    poly[6576] = mono[1451]
        + mono[1452]
        + mono[1453]
        + mono[1454]
        + mono[1455]
        + mono[1456]
        + mono[1457]
        + mono[1458];
    poly[6577] = poly[16] * poly[517] - poly[6576];
    poly[6578] = poly[17] * poly[517];
    poly[6579] = poly[18] * poly[517];
    poly[6580] = poly[19] * poly[517];
    poly[6581] = poly[20] * poly[517];
    poly[6582] = poly[21] * poly[517];
    poly[6583] = mono[1459]
        + mono[1460]
        + mono[1461]
        + mono[1462]
        + mono[1463]
        + mono[1464]
        + mono[1465]
        + mono[1466];
    poly[6584] = poly[22] * poly[517] - poly[6583];
    poly[6585] = poly[23] * poly[517];
    poly[6586] = poly[24] * poly[517];
    poly[6587] = poly[1] * poly[518];
    poly[6588] = poly[2] * poly[518];
    poly[6589] = poly[3] * poly[518];
    poly[6590] = poly[4] * poly[518];
    poly[6591] = poly[5] * poly[518];
    poly[6592] = poly[6] * poly[518];
    poly[6593] = mono[1467] + mono[1468] + mono[1469] + mono[1470];
    poly[6594] = mono[1471] + mono[1472] + mono[1473] + mono[1474];
    poly[6595] = mono[1475] + mono[1476] + mono[1477] + mono[1478];
    poly[6596] = mono[1479] + mono[1480] + mono[1481] + mono[1482];
    poly[6597] = poly[7] * poly[518] - poly[6593];
    poly[6598] = poly[8] * poly[518] - poly[6594];
    poly[6599] = poly[9] * poly[518] - poly[6595];
}

fn f_polynomials132(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6600] = poly[10] * poly[518] - poly[6596];
    poly[6601] = poly[11] * poly[518];
    poly[6602] = poly[12] * poly[518];
    poly[6603] = poly[13] * poly[518];
    poly[6604] = poly[14] * poly[518];
    poly[6605] = poly[15] * poly[518];
    poly[6606] = mono[1483]
        + mono[1484]
        + mono[1485]
        + mono[1486]
        + mono[1487]
        + mono[1488]
        + mono[1489]
        + mono[1490];
    poly[6607] = poly[16] * poly[518] - poly[6606];
    poly[6608] = poly[17] * poly[518];
    poly[6609] = poly[18] * poly[518];
    poly[6610] = poly[19] * poly[518];
    poly[6611] = poly[20] * poly[518];
    poly[6612] = poly[21] * poly[518];
    poly[6613] = mono[1491]
        + mono[1492]
        + mono[1493]
        + mono[1494]
        + mono[1495]
        + mono[1496]
        + mono[1497]
        + mono[1498];
    poly[6614] = poly[22] * poly[518] - poly[6613];
    poly[6615] = poly[23] * poly[518];
    poly[6616] = poly[24] * poly[518];
    poly[6617] = mono[1499] + mono[1500] + mono[1501] + mono[1502];
    poly[6618] = poly[1] * poly[519];
    poly[6619] = poly[2] * poly[519];
    poly[6620] = poly[3] * poly[519];
    poly[6621] = poly[4] * poly[519];
    poly[6622] = poly[5] * poly[519];
    poly[6623] = poly[6] * poly[519];
    poly[6624] = mono[1503] + mono[1504] + mono[1505] + mono[1506];
    poly[6625] = mono[1507] + mono[1508] + mono[1509] + mono[1510];
    poly[6626] = mono[1511] + mono[1512] + mono[1513] + mono[1514];
    poly[6627] = mono[1515] + mono[1516] + mono[1517] + mono[1518];
    poly[6628] = poly[7] * poly[519] - poly[6624];
    poly[6629] = poly[8] * poly[519] - poly[6625];
    poly[6630] = poly[9] * poly[519] - poly[6626];
    poly[6631] = poly[10] * poly[519] - poly[6627];
    poly[6632] = poly[11] * poly[519];
    poly[6633] = poly[12] * poly[519];
    poly[6634] = poly[13] * poly[519];
    poly[6635] = poly[14] * poly[519];
    poly[6636] = poly[15] * poly[519];
    poly[6637] = mono[1519]
        + mono[1520]
        + mono[1521]
        + mono[1522]
        + mono[1523]
        + mono[1524]
        + mono[1525]
        + mono[1526];
    poly[6638] = poly[16] * poly[519] - poly[6637];
    poly[6639] = poly[17] * poly[519];
    poly[6640] = poly[18] * poly[519];
    poly[6641] = poly[19] * poly[519];
    poly[6642] = poly[20] * poly[519];
    poly[6643] = poly[21] * poly[519];
    poly[6644] = mono[1527]
        + mono[1528]
        + mono[1529]
        + mono[1530]
        + mono[1531]
        + mono[1532]
        + mono[1533]
        + mono[1534];
    poly[6645] = poly[22] * poly[519] - poly[6644];
    poly[6646] = poly[23] * poly[519];
    poly[6647] = poly[24] * poly[519];
    poly[6648] = mono[1535] + mono[1536] + mono[1537] + mono[1538];
    poly[6649] = mono[1539] + mono[1540] + mono[1541] + mono[1542];
}

fn f_polynomials133(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6650] = poly[1] * poly[520];
    poly[6651] = poly[2] * poly[520];
    poly[6652] = poly[3] * poly[520];
    poly[6653] = poly[4] * poly[520];
    poly[6654] = poly[5] * poly[520];
    poly[6655] = poly[6] * poly[520];
    poly[6656] = mono[1543] + mono[1544] + mono[1545] + mono[1546];
    poly[6657] = mono[1547] + mono[1548] + mono[1549] + mono[1550];
    poly[6658] = mono[1551] + mono[1552] + mono[1553] + mono[1554];
    poly[6659] = mono[1555] + mono[1556] + mono[1557] + mono[1558];
    poly[6660] = poly[7] * poly[520] - poly[6656];
    poly[6661] = poly[8] * poly[520] - poly[6657];
    poly[6662] = poly[9] * poly[520] - poly[6658];
    poly[6663] = poly[10] * poly[520] - poly[6659];
    poly[6664] = poly[11] * poly[520];
    poly[6665] = poly[12] * poly[520];
    poly[6666] = poly[13] * poly[520];
    poly[6667] = poly[14] * poly[520];
    poly[6668] = poly[15] * poly[520];
    poly[6669] = mono[1559]
        + mono[1560]
        + mono[1561]
        + mono[1562]
        + mono[1563]
        + mono[1564]
        + mono[1565]
        + mono[1566];
    poly[6670] = poly[16] * poly[520] - poly[6669];
    poly[6671] = poly[17] * poly[520];
    poly[6672] = poly[18] * poly[520];
    poly[6673] = poly[19] * poly[520];
    poly[6674] = poly[20] * poly[520];
    poly[6675] = poly[21] * poly[520];
    poly[6676] = mono[1567]
        + mono[1568]
        + mono[1569]
        + mono[1570]
        + mono[1571]
        + mono[1572]
        + mono[1573]
        + mono[1574];
    poly[6677] = poly[22] * poly[520] - poly[6676];
    poly[6678] = poly[23] * poly[520];
    poly[6679] = poly[24] * poly[520];
    poly[6680] = mono[1575] + mono[1576] + mono[1577] + mono[1578];
    poly[6681] = mono[1579] + mono[1580] + mono[1581] + mono[1582];
    poly[6682] = mono[1583] + mono[1584] + mono[1585] + mono[1586];
    poly[6683] = poly[1] * poly[521];
    poly[6684] = poly[2] * poly[521];
    poly[6685] = poly[3] * poly[521];
    poly[6686] = poly[4] * poly[521];
    poly[6687] = poly[5] * poly[521];
    poly[6688] = poly[6] * poly[521];
    poly[6689] = poly[7] * poly[521];
    poly[6690] = poly[8] * poly[521];
    poly[6691] = poly[9] * poly[521];
    poly[6692] = poly[10] * poly[521];
    poly[6693] = poly[11] * poly[521];
    poly[6694] = poly[12] * poly[521];
    poly[6695] = poly[13] * poly[521];
    poly[6696] = poly[14] * poly[521];
    poly[6697] = poly[15] * poly[521];
    poly[6698] = poly[16] * poly[521];
    poly[6699] = poly[17] * poly[521];
}

fn f_polynomials134(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6700] = poly[18] * poly[521];
    poly[6701] = poly[19] * poly[521];
    poly[6702] = poly[20] * poly[521];
    poly[6703] = poly[21] * poly[521];
    poly[6704] = poly[22] * poly[521];
    poly[6705] = poly[23] * poly[521];
    poly[6706] = poly[24] * poly[521];
    poly[6707] = poly[1] * poly[522];
    poly[6708] = poly[2] * poly[522];
    poly[6709] = poly[3] * poly[522];
    poly[6710] = poly[4] * poly[522];
    poly[6711] = poly[5] * poly[522];
    poly[6712] = poly[6] * poly[522];
    poly[6713] = mono[1587] + mono[1588];
    poly[6714] = mono[1589] + mono[1590];
    poly[6715] = mono[1591] + mono[1592];
    poly[6716] = mono[1593] + mono[1594];
    poly[6717] = poly[7] * poly[522] - poly[6713];
    poly[6718] = poly[8] * poly[522] - poly[6714];
    poly[6719] = poly[9] * poly[522] - poly[6715];
    poly[6720] = poly[10] * poly[522] - poly[6716];
    poly[6721] = poly[11] * poly[522];
    poly[6722] = poly[12] * poly[522];
    poly[6723] = poly[13] * poly[522];
    poly[6724] = poly[14] * poly[522];
    poly[6725] = poly[15] * poly[522];
    poly[6726] = mono[1595] + mono[1596] + mono[1597] + mono[1598];
    poly[6727] = poly[16] * poly[522] - poly[6726];
    poly[6728] = poly[17] * poly[522];
    poly[6729] = poly[18] * poly[522];
    poly[6730] = poly[19] * poly[522];
    poly[6731] = poly[20] * poly[522];
    poly[6732] = poly[21] * poly[522];
    poly[6733] = mono[1599] + mono[1600] + mono[1601] + mono[1602];
    poly[6734] = poly[22] * poly[522] - poly[6733];
    poly[6735] = poly[23] * poly[522];
    poly[6736] = poly[24] * poly[522];
    poly[6737] = poly[1] * poly[523];
    poly[6738] = poly[2] * poly[523];
    poly[6739] = poly[3] * poly[523];
    poly[6740] = poly[4] * poly[523];
    poly[6741] = poly[5] * poly[523];
    poly[6742] = poly[6] * poly[523];
    poly[6743] = poly[25] * poly[493] - poly[6563];
    poly[6744] = poly[25] * poly[494] - poly[6564];
    poly[6745] = poly[25] * poly[495] - poly[6565];
    poly[6746] = poly[25] * poly[496] - poly[6566];
    poly[6747] = poly[7] * poly[523] - poly[6743];
    poly[6748] = poly[8] * poly[523] - poly[6744];
    poly[6749] = poly[9] * poly[523] - poly[6745];
}

fn f_polynomials135(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6750] = poly[10] * poly[523] - poly[6746];
    poly[6751] = poly[11] * poly[523];
    poly[6752] = poly[12] * poly[523];
    poly[6753] = poly[13] * poly[523];
    poly[6754] = poly[14] * poly[523];
    poly[6755] = poly[15] * poly[523];
    poly[6756] = poly[25] * poly[506] - poly[6576];
    poly[6757] = poly[16] * poly[523] - poly[6756];
    poly[6758] = poly[17] * poly[523];
    poly[6759] = poly[18] * poly[523];
    poly[6760] = poly[19] * poly[523];
    poly[6761] = poly[20] * poly[523];
    poly[6762] = poly[21] * poly[523];
    poly[6763] = poly[25] * poly[513] - poly[6583];
    poly[6764] = poly[22] * poly[523] - poly[6763];
    poly[6765] = poly[23] * poly[523];
    poly[6766] = poly[24] * poly[523];
    poly[6767] = poly[399] * poly[29];
    poly[6768] = poly[25] * poly[518] - poly[6617];
    poly[6769] = poly[25] * poly[519] - poly[6648];
    poly[6770] = poly[25] * poly[520] - poly[6680];
    poly[6771] = poly[25] * poly[521];
    poly[6772] = poly[25] * poly[522];
    poly[6773] = poly[1] * poly[524];
    poly[6774] = poly[2] * poly[524];
    poly[6775] = poly[3] * poly[524];
    poly[6776] = poly[4] * poly[524];
    poly[6777] = poly[5] * poly[524];
    poly[6778] = poly[6] * poly[524];
    poly[6779] = poly[26] * poly[493] - poly[6593];
    poly[6780] = poly[26] * poly[494] - poly[6594];
    poly[6781] = poly[26] * poly[495] - poly[6595];
    poly[6782] = poly[26] * poly[496] - poly[6596];
    poly[6783] = poly[7] * poly[524] - poly[6779];
    poly[6784] = poly[8] * poly[524] - poly[6780];
    poly[6785] = poly[9] * poly[524] - poly[6781];
    poly[6786] = poly[10] * poly[524] - poly[6782];
    poly[6787] = poly[11] * poly[524];
    poly[6788] = poly[12] * poly[524];
    poly[6789] = poly[13] * poly[524];
    poly[6790] = poly[14] * poly[524];
    poly[6791] = poly[15] * poly[524];
    poly[6792] = poly[26] * poly[506] - poly[6606];
    poly[6793] = poly[16] * poly[524] - poly[6792];
    poly[6794] = poly[17] * poly[524];
    poly[6795] = poly[18] * poly[524];
    poly[6796] = poly[19] * poly[524];
    poly[6797] = poly[20] * poly[524];
    poly[6798] = poly[21] * poly[524];
    poly[6799] = poly[26] * poly[513] - poly[6613];
}

fn f_polynomials136(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6800] = poly[22] * poly[524] - poly[6799];
    poly[6801] = poly[23] * poly[524];
    poly[6802] = poly[24] * poly[524];
    poly[6803] = poly[26] * poly[517] - poly[6617];
    poly[6804] = poly[425] * poly[29];
    poly[6805] = poly[26] * poly[519] - poly[6649];
    poly[6806] = poly[26] * poly[520] - poly[6681];
    poly[6807] = poly[26] * poly[521];
    poly[6808] = poly[26] * poly[522];
    poly[6809] = poly[25] * poly[524] - poly[6803];
    poly[6810] = poly[1] * poly[525];
    poly[6811] = poly[2] * poly[525];
    poly[6812] = poly[3] * poly[525];
    poly[6813] = poly[4] * poly[525];
    poly[6814] = poly[5] * poly[525];
    poly[6815] = poly[6] * poly[525];
    poly[6816] = poly[27] * poly[493] - poly[6624];
    poly[6817] = poly[27] * poly[494] - poly[6625];
    poly[6818] = poly[27] * poly[495] - poly[6626];
    poly[6819] = poly[27] * poly[496] - poly[6627];
    poly[6820] = poly[7] * poly[525] - poly[6816];
    poly[6821] = poly[8] * poly[525] - poly[6817];
    poly[6822] = poly[9] * poly[525] - poly[6818];
    poly[6823] = poly[10] * poly[525] - poly[6819];
    poly[6824] = poly[11] * poly[525];
    poly[6825] = poly[12] * poly[525];
    poly[6826] = poly[13] * poly[525];
    poly[6827] = poly[14] * poly[525];
    poly[6828] = poly[15] * poly[525];
    poly[6829] = poly[27] * poly[506] - poly[6637];
    poly[6830] = poly[16] * poly[525] - poly[6829];
    poly[6831] = poly[17] * poly[525];
    poly[6832] = poly[18] * poly[525];
    poly[6833] = poly[19] * poly[525];
    poly[6834] = poly[20] * poly[525];
    poly[6835] = poly[21] * poly[525];
    poly[6836] = poly[27] * poly[513] - poly[6644];
    poly[6837] = poly[22] * poly[525] - poly[6836];
    poly[6838] = poly[23] * poly[525];
    poly[6839] = poly[24] * poly[525];
    poly[6840] = poly[27] * poly[517] - poly[6648];
    poly[6841] = poly[27] * poly[518] - poly[6649];
    poly[6842] = poly[453] * poly[29];
    poly[6843] = poly[27] * poly[520] - poly[6682];
    poly[6844] = poly[27] * poly[521];
    poly[6845] = poly[27] * poly[522];
    poly[6846] = poly[25] * poly[525] - poly[6840];
    poly[6847] = poly[26] * poly[525] - poly[6841];
    poly[6848] = poly[1] * poly[526];
    poly[6849] = poly[2] * poly[526];
}

fn f_polynomials137(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6850] = poly[3] * poly[526];
    poly[6851] = poly[4] * poly[526];
    poly[6852] = poly[5] * poly[526];
    poly[6853] = poly[6] * poly[526];
    poly[6854] = poly[28] * poly[493] - poly[6656];
    poly[6855] = poly[28] * poly[494] - poly[6657];
    poly[6856] = poly[28] * poly[495] - poly[6658];
    poly[6857] = poly[28] * poly[496] - poly[6659];
    poly[6858] = poly[7] * poly[526] - poly[6854];
    poly[6859] = poly[8] * poly[526] - poly[6855];
    poly[6860] = poly[9] * poly[526] - poly[6856];
    poly[6861] = poly[10] * poly[526] - poly[6857];
    poly[6862] = poly[11] * poly[526];
    poly[6863] = poly[12] * poly[526];
    poly[6864] = poly[13] * poly[526];
    poly[6865] = poly[14] * poly[526];
    poly[6866] = poly[15] * poly[526];
    poly[6867] = poly[28] * poly[506] - poly[6669];
    poly[6868] = poly[16] * poly[526] - poly[6867];
    poly[6869] = poly[17] * poly[526];
    poly[6870] = poly[18] * poly[526];
    poly[6871] = poly[19] * poly[526];
    poly[6872] = poly[20] * poly[526];
    poly[6873] = poly[21] * poly[526];
    poly[6874] = poly[28] * poly[513] - poly[6676];
    poly[6875] = poly[22] * poly[526] - poly[6874];
    poly[6876] = poly[23] * poly[526];
    poly[6877] = poly[24] * poly[526];
    poly[6878] = poly[28] * poly[517] - poly[6680];
    poly[6879] = poly[28] * poly[518] - poly[6681];
    poly[6880] = poly[28] * poly[519] - poly[6682];
    poly[6881] = poly[483] * poly[29];
    poly[6882] = poly[28] * poly[521];
    poly[6883] = poly[28] * poly[522];
    poly[6884] = poly[25] * poly[526] - poly[6878];
    poly[6885] = poly[26] * poly[526] - poly[6879];
    poly[6886] = poly[27] * poly[526] - poly[6880];
    poly[6887] = poly[1] * poly[527];
    poly[6888] = poly[2] * poly[527];
    poly[6889] = poly[3] * poly[527];
    poly[6890] = poly[4] * poly[527];
    poly[6891] = poly[5] * poly[527];
    poly[6892] = poly[6] * poly[527];
    poly[6893] = poly[7] * poly[527];
    poly[6894] = poly[8] * poly[527];
    poly[6895] = poly[9] * poly[527];
    poly[6896] = poly[10] * poly[527];
    poly[6897] = poly[11] * poly[527];
    poly[6898] = poly[12] * poly[527];
    poly[6899] = poly[13] * poly[527];
}

fn f_polynomials138(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6900] = poly[14] * poly[527];
    poly[6901] = poly[15] * poly[527];
    poly[6902] = poly[16] * poly[527];
    poly[6903] = poly[17] * poly[527];
    poly[6904] = poly[18] * poly[527];
    poly[6905] = poly[19] * poly[527];
    poly[6906] = poly[20] * poly[527];
    poly[6907] = poly[21] * poly[527];
    poly[6908] = poly[22] * poly[527];
    poly[6909] = poly[23] * poly[527];
    poly[6910] = poly[24] * poly[527];
    poly[6911] = mono[1603] + mono[1604];
    poly[6912] = mono[1605] + mono[1606];
    poly[6913] = mono[1607] + mono[1608];
    poly[6914] = mono[1609] + mono[1610];
    poly[6915] = mono[1611] + mono[1612] + mono[1613] + mono[1614];
    poly[6916] = poly[25] * poly[527] - poly[6911];
    poly[6917] = poly[26] * poly[527] - poly[6912];
    poly[6918] = poly[27] * poly[527] - poly[6913];
    poly[6919] = poly[28] * poly[527] - poly[6914];
    poly[6920] = poly[1] * poly[529];
    poly[6921] = poly[1] * poly[530];
    poly[6922] = poly[2] * poly[530];
    poly[6923] = poly[1] * poly[531];
    poly[6924] = poly[2] * poly[531];
    poly[6925] = poly[3] * poly[531];
    poly[6926] = poly[1] * poly[532];
    poly[6927] = poly[2] * poly[532];
    poly[6928] = poly[3] * poly[532];
    poly[6929] = poly[4] * poly[532];
    poly[6930] = poly[1] * poly[533];
    poly[6931] = poly[2] * poly[533];
    poly[6932] = poly[3] * poly[533];
    poly[6933] = poly[4] * poly[533];
    poly[6934] = poly[5] * poly[533];
    poly[6935] = poly[1] * poly[534];
    poly[6936] = poly[2] * poly[534];
    poly[6937] = poly[3] * poly[534];
    poly[6938] = poly[4] * poly[534];
    poly[6939] = poly[5] * poly[534];
    poly[6940] = poly[6] * poly[534];
    poly[6941] = poly[54] * poly[30];
    poly[6942] = poly[1] * poly[535];
    poly[6943] = poly[2] * poly[535];
    poly[6944] = poly[3] * poly[535];
    poly[6945] = poly[4] * poly[535];
    poly[6946] = poly[5] * poly[535];
    poly[6947] = poly[6] * poly[535];
    poly[6948] = poly[30] * poly[61];
    poly[6949] = poly[62] * poly[30];
}

fn f_polynomials139(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[6950] = poly[30] * poly[63];
    poly[6951] = poly[1] * poly[536];
    poly[6952] = poly[2] * poly[536];
    poly[6953] = poly[3] * poly[536];
    poly[6954] = poly[4] * poly[536];
    poly[6955] = poly[5] * poly[536];
    poly[6956] = poly[6] * poly[536];
    poly[6957] = poly[30] * poly[70];
    poly[6958] = poly[30] * poly[71];
    poly[6959] = poly[72] * poly[30];
    poly[6960] = poly[30] * poly[73];
    poly[6961] = poly[30] * poly[74];
    poly[6962] = poly[1] * poly[537];
    poly[6963] = poly[2] * poly[537];
    poly[6964] = poly[3] * poly[537];
    poly[6965] = poly[4] * poly[537];
    poly[6966] = poly[5] * poly[537];
    poly[6967] = poly[6] * poly[537];
    poly[6968] = poly[30] * poly[81];
    poly[6969] = poly[30] * poly[82];
    poly[6970] = poly[30] * poly[83];
    poly[6971] = poly[84] * poly[30];
    poly[6972] = poly[30] * poly[85];
    poly[6973] = poly[30] * poly[86];
    poly[6974] = poly[30] * poly[87];
    poly[6975] = poly[1] * poly[538];
    poly[6976] = poly[2] * poly[538];
    poly[6977] = poly[3] * poly[538];
    poly[6978] = poly[4] * poly[538];
    poly[6979] = poly[5] * poly[538];
    poly[6980] = poly[6] * poly[538];
    poly[6981] = poly[11] * poly[534];
    poly[6982] = poly[11] * poly[535];
    poly[6983] = poly[11] * poly[536];
    poly[6984] = poly[11] * poly[537];
    poly[6985] = poly[1] * poly[539];
    poly[6986] = poly[2] * poly[539];
    poly[6987] = poly[3] * poly[539];
    poly[6988] = poly[4] * poly[539];
    poly[6989] = poly[5] * poly[539];
    poly[6990] = poly[6] * poly[539];
    poly[6991] = poly[7] * poly[539];
    poly[6992] = poly[8] * poly[539];
    poly[6993] = poly[9] * poly[539];
    poly[6994] = poly[10] * poly[539];
    poly[6995] = poly[11] * poly[539];
    poly[6996] = poly[1] * poly[540];
    poly[6997] = poly[2] * poly[540];
    poly[6998] = poly[3] * poly[540];
    poly[6999] = poly[4] * poly[540];
}

fn f_polynomials140(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7000] = poly[5] * poly[540];
    poly[7001] = poly[6] * poly[540];
    poly[7002] = poly[7] * poly[540];
    poly[7003] = poly[8] * poly[540];
    poly[7004] = poly[9] * poly[540];
    poly[7005] = poly[10] * poly[540];
    poly[7006] = poly[11] * poly[540];
    poly[7007] = mono[1615] + mono[1616] + mono[1617] + mono[1618];
    poly[7008] = poly[1] * poly[541];
    poly[7009] = poly[2] * poly[541];
    poly[7010] = poly[3] * poly[541];
    poly[7011] = poly[4] * poly[541];
    poly[7012] = poly[5] * poly[541];
    poly[7013] = poly[6] * poly[541];
    poly[7014] = poly[7] * poly[541];
    poly[7015] = poly[8] * poly[541];
    poly[7016] = poly[9] * poly[541];
    poly[7017] = poly[10] * poly[541];
    poly[7018] = poly[11] * poly[541];
    poly[7019] = mono[1619] + mono[1620] + mono[1621] + mono[1622];
    poly[7020] = mono[1623] + mono[1624] + mono[1625] + mono[1626];
    poly[7021] = poly[1] * poly[542];
    poly[7022] = poly[2] * poly[542];
    poly[7023] = poly[3] * poly[542];
    poly[7024] = poly[4] * poly[542];
    poly[7025] = poly[5] * poly[542];
    poly[7026] = poly[6] * poly[542];
    poly[7027] = poly[7] * poly[542];
    poly[7028] = poly[8] * poly[542];
    poly[7029] = poly[9] * poly[542];
    poly[7030] = poly[10] * poly[542];
    poly[7031] = poly[11] * poly[542];
    poly[7032] = mono[1627] + mono[1628] + mono[1629] + mono[1630];
    poly[7033] = mono[1631] + mono[1632] + mono[1633] + mono[1634];
    poly[7034] = mono[1635] + mono[1636] + mono[1637] + mono[1638];
    poly[7035] = poly[1] * poly[543];
    poly[7036] = poly[2] * poly[543];
    poly[7037] = poly[3] * poly[543];
    poly[7038] = poly[4] * poly[543];
    poly[7039] = poly[5] * poly[543];
    poly[7040] = poly[6] * poly[543];
    poly[7041] = mono[1639]
        + mono[1640]
        + mono[1641]
        + mono[1642]
        + mono[1643]
        + mono[1644]
        + mono[1645]
        + mono[1646];
    poly[7042] = mono[1647]
        + mono[1648]
        + mono[1649]
        + mono[1650]
        + mono[1651]
        + mono[1652]
        + mono[1653]
        + mono[1654];
    poly[7043] = mono[1655]
        + mono[1656]
        + mono[1657]
        + mono[1658]
        + mono[1659]
        + mono[1660]
        + mono[1661]
        + mono[1662];
    poly[7044] = mono[1663]
        + mono[1664]
        + mono[1665]
        + mono[1666]
        + mono[1667]
        + mono[1668]
        + mono[1669]
        + mono[1670];
    poly[7045] = poly[7] * poly[543] - poly[7041];
    poly[7046] = poly[8] * poly[543] - poly[7042];
    poly[7047] = poly[9] * poly[543] - poly[7043];
    poly[7048] = poly[10] * poly[543] - poly[7044];
    poly[7049] = poly[11] * poly[543];
}

fn f_polynomials141(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7050] = mono[1671]
        + mono[1672]
        + mono[1673]
        + mono[1674]
        + mono[1675]
        + mono[1676]
        + mono[1677]
        + mono[1678];
    poly[7051] = mono[1679]
        + mono[1680]
        + mono[1681]
        + mono[1682]
        + mono[1683]
        + mono[1684]
        + mono[1685]
        + mono[1686];
    poly[7052] = mono[1687]
        + mono[1688]
        + mono[1689]
        + mono[1690]
        + mono[1691]
        + mono[1692]
        + mono[1693]
        + mono[1694];
    poly[7053] = mono[1695]
        + mono[1696]
        + mono[1697]
        + mono[1698]
        + mono[1699]
        + mono[1700]
        + mono[1701]
        + mono[1702];
    poly[7054] = mono[1703] + mono[1704] + mono[1705] + mono[1706];
    poly[7055] = poly[1] * poly[544];
    poly[7056] = poly[2] * poly[544];
    poly[7057] = poly[3] * poly[544];
    poly[7058] = poly[4] * poly[544];
    poly[7059] = poly[5] * poly[544];
    poly[7060] = poly[6] * poly[544];
    poly[7061] = poly[7] * poly[544];
    poly[7062] = poly[8] * poly[544];
    poly[7063] = poly[9] * poly[544];
    poly[7064] = poly[10] * poly[544];
    poly[7065] = poly[11] * poly[544];
    poly[7066] = poly[109] * poly[30];
    poly[7067] = poly[12] * poly[540] - poly[7007];
    poly[7068] = poly[12] * poly[541] - poly[7019];
    poly[7069] = poly[12] * poly[542] - poly[7032];
    poly[7070] = poly[12] * poly[543] - poly[7050];
    poly[7071] = poly[1] * poly[545];
    poly[7072] = poly[2] * poly[545];
    poly[7073] = poly[3] * poly[545];
    poly[7074] = poly[4] * poly[545];
    poly[7075] = poly[5] * poly[545];
    poly[7076] = poly[6] * poly[545];
    poly[7077] = poly[7] * poly[545];
    poly[7078] = poly[8] * poly[545];
    poly[7079] = poly[9] * poly[545];
    poly[7080] = poly[10] * poly[545];
    poly[7081] = poly[11] * poly[545];
    poly[7082] = poly[13] * poly[539] - poly[7007];
    poly[7083] = poly[122] * poly[30];
    poly[7084] = poly[13] * poly[541] - poly[7020];
    poly[7085] = poly[13] * poly[542] - poly[7033];
    poly[7086] = poly[13] * poly[543] - poly[7051];
    poly[7087] = poly[12] * poly[545] - poly[7082];
    poly[7088] = poly[1] * poly[546];
    poly[7089] = poly[2] * poly[546];
    poly[7090] = poly[3] * poly[546];
    poly[7091] = poly[4] * poly[546];
    poly[7092] = poly[5] * poly[546];
    poly[7093] = poly[6] * poly[546];
    poly[7094] = poly[7] * poly[546];
    poly[7095] = poly[8] * poly[546];
    poly[7096] = poly[9] * poly[546];
    poly[7097] = poly[10] * poly[546];
    poly[7098] = poly[11] * poly[546];
    poly[7099] = poly[14] * poly[539] - poly[7019];
}

fn f_polynomials142(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7100] = poly[14] * poly[540] - poly[7020];
    poly[7101] = poly[137] * poly[30];
    poly[7102] = poly[14] * poly[542] - poly[7034];
    poly[7103] = poly[14] * poly[543] - poly[7052];
    poly[7104] = poly[12] * poly[546] - poly[7099];
    poly[7105] = poly[13] * poly[546] - poly[7100];
    poly[7106] = poly[1] * poly[547];
    poly[7107] = poly[2] * poly[547];
    poly[7108] = poly[3] * poly[547];
    poly[7109] = poly[4] * poly[547];
    poly[7110] = poly[5] * poly[547];
    poly[7111] = poly[6] * poly[547];
    poly[7112] = poly[7] * poly[547];
    poly[7113] = poly[8] * poly[547];
    poly[7114] = poly[9] * poly[547];
    poly[7115] = poly[10] * poly[547];
    poly[7116] = poly[11] * poly[547];
    poly[7117] = poly[15] * poly[539] - poly[7032];
    poly[7118] = poly[15] * poly[540] - poly[7033];
    poly[7119] = poly[15] * poly[541] - poly[7034];
    poly[7120] = poly[154] * poly[30];
    poly[7121] = poly[15] * poly[543] - poly[7053];
    poly[7122] = poly[12] * poly[547] - poly[7117];
    poly[7123] = poly[13] * poly[547] - poly[7118];
    poly[7124] = poly[14] * poly[547] - poly[7119];
    poly[7125] = poly[1] * poly[548];
    poly[7126] = poly[2] * poly[548];
    poly[7127] = poly[3] * poly[548];
    poly[7128] = poly[4] * poly[548];
    poly[7129] = poly[5] * poly[548];
    poly[7130] = poly[6] * poly[548];
    poly[7131] = poly[30] * poly[164] - poly[7041];
    poly[7132] = poly[30] * poly[165] - poly[7042];
    poly[7133] = poly[30] * poly[166] - poly[7043];
    poly[7134] = poly[30] * poly[167] - poly[7044];
    poly[7135] = poly[7] * poly[548] - poly[7131];
    poly[7136] = poly[8] * poly[548] - poly[7132];
    poly[7137] = poly[9] * poly[548] - poly[7133];
    poly[7138] = poly[10] * poly[548] - poly[7134];
    poly[7139] = poly[11] * poly[548];
    poly[7140] = poly[16] * poly[539] - poly[7050];
    poly[7141] = poly[16] * poly[540] - poly[7051];
    poly[7142] = poly[16] * poly[541] - poly[7052];
    poly[7143] = poly[16] * poly[542] - poly[7053];
    poly[7144] = poly[30] * poly[177];
    poly[7145] = poly[30] * poly[178];
    poly[7146] = poly[12] * poly[548] - poly[7140];
    poly[7147] = poly[13] * poly[548] - poly[7141];
    poly[7148] = poly[14] * poly[548] - poly[7142];
    poly[7149] = poly[15] * poly[548] - poly[7143];
}

fn f_polynomials143(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7150] = poly[30] * poly[183] - poly[7054];
    poly[7151] = poly[1] * poly[549];
    poly[7152] = poly[2] * poly[549];
    poly[7153] = poly[3] * poly[549];
    poly[7154] = poly[4] * poly[549];
    poly[7155] = poly[5] * poly[549];
    poly[7156] = poly[6] * poly[549];
    poly[7157] = poly[17] * poly[534];
    poly[7158] = poly[17] * poly[535];
    poly[7159] = poly[17] * poly[536];
    poly[7160] = poly[17] * poly[537];
    poly[7161] = poly[11] * poly[549];
    poly[7162] = poly[17] * poly[539];
    poly[7163] = poly[17] * poly[540];
    poly[7164] = poly[17] * poly[541];
    poly[7165] = poly[17] * poly[542];
    poly[7166] = poly[17] * poly[543];
    poly[7167] = poly[17] * poly[544];
    poly[7168] = poly[17] * poly[545];
    poly[7169] = poly[17] * poly[546];
    poly[7170] = poly[17] * poly[547];
    poly[7171] = poly[17] * poly[548];
    poly[7172] = poly[1] * poly[550];
    poly[7173] = poly[2] * poly[550];
    poly[7174] = poly[3] * poly[550];
    poly[7175] = poly[4] * poly[550];
    poly[7176] = poly[5] * poly[550];
    poly[7177] = poly[6] * poly[550];
    poly[7178] = poly[7] * poly[550];
    poly[7179] = poly[8] * poly[550];
    poly[7180] = poly[9] * poly[550];
    poly[7181] = poly[10] * poly[550];
    poly[7182] = poly[11] * poly[550];
    poly[7183] = poly[18] * poly[539];
    poly[7184] = poly[18] * poly[540];
    poly[7185] = poly[18] * poly[541];
    poly[7186] = poly[18] * poly[542];
    poly[7187] = poly[18] * poly[543];
    poly[7188] = poly[18] * poly[544];
    poly[7189] = poly[18] * poly[545];
    poly[7190] = poly[18] * poly[546];
    poly[7191] = poly[18] * poly[547];
    poly[7192] = poly[18] * poly[548];
    poly[7193] = poly[17] * poly[550];
    poly[7194] = poly[217] * poly[30];
    poly[7195] = poly[1] * poly[551];
    poly[7196] = poly[2] * poly[551];
    poly[7197] = poly[3] * poly[551];
    poly[7198] = poly[4] * poly[551];
    poly[7199] = poly[5] * poly[551];
}

fn f_polynomials144(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7200] = poly[6] * poly[551];
    poly[7201] = poly[7] * poly[551];
    poly[7202] = poly[8] * poly[551];
    poly[7203] = poly[9] * poly[551];
    poly[7204] = poly[10] * poly[551];
    poly[7205] = poly[11] * poly[551];
    poly[7206] = poly[19] * poly[539];
    poly[7207] = poly[19] * poly[540];
    poly[7208] = poly[19] * poly[541];
    poly[7209] = poly[19] * poly[542];
    poly[7210] = poly[19] * poly[543];
    poly[7211] = poly[19] * poly[544];
    poly[7212] = poly[19] * poly[545];
    poly[7213] = poly[19] * poly[546];
    poly[7214] = poly[19] * poly[547];
    poly[7215] = poly[19] * poly[548];
    poly[7216] = poly[17] * poly[551];
    poly[7217] = poly[30] * poly[235];
    poly[7218] = poly[236] * poly[30];
    poly[7219] = poly[30] * poly[237];
    poly[7220] = poly[1] * poly[552];
    poly[7221] = poly[2] * poly[552];
    poly[7222] = poly[3] * poly[552];
    poly[7223] = poly[4] * poly[552];
    poly[7224] = poly[5] * poly[552];
    poly[7225] = poly[6] * poly[552];
    poly[7226] = poly[7] * poly[552];
    poly[7227] = poly[8] * poly[552];
    poly[7228] = poly[9] * poly[552];
    poly[7229] = poly[10] * poly[552];
    poly[7230] = poly[11] * poly[552];
    poly[7231] = poly[20] * poly[539];
    poly[7232] = poly[20] * poly[540];
    poly[7233] = poly[20] * poly[541];
    poly[7234] = poly[20] * poly[542];
    poly[7235] = poly[20] * poly[543];
    poly[7236] = poly[20] * poly[544];
    poly[7237] = poly[20] * poly[545];
    poly[7238] = poly[20] * poly[546];
    poly[7239] = poly[20] * poly[547];
    poly[7240] = poly[20] * poly[548];
    poly[7241] = poly[17] * poly[552];
    poly[7242] = poly[30] * poly[255];
    poly[7243] = poly[30] * poly[256];
    poly[7244] = poly[257] * poly[30];
    poly[7245] = poly[30] * poly[258];
    poly[7246] = poly[30] * poly[259];
    poly[7247] = poly[1] * poly[553];
    poly[7248] = poly[2] * poly[553];
    poly[7249] = poly[3] * poly[553];
}

fn f_polynomials145(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7250] = poly[4] * poly[553];
    poly[7251] = poly[5] * poly[553];
    poly[7252] = poly[6] * poly[553];
    poly[7253] = poly[7] * poly[553];
    poly[7254] = poly[8] * poly[553];
    poly[7255] = poly[9] * poly[553];
    poly[7256] = poly[10] * poly[553];
    poly[7257] = poly[11] * poly[553];
    poly[7258] = poly[21] * poly[539];
    poly[7259] = poly[21] * poly[540];
    poly[7260] = poly[21] * poly[541];
    poly[7261] = poly[21] * poly[542];
    poly[7262] = poly[21] * poly[543];
    poly[7263] = poly[21] * poly[544];
    poly[7264] = poly[21] * poly[545];
    poly[7265] = poly[21] * poly[546];
    poly[7266] = poly[21] * poly[547];
    poly[7267] = poly[21] * poly[548];
    poly[7268] = poly[17] * poly[553];
    poly[7269] = poly[30] * poly[277];
    poly[7270] = poly[30] * poly[278];
    poly[7271] = poly[30] * poly[279];
    poly[7272] = poly[280] * poly[30];
    poly[7273] = poly[30] * poly[281];
    poly[7274] = poly[30] * poly[282];
    poly[7275] = poly[30] * poly[283];
    poly[7276] = poly[1] * poly[554];
    poly[7277] = poly[2] * poly[554];
    poly[7278] = poly[3] * poly[554];
    poly[7279] = poly[4] * poly[554];
    poly[7280] = poly[5] * poly[554];
    poly[7281] = poly[6] * poly[554];
    poly[7282] = poly[30] * poly[290];
    poly[7283] = poly[30] * poly[291];
    poly[7284] = poly[30] * poly[292];
    poly[7285] = poly[30] * poly[293];
    poly[7286] = poly[30] * poly[294];
    poly[7287] = poly[30] * poly[295];
    poly[7288] = poly[30] * poly[296];
    poly[7289] = poly[30] * poly[297];
    poly[7290] = poly[11] * poly[554];
    poly[7291] = poly[22] * poly[539];
    poly[7292] = poly[22] * poly[540];
    poly[7293] = poly[22] * poly[541];
    poly[7294] = poly[22] * poly[542];
    poly[7295] = mono[1707]
        + mono[1708]
        + mono[1709]
        + mono[1710]
        + mono[1711]
        + mono[1712]
        + mono[1713]
        + mono[1714]
        + mono[1715]
        + mono[1716]
        + mono[1717]
        + mono[1718]
        + mono[1719]
        + mono[1720]
        + mono[1721]
        + mono[1722];
    poly[7296] = poly[22] * poly[543] - poly[7295];
    poly[7297] = poly[22] * poly[544];
    poly[7298] = poly[22] * poly[545];
    poly[7299] = poly[22] * poly[546];
}

fn f_polynomials146(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7300] = poly[22] * poly[547];
    poly[7301] = poly[30] * poly[303] - poly[7295];
    poly[7302] = poly[22] * poly[548] - poly[7301];
    poly[7303] = poly[17] * poly[554];
    poly[7304] = poly[30] * poly[306];
    poly[7305] = poly[30] * poly[307];
    poly[7306] = poly[30] * poly[308];
    poly[7307] = poly[30] * poly[309];
    poly[7308] = poly[30] * poly[310];
    poly[7309] = poly[30] * poly[311];
    poly[7310] = poly[30] * poly[312];
    poly[7311] = poly[30] * poly[313];
    poly[7312] = poly[30] * poly[314];
    poly[7313] = poly[30] * poly[315];
    poly[7314] = poly[30] * poly[316];
    poly[7315] = poly[1] * poly[555];
    poly[7316] = poly[2] * poly[555];
    poly[7317] = poly[3] * poly[555];
    poly[7318] = poly[4] * poly[555];
    poly[7319] = poly[5] * poly[555];
    poly[7320] = poly[6] * poly[555];
    poly[7321] = poly[7] * poly[555];
    poly[7322] = poly[8] * poly[555];
    poly[7323] = poly[9] * poly[555];
    poly[7324] = poly[10] * poly[555];
    poly[7325] = poly[11] * poly[555];
    poly[7326] = mono[1723]
        + mono[1724]
        + mono[1725]
        + mono[1726]
        + mono[1727]
        + mono[1728]
        + mono[1729]
        + mono[1730];
    poly[7327] = mono[1731]
        + mono[1732]
        + mono[1733]
        + mono[1734]
        + mono[1735]
        + mono[1736]
        + mono[1737]
        + mono[1738];
    poly[7328] = mono[1739]
        + mono[1740]
        + mono[1741]
        + mono[1742]
        + mono[1743]
        + mono[1744]
        + mono[1745]
        + mono[1746];
    poly[7329] = mono[1747]
        + mono[1748]
        + mono[1749]
        + mono[1750]
        + mono[1751]
        + mono[1752]
        + mono[1753]
        + mono[1754];
    poly[7330] = mono[1755]
        + mono[1756]
        + mono[1757]
        + mono[1758]
        + mono[1759]
        + mono[1760]
        + mono[1761]
        + mono[1762]
        + mono[1763]
        + mono[1764]
        + mono[1765]
        + mono[1766]
        + mono[1767]
        + mono[1768]
        + mono[1769]
        + mono[1770];
    poly[7331] = poly[12] * poly[555] - poly[7326];
    poly[7332] = poly[13] * poly[555] - poly[7327];
    poly[7333] = poly[14] * poly[555] - poly[7328];
    poly[7334] = poly[15] * poly[555] - poly[7329];
    poly[7335] = poly[16] * poly[555] - poly[7330];
    poly[7336] = poly[17] * poly[555];
    poly[7337] = mono[1771]
        + mono[1772]
        + mono[1773]
        + mono[1774]
        + mono[1775]
        + mono[1776]
        + mono[1777]
        + mono[1778];
    poly[7338] = mono[1779]
        + mono[1780]
        + mono[1781]
        + mono[1782]
        + mono[1783]
        + mono[1784]
        + mono[1785]
        + mono[1786];
    poly[7339] = mono[1787]
        + mono[1788]
        + mono[1789]
        + mono[1790]
        + mono[1791]
        + mono[1792]
        + mono[1793]
        + mono[1794];
    poly[7340] = mono[1795]
        + mono[1796]
        + mono[1797]
        + mono[1798]
        + mono[1799]
        + mono[1800]
        + mono[1801]
        + mono[1802];
    poly[7341] = mono[1803]
        + mono[1804]
        + mono[1805]
        + mono[1806]
        + mono[1807]
        + mono[1808]
        + mono[1809]
        + mono[1810]
        + mono[1811]
        + mono[1812]
        + mono[1813]
        + mono[1814]
        + mono[1815]
        + mono[1816]
        + mono[1817]
        + mono[1818];
    poly[7342] = mono[1819] + mono[1820] + mono[1821] + mono[1822];
    poly[7343] = poly[18] * poly[555] - poly[7337];
    poly[7344] = poly[19] * poly[555] - poly[7338];
    poly[7345] = poly[20] * poly[555] - poly[7339];
    poly[7346] = poly[21] * poly[555] - poly[7340];
    poly[7347] = poly[22] * poly[555] - poly[7341];
    poly[7348] = poly[1] * poly[556];
    poly[7349] = poly[2] * poly[556];
}

fn f_polynomials147(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7350] = poly[3] * poly[556];
    poly[7351] = poly[4] * poly[556];
    poly[7352] = poly[5] * poly[556];
    poly[7353] = poly[6] * poly[556];
    poly[7354] = poly[7] * poly[556];
    poly[7355] = poly[8] * poly[556];
    poly[7356] = poly[9] * poly[556];
    poly[7357] = poly[10] * poly[556];
    poly[7358] = poly[11] * poly[556];
    poly[7359] = poly[23] * poly[539] - poly[7326];
    poly[7360] = poly[23] * poly[540] - poly[7327];
    poly[7361] = poly[23] * poly[541] - poly[7328];
    poly[7362] = poly[23] * poly[542] - poly[7329];
    poly[7363] = poly[23] * poly[543] - poly[7330];
    poly[7364] = poly[12] * poly[556] - poly[7359];
    poly[7365] = poly[13] * poly[556] - poly[7360];
    poly[7366] = poly[14] * poly[556] - poly[7361];
    poly[7367] = poly[15] * poly[556] - poly[7362];
    poly[7368] = poly[16] * poly[556] - poly[7363];
    poly[7369] = poly[17] * poly[556];
    poly[7370] = poly[30] * poly[339] - poly[7337];
    poly[7371] = poly[30] * poly[340] - poly[7338];
    poly[7372] = poly[30] * poly[341] - poly[7339];
    poly[7373] = poly[30] * poly[342] - poly[7340];
    poly[7374] = poly[30] * poly[343] - poly[7341];
    poly[7375] = poly[30] * poly[344];
    poly[7376] = poly[30] * poly[345] - poly[7342];
    poly[7377] = poly[18] * poly[556] - poly[7370];
    poly[7378] = poly[19] * poly[556] - poly[7371];
    poly[7379] = poly[20] * poly[556] - poly[7372];
    poly[7380] = poly[21] * poly[556] - poly[7373];
    poly[7381] = poly[22] * poly[556] - poly[7374];
    poly[7382] = poly[30] * poly[351];
    poly[7383] = poly[1] * poly[557];
    poly[7384] = poly[2] * poly[557];
    poly[7385] = poly[3] * poly[557];
    poly[7386] = poly[4] * poly[557];
    poly[7387] = poly[5] * poly[557];
    poly[7388] = poly[6] * poly[557];
    poly[7389] = poly[24] * poly[534];
    poly[7390] = poly[24] * poly[535];
    poly[7391] = poly[24] * poly[536];
    poly[7392] = poly[24] * poly[537];
    poly[7393] = poly[11] * poly[557];
    poly[7394] = poly[24] * poly[539];
    poly[7395] = poly[24] * poly[540];
    poly[7396] = poly[24] * poly[541];
    poly[7397] = poly[24] * poly[542];
    poly[7398] = poly[24] * poly[543];
    poly[7399] = poly[24] * poly[544];
}

fn f_polynomials148(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7400] = poly[24] * poly[545];
    poly[7401] = poly[24] * poly[546];
    poly[7402] = poly[24] * poly[547];
    poly[7403] = poly[24] * poly[548];
    poly[7404] = poly[17] * poly[557];
    poly[7405] = poly[24] * poly[550];
    poly[7406] = poly[24] * poly[551];
    poly[7407] = poly[24] * poly[552];
    poly[7408] = poly[24] * poly[553];
    poly[7409] = poly[24] * poly[554];
    poly[7410] = poly[24] * poly[555];
    poly[7411] = poly[24] * poly[556];
    poly[7412] = poly[1] * poly[558];
    poly[7413] = poly[2] * poly[558];
    poly[7414] = poly[3] * poly[558];
    poly[7415] = poly[4] * poly[558];
    poly[7416] = poly[5] * poly[558];
    poly[7417] = poly[6] * poly[558];
    poly[7418] = poly[7] * poly[558];
    poly[7419] = poly[8] * poly[558];
    poly[7420] = poly[9] * poly[558];
    poly[7421] = poly[10] * poly[558];
    poly[7422] = poly[11] * poly[558];
    poly[7423] = mono[1823] + mono[1824] + mono[1825] + mono[1826];
    poly[7424] = mono[1827] + mono[1828] + mono[1829] + mono[1830];
    poly[7425] = mono[1831] + mono[1832] + mono[1833] + mono[1834];
    poly[7426] = mono[1835] + mono[1836] + mono[1837] + mono[1838];
    poly[7427] = mono[1839]
        + mono[1840]
        + mono[1841]
        + mono[1842]
        + mono[1843]
        + mono[1844]
        + mono[1845]
        + mono[1846];
    poly[7428] = poly[12] * poly[558] - poly[7423];
    poly[7429] = poly[13] * poly[558] - poly[7424];
    poly[7430] = poly[14] * poly[558] - poly[7425];
    poly[7431] = poly[15] * poly[558] - poly[7426];
    poly[7432] = poly[16] * poly[558] - poly[7427];
    poly[7433] = poly[17] * poly[558];
    poly[7434] = poly[18] * poly[558];
    poly[7435] = poly[19] * poly[558];
    poly[7436] = poly[20] * poly[558];
    poly[7437] = poly[21] * poly[558];
    poly[7438] = poly[22] * poly[558];
    poly[7439] = mono[1847]
        + mono[1848]
        + mono[1849]
        + mono[1850]
        + mono[1851]
        + mono[1852]
        + mono[1853]
        + mono[1854];
    poly[7440] = poly[23] * poly[558] - poly[7439];
    poly[7441] = poly[24] * poly[558];
    poly[7442] = poly[1] * poly[559];
    poly[7443] = poly[2] * poly[559];
    poly[7444] = poly[3] * poly[559];
    poly[7445] = poly[4] * poly[559];
    poly[7446] = poly[5] * poly[559];
    poly[7447] = poly[6] * poly[559];
    poly[7448] = poly[7] * poly[559];
    poly[7449] = poly[8] * poly[559];
}

fn f_polynomials149(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7450] = poly[9] * poly[559];
    poly[7451] = poly[10] * poly[559];
    poly[7452] = poly[11] * poly[559];
    poly[7453] = mono[1855] + mono[1856] + mono[1857] + mono[1858];
    poly[7454] = mono[1859] + mono[1860] + mono[1861] + mono[1862];
    poly[7455] = mono[1863] + mono[1864] + mono[1865] + mono[1866];
    poly[7456] = mono[1867] + mono[1868] + mono[1869] + mono[1870];
    poly[7457] = mono[1871]
        + mono[1872]
        + mono[1873]
        + mono[1874]
        + mono[1875]
        + mono[1876]
        + mono[1877]
        + mono[1878];
    poly[7458] = poly[12] * poly[559] - poly[7453];
    poly[7459] = poly[13] * poly[559] - poly[7454];
    poly[7460] = poly[14] * poly[559] - poly[7455];
    poly[7461] = poly[15] * poly[559] - poly[7456];
    poly[7462] = poly[16] * poly[559] - poly[7457];
    poly[7463] = poly[17] * poly[559];
    poly[7464] = poly[18] * poly[559];
    poly[7465] = poly[19] * poly[559];
    poly[7466] = poly[20] * poly[559];
    poly[7467] = poly[21] * poly[559];
    poly[7468] = poly[22] * poly[559];
    poly[7469] = mono[1879]
        + mono[1880]
        + mono[1881]
        + mono[1882]
        + mono[1883]
        + mono[1884]
        + mono[1885]
        + mono[1886];
    poly[7470] = poly[23] * poly[559] - poly[7469];
    poly[7471] = poly[24] * poly[559];
    poly[7472] = mono[1887] + mono[1888] + mono[1889] + mono[1890];
    poly[7473] = poly[1] * poly[560];
    poly[7474] = poly[2] * poly[560];
    poly[7475] = poly[3] * poly[560];
    poly[7476] = poly[4] * poly[560];
    poly[7477] = poly[5] * poly[560];
    poly[7478] = poly[6] * poly[560];
    poly[7479] = poly[7] * poly[560];
    poly[7480] = poly[8] * poly[560];
    poly[7481] = poly[9] * poly[560];
    poly[7482] = poly[10] * poly[560];
    poly[7483] = poly[11] * poly[560];
    poly[7484] = mono[1891] + mono[1892] + mono[1893] + mono[1894];
    poly[7485] = mono[1895] + mono[1896] + mono[1897] + mono[1898];
    poly[7486] = mono[1899] + mono[1900] + mono[1901] + mono[1902];
    poly[7487] = mono[1903] + mono[1904] + mono[1905] + mono[1906];
    poly[7488] = mono[1907]
        + mono[1908]
        + mono[1909]
        + mono[1910]
        + mono[1911]
        + mono[1912]
        + mono[1913]
        + mono[1914];
    poly[7489] = poly[12] * poly[560] - poly[7484];
    poly[7490] = poly[13] * poly[560] - poly[7485];
    poly[7491] = poly[14] * poly[560] - poly[7486];
    poly[7492] = poly[15] * poly[560] - poly[7487];
    poly[7493] = poly[16] * poly[560] - poly[7488];
    poly[7494] = poly[17] * poly[560];
    poly[7495] = poly[18] * poly[560];
    poly[7496] = poly[19] * poly[560];
    poly[7497] = poly[20] * poly[560];
    poly[7498] = poly[21] * poly[560];
    poly[7499] = poly[22] * poly[560];
}

fn f_polynomials150(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7500] = mono[1915]
        + mono[1916]
        + mono[1917]
        + mono[1918]
        + mono[1919]
        + mono[1920]
        + mono[1921]
        + mono[1922];
    poly[7501] = poly[23] * poly[560] - poly[7500];
    poly[7502] = poly[24] * poly[560];
    poly[7503] = mono[1923] + mono[1924] + mono[1925] + mono[1926];
    poly[7504] = mono[1927] + mono[1928] + mono[1929] + mono[1930];
    poly[7505] = poly[1] * poly[561];
    poly[7506] = poly[2] * poly[561];
    poly[7507] = poly[3] * poly[561];
    poly[7508] = poly[4] * poly[561];
    poly[7509] = poly[5] * poly[561];
    poly[7510] = poly[6] * poly[561];
    poly[7511] = poly[7] * poly[561];
    poly[7512] = poly[8] * poly[561];
    poly[7513] = poly[9] * poly[561];
    poly[7514] = poly[10] * poly[561];
    poly[7515] = poly[11] * poly[561];
    poly[7516] = mono[1931] + mono[1932] + mono[1933] + mono[1934];
    poly[7517] = mono[1935] + mono[1936] + mono[1937] + mono[1938];
    poly[7518] = mono[1939] + mono[1940] + mono[1941] + mono[1942];
    poly[7519] = mono[1943] + mono[1944] + mono[1945] + mono[1946];
    poly[7520] = mono[1947]
        + mono[1948]
        + mono[1949]
        + mono[1950]
        + mono[1951]
        + mono[1952]
        + mono[1953]
        + mono[1954];
    poly[7521] = poly[12] * poly[561] - poly[7516];
    poly[7522] = poly[13] * poly[561] - poly[7517];
    poly[7523] = poly[14] * poly[561] - poly[7518];
    poly[7524] = poly[15] * poly[561] - poly[7519];
    poly[7525] = poly[16] * poly[561] - poly[7520];
    poly[7526] = poly[17] * poly[561];
    poly[7527] = poly[18] * poly[561];
    poly[7528] = poly[19] * poly[561];
    poly[7529] = poly[20] * poly[561];
    poly[7530] = poly[21] * poly[561];
    poly[7531] = poly[22] * poly[561];
    poly[7532] = mono[1955]
        + mono[1956]
        + mono[1957]
        + mono[1958]
        + mono[1959]
        + mono[1960]
        + mono[1961]
        + mono[1962];
    poly[7533] = poly[23] * poly[561] - poly[7532];
    poly[7534] = poly[24] * poly[561];
    poly[7535] = mono[1963] + mono[1964] + mono[1965] + mono[1966];
    poly[7536] = mono[1967] + mono[1968] + mono[1969] + mono[1970];
    poly[7537] = mono[1971] + mono[1972] + mono[1973] + mono[1974];
    poly[7538] = poly[1] * poly[562];
    poly[7539] = poly[2] * poly[562];
    poly[7540] = poly[3] * poly[562];
    poly[7541] = poly[4] * poly[562];
    poly[7542] = poly[5] * poly[562];
    poly[7543] = poly[6] * poly[562];
    poly[7544] = mono[1975]
        + mono[1976]
        + mono[1977]
        + mono[1978]
        + mono[1979]
        + mono[1980]
        + mono[1981]
        + mono[1982];
    poly[7545] = mono[1983]
        + mono[1984]
        + mono[1985]
        + mono[1986]
        + mono[1987]
        + mono[1988]
        + mono[1989]
        + mono[1990];
    poly[7546] = mono[1991]
        + mono[1992]
        + mono[1993]
        + mono[1994]
        + mono[1995]
        + mono[1996]
        + mono[1997]
        + mono[1998];
    poly[7547] = mono[1999]
        + mono[2000]
        + mono[2001]
        + mono[2002]
        + mono[2003]
        + mono[2004]
        + mono[2005]
        + mono[2006];
    poly[7548] = poly[7] * poly[562] - poly[7544];
    poly[7549] = poly[8] * poly[562] - poly[7545];
}

fn f_polynomials151(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7550] = poly[9] * poly[562] - poly[7546];
    poly[7551] = poly[10] * poly[562] - poly[7547];
    poly[7552] = poly[11] * poly[562];
    poly[7553] = mono[2007]
        + mono[2008]
        + mono[2009]
        + mono[2010]
        + mono[2011]
        + mono[2012]
        + mono[2013]
        + mono[2014];
    poly[7554] = mono[2015]
        + mono[2016]
        + mono[2017]
        + mono[2018]
        + mono[2019]
        + mono[2020]
        + mono[2021]
        + mono[2022];
    poly[7555] = mono[2023]
        + mono[2024]
        + mono[2025]
        + mono[2026]
        + mono[2027]
        + mono[2028]
        + mono[2029]
        + mono[2030];
    poly[7556] = mono[2031]
        + mono[2032]
        + mono[2033]
        + mono[2034]
        + mono[2035]
        + mono[2036]
        + mono[2037]
        + mono[2038];
    poly[7557] = mono[2039]
        + mono[2040]
        + mono[2041]
        + mono[2042]
        + mono[2043]
        + mono[2044]
        + mono[2045]
        + mono[2046];
    poly[7558] = mono[2047]
        + mono[2048]
        + mono[2049]
        + mono[2050]
        + mono[2051]
        + mono[2052]
        + mono[2053]
        + mono[2054];
    poly[7559] = poly[12] * poly[562] - poly[7553];
    poly[7560] = poly[13] * poly[562] - poly[7554];
    poly[7561] = poly[14] * poly[562] - poly[7555];
    poly[7562] = poly[15] * poly[562] - poly[7556];
    poly[7563] = mono[2055]
        + mono[2056]
        + mono[2057]
        + mono[2058]
        + mono[2059]
        + mono[2060]
        + mono[2061]
        + mono[2062];
    poly[7564] = poly[16] * poly[562] - poly[7563] - poly[7558] - poly[7557];
    poly[7565] = poly[17] * poly[562];
    poly[7566] = poly[18] * poly[562];
    poly[7567] = poly[19] * poly[562];
    poly[7568] = poly[20] * poly[562];
    poly[7569] = poly[21] * poly[562];
    poly[7570] = mono[2063]
        + mono[2064]
        + mono[2065]
        + mono[2066]
        + mono[2067]
        + mono[2068]
        + mono[2069]
        + mono[2070]
        + mono[2071]
        + mono[2072]
        + mono[2073]
        + mono[2074]
        + mono[2075]
        + mono[2076]
        + mono[2077]
        + mono[2078];
    poly[7571] = poly[22] * poly[562] - poly[7570];
    poly[7572] = mono[2079]
        + mono[2080]
        + mono[2081]
        + mono[2082]
        + mono[2083]
        + mono[2084]
        + mono[2085]
        + mono[2086]
        + mono[2087]
        + mono[2088]
        + mono[2089]
        + mono[2090]
        + mono[2091]
        + mono[2092]
        + mono[2093]
        + mono[2094];
    poly[7573] = poly[23] * poly[562] - poly[7572];
    poly[7574] = poly[24] * poly[562];
    poly[7575] = mono[2095]
        + mono[2096]
        + mono[2097]
        + mono[2098]
        + mono[2099]
        + mono[2100]
        + mono[2101]
        + mono[2102];
    poly[7576] = mono[2103]
        + mono[2104]
        + mono[2105]
        + mono[2106]
        + mono[2107]
        + mono[2108]
        + mono[2109]
        + mono[2110];
    poly[7577] = mono[2111]
        + mono[2112]
        + mono[2113]
        + mono[2114]
        + mono[2115]
        + mono[2116]
        + mono[2117]
        + mono[2118];
    poly[7578] = mono[2119]
        + mono[2120]
        + mono[2121]
        + mono[2122]
        + mono[2123]
        + mono[2124]
        + mono[2125]
        + mono[2126];
    poly[7579] = mono[2127] + mono[2128] + mono[2129] + mono[2130];
    poly[7580] = poly[1] * poly[563];
    poly[7581] = poly[2] * poly[563];
    poly[7582] = poly[3] * poly[563];
    poly[7583] = poly[4] * poly[563];
    poly[7584] = poly[5] * poly[563];
    poly[7585] = poly[6] * poly[563];
    poly[7586] = poly[7] * poly[563];
    poly[7587] = poly[8] * poly[563];
    poly[7588] = poly[9] * poly[563];
    poly[7589] = poly[10] * poly[563];
    poly[7590] = poly[11] * poly[563];
    poly[7591] = poly[12] * poly[563];
    poly[7592] = poly[13] * poly[563];
    poly[7593] = poly[14] * poly[563];
    poly[7594] = poly[15] * poly[563];
    poly[7595] = poly[16] * poly[563];
    poly[7596] = poly[17] * poly[563];
    poly[7597] = poly[18] * poly[563];
    poly[7598] = poly[19] * poly[563];
    poly[7599] = poly[20] * poly[563];
}

fn f_polynomials152(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7600] = poly[21] * poly[563];
    poly[7601] = poly[22] * poly[563];
    poly[7602] = poly[23] * poly[563];
    poly[7603] = poly[24] * poly[563];
    poly[7604] = poly[1] * poly[564];
    poly[7605] = poly[2] * poly[564];
    poly[7606] = poly[3] * poly[564];
    poly[7607] = poly[4] * poly[564];
    poly[7608] = poly[5] * poly[564];
    poly[7609] = poly[6] * poly[564];
    poly[7610] = poly[7] * poly[564];
    poly[7611] = poly[8] * poly[564];
    poly[7612] = poly[9] * poly[564];
    poly[7613] = poly[10] * poly[564];
    poly[7614] = poly[11] * poly[564];
    poly[7615] = mono[2131] + mono[2132];
    poly[7616] = mono[2133] + mono[2134];
    poly[7617] = mono[2135] + mono[2136];
    poly[7618] = mono[2137] + mono[2138];
    poly[7619] = mono[2139] + mono[2140] + mono[2141] + mono[2142];
    poly[7620] = poly[12] * poly[564] - poly[7615];
    poly[7621] = poly[13] * poly[564] - poly[7616];
    poly[7622] = poly[14] * poly[564] - poly[7617];
    poly[7623] = poly[15] * poly[564] - poly[7618];
    poly[7624] = poly[16] * poly[564] - poly[7619];
    poly[7625] = poly[17] * poly[564];
    poly[7626] = poly[18] * poly[564];
    poly[7627] = poly[19] * poly[564];
    poly[7628] = poly[20] * poly[564];
    poly[7629] = poly[21] * poly[564];
    poly[7630] = poly[22] * poly[564];
    poly[7631] = mono[2143] + mono[2144] + mono[2145] + mono[2146];
    poly[7632] = poly[23] * poly[564] - poly[7631];
    poly[7633] = poly[24] * poly[564];
    poly[7634] = poly[1] * poly[565];
    poly[7635] = poly[2] * poly[565];
    poly[7636] = poly[3] * poly[565];
    poly[7637] = poly[4] * poly[565];
    poly[7638] = poly[5] * poly[565];
    poly[7639] = poly[6] * poly[565];
    poly[7640] = poly[7] * poly[565];
    poly[7641] = poly[8] * poly[565];
    poly[7642] = poly[9] * poly[565];
    poly[7643] = poly[10] * poly[565];
    poly[7644] = poly[11] * poly[565];
    poly[7645] = poly[25] * poly[539] - poly[7423];
    poly[7646] = poly[25] * poly[540] - poly[7424];
    poly[7647] = poly[25] * poly[541] - poly[7425];
    poly[7648] = poly[25] * poly[542] - poly[7426];
    poly[7649] = poly[25] * poly[543] - poly[7427];
}

fn f_polynomials153(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7650] = poly[12] * poly[565] - poly[7645];
    poly[7651] = poly[13] * poly[565] - poly[7646];
    poly[7652] = poly[14] * poly[565] - poly[7647];
    poly[7653] = poly[15] * poly[565] - poly[7648];
    poly[7654] = poly[16] * poly[565] - poly[7649];
    poly[7655] = poly[17] * poly[565];
    poly[7656] = poly[18] * poly[565];
    poly[7657] = poly[19] * poly[565];
    poly[7658] = poly[20] * poly[565];
    poly[7659] = poly[21] * poly[565];
    poly[7660] = poly[22] * poly[565];
    poly[7661] = poly[25] * poly[555] - poly[7439];
    poly[7662] = poly[23] * poly[565] - poly[7661];
    poly[7663] = poly[24] * poly[565];
    poly[7664] = poly[399] * poly[30];
    poly[7665] = poly[25] * poly[559] - poly[7472];
    poly[7666] = poly[25] * poly[560] - poly[7503];
    poly[7667] = poly[25] * poly[561] - poly[7535];
    poly[7668] = poly[25] * poly[562] - poly[7575];
    poly[7669] = poly[25] * poly[563];
    poly[7670] = poly[25] * poly[564];
    poly[7671] = poly[1] * poly[566];
    poly[7672] = poly[2] * poly[566];
    poly[7673] = poly[3] * poly[566];
    poly[7674] = poly[4] * poly[566];
    poly[7675] = poly[5] * poly[566];
    poly[7676] = poly[6] * poly[566];
    poly[7677] = poly[7] * poly[566];
    poly[7678] = poly[8] * poly[566];
    poly[7679] = poly[9] * poly[566];
    poly[7680] = poly[10] * poly[566];
    poly[7681] = poly[11] * poly[566];
    poly[7682] = poly[26] * poly[539] - poly[7453];
    poly[7683] = poly[26] * poly[540] - poly[7454];
    poly[7684] = poly[26] * poly[541] - poly[7455];
    poly[7685] = poly[26] * poly[542] - poly[7456];
    poly[7686] = poly[26] * poly[543] - poly[7457];
    poly[7687] = poly[12] * poly[566] - poly[7682];
    poly[7688] = poly[13] * poly[566] - poly[7683];
    poly[7689] = poly[14] * poly[566] - poly[7684];
    poly[7690] = poly[15] * poly[566] - poly[7685];
    poly[7691] = poly[16] * poly[566] - poly[7686];
    poly[7692] = poly[17] * poly[566];
    poly[7693] = poly[18] * poly[566];
    poly[7694] = poly[19] * poly[566];
    poly[7695] = poly[20] * poly[566];
    poly[7696] = poly[21] * poly[566];
    poly[7697] = poly[22] * poly[566];
    poly[7698] = poly[26] * poly[555] - poly[7469];
    poly[7699] = poly[23] * poly[566] - poly[7698];
}

fn f_polynomials154(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7700] = poly[24] * poly[566];
    poly[7701] = poly[26] * poly[558] - poly[7472];
    poly[7702] = poly[425] * poly[30];
    poly[7703] = poly[26] * poly[560] - poly[7504];
    poly[7704] = poly[26] * poly[561] - poly[7536];
    poly[7705] = poly[26] * poly[562] - poly[7576];
    poly[7706] = poly[26] * poly[563];
    poly[7707] = poly[26] * poly[564];
    poly[7708] = poly[25] * poly[566] - poly[7701];
    poly[7709] = poly[1] * poly[567];
    poly[7710] = poly[2] * poly[567];
    poly[7711] = poly[3] * poly[567];
    poly[7712] = poly[4] * poly[567];
    poly[7713] = poly[5] * poly[567];
    poly[7714] = poly[6] * poly[567];
    poly[7715] = poly[7] * poly[567];
    poly[7716] = poly[8] * poly[567];
    poly[7717] = poly[9] * poly[567];
    poly[7718] = poly[10] * poly[567];
    poly[7719] = poly[11] * poly[567];
    poly[7720] = poly[27] * poly[539] - poly[7484];
    poly[7721] = poly[27] * poly[540] - poly[7485];
    poly[7722] = poly[27] * poly[541] - poly[7486];
    poly[7723] = poly[27] * poly[542] - poly[7487];
    poly[7724] = poly[27] * poly[543] - poly[7488];
    poly[7725] = poly[12] * poly[567] - poly[7720];
    poly[7726] = poly[13] * poly[567] - poly[7721];
    poly[7727] = poly[14] * poly[567] - poly[7722];
    poly[7728] = poly[15] * poly[567] - poly[7723];
    poly[7729] = poly[16] * poly[567] - poly[7724];
    poly[7730] = poly[17] * poly[567];
    poly[7731] = poly[18] * poly[567];
    poly[7732] = poly[19] * poly[567];
    poly[7733] = poly[20] * poly[567];
    poly[7734] = poly[21] * poly[567];
    poly[7735] = poly[22] * poly[567];
    poly[7736] = poly[27] * poly[555] - poly[7500];
    poly[7737] = poly[23] * poly[567] - poly[7736];
    poly[7738] = poly[24] * poly[567];
    poly[7739] = poly[27] * poly[558] - poly[7503];
    poly[7740] = poly[27] * poly[559] - poly[7504];
    poly[7741] = poly[453] * poly[30];
    poly[7742] = poly[27] * poly[561] - poly[7537];
    poly[7743] = poly[27] * poly[562] - poly[7577];
    poly[7744] = poly[27] * poly[563];
    poly[7745] = poly[27] * poly[564];
    poly[7746] = poly[25] * poly[567] - poly[7739];
    poly[7747] = poly[26] * poly[567] - poly[7740];
    poly[7748] = poly[1] * poly[568];
    poly[7749] = poly[2] * poly[568];
}

fn f_polynomials155(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7750] = poly[3] * poly[568];
    poly[7751] = poly[4] * poly[568];
    poly[7752] = poly[5] * poly[568];
    poly[7753] = poly[6] * poly[568];
    poly[7754] = poly[7] * poly[568];
    poly[7755] = poly[8] * poly[568];
    poly[7756] = poly[9] * poly[568];
    poly[7757] = poly[10] * poly[568];
    poly[7758] = poly[11] * poly[568];
    poly[7759] = poly[28] * poly[539] - poly[7516];
    poly[7760] = poly[28] * poly[540] - poly[7517];
    poly[7761] = poly[28] * poly[541] - poly[7518];
    poly[7762] = poly[28] * poly[542] - poly[7519];
    poly[7763] = poly[28] * poly[543] - poly[7520];
    poly[7764] = poly[12] * poly[568] - poly[7759];
    poly[7765] = poly[13] * poly[568] - poly[7760];
    poly[7766] = poly[14] * poly[568] - poly[7761];
    poly[7767] = poly[15] * poly[568] - poly[7762];
    poly[7768] = poly[16] * poly[568] - poly[7763];
    poly[7769] = poly[17] * poly[568];
    poly[7770] = poly[18] * poly[568];
    poly[7771] = poly[19] * poly[568];
    poly[7772] = poly[20] * poly[568];
    poly[7773] = poly[21] * poly[568];
    poly[7774] = poly[22] * poly[568];
    poly[7775] = poly[28] * poly[555] - poly[7532];
    poly[7776] = poly[23] * poly[568] - poly[7775];
    poly[7777] = poly[24] * poly[568];
    poly[7778] = poly[28] * poly[558] - poly[7535];
    poly[7779] = poly[28] * poly[559] - poly[7536];
    poly[7780] = poly[28] * poly[560] - poly[7537];
    poly[7781] = poly[483] * poly[30];
    poly[7782] = poly[28] * poly[562] - poly[7578];
    poly[7783] = poly[28] * poly[563];
    poly[7784] = poly[28] * poly[564];
    poly[7785] = poly[25] * poly[568] - poly[7778];
    poly[7786] = poly[26] * poly[568] - poly[7779];
    poly[7787] = poly[27] * poly[568] - poly[7780];
    poly[7788] = poly[1] * poly[569];
    poly[7789] = poly[2] * poly[569];
    poly[7790] = poly[3] * poly[569];
    poly[7791] = poly[4] * poly[569];
    poly[7792] = poly[5] * poly[569];
    poly[7793] = poly[6] * poly[569];
    poly[7794] = poly[30] * poly[493] - poly[7544];
    poly[7795] = poly[30] * poly[494] - poly[7545];
    poly[7796] = poly[30] * poly[495] - poly[7546];
    poly[7797] = poly[30] * poly[496] - poly[7547];
    poly[7798] = poly[7] * poly[569] - poly[7794];
    poly[7799] = poly[8] * poly[569] - poly[7795];
}

fn f_polynomials156(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7800] = poly[9] * poly[569] - poly[7796];
    poly[7801] = poly[10] * poly[569] - poly[7797];
    poly[7802] = poly[11] * poly[569];
    poly[7803] = poly[29] * poly[539] - poly[7553];
    poly[7804] = poly[29] * poly[540] - poly[7554];
    poly[7805] = poly[29] * poly[541] - poly[7555];
    poly[7806] = poly[29] * poly[542] - poly[7556];
    poly[7807] = mono[2147]
        + mono[2148]
        + mono[2149]
        + mono[2150]
        + mono[2151]
        + mono[2152]
        + mono[2153]
        + mono[2154];
    poly[7808] = poly[29] * poly[543] - poly[7807] - poly[7558] - poly[7557];
    poly[7809] = poly[12] * poly[569] - poly[7803];
    poly[7810] = poly[13] * poly[569] - poly[7804];
    poly[7811] = poly[14] * poly[569] - poly[7805];
    poly[7812] = poly[15] * poly[569] - poly[7806];
    poly[7813] = poly[30] * poly[506] - poly[7807] - poly[7563] - poly[7557];
    poly[7814] = poly[16] * poly[569] - poly[7813] - poly[7808] - poly[7807];
    poly[7815] = poly[17] * poly[569];
    poly[7816] = poly[18] * poly[569];
    poly[7817] = poly[19] * poly[569];
    poly[7818] = poly[20] * poly[569];
    poly[7819] = poly[21] * poly[569];
    poly[7820] = poly[30] * poly[513] - poly[7570];
    poly[7821] = poly[22] * poly[569] - poly[7820];
    poly[7822] = poly[29] * poly[555] - poly[7572];
    poly[7823] = poly[23] * poly[569] - poly[7822];
    poly[7824] = poly[24] * poly[569];
    poly[7825] = poly[29] * poly[558] - poly[7575];
    poly[7826] = poly[29] * poly[559] - poly[7576];
    poly[7827] = poly[29] * poly[560] - poly[7577];
    poly[7828] = poly[29] * poly[561] - poly[7578];
    poly[7829] = poly[30] * poly[521];
    poly[7830] = poly[30] * poly[522];
    poly[7831] = poly[29] * poly[563];
    poly[7832] = poly[29] * poly[564];
    poly[7833] = poly[25] * poly[569] - poly[7825];
    poly[7834] = poly[26] * poly[569] - poly[7826];
    poly[7835] = poly[27] * poly[569] - poly[7827];
    poly[7836] = poly[28] * poly[569] - poly[7828];
    poly[7837] = poly[30] * poly[527] - poly[7579];
    poly[7838] = poly[1] * poly[570];
    poly[7839] = poly[2] * poly[570];
    poly[7840] = poly[3] * poly[570];
    poly[7841] = poly[4] * poly[570];
    poly[7842] = poly[5] * poly[570];
    poly[7843] = poly[6] * poly[570];
    poly[7844] = poly[7] * poly[570];
    poly[7845] = poly[8] * poly[570];
    poly[7846] = poly[9] * poly[570];
    poly[7847] = poly[10] * poly[570];
    poly[7848] = poly[11] * poly[570];
    poly[7849] = poly[12] * poly[570];
}

fn f_polynomials157(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7850] = poly[13] * poly[570];
    poly[7851] = poly[14] * poly[570];
    poly[7852] = poly[15] * poly[570];
    poly[7853] = poly[16] * poly[570];
    poly[7854] = poly[17] * poly[570];
    poly[7855] = poly[18] * poly[570];
    poly[7856] = poly[19] * poly[570];
    poly[7857] = poly[20] * poly[570];
    poly[7858] = poly[21] * poly[570];
    poly[7859] = poly[22] * poly[570];
    poly[7860] = poly[23] * poly[570];
    poly[7861] = poly[24] * poly[570];
    poly[7862] = mono[2155] + mono[2156];
    poly[7863] = mono[2157] + mono[2158];
    poly[7864] = mono[2159] + mono[2160];
    poly[7865] = mono[2161] + mono[2162];
    poly[7866] = mono[2163] + mono[2164] + mono[2165] + mono[2166];
    poly[7867] = mono[2167] + mono[2168] + mono[2169] + mono[2170];
    poly[7868] = poly[25] * poly[570] - poly[7862];
    poly[7869] = poly[26] * poly[570] - poly[7863];
    poly[7870] = poly[27] * poly[570] - poly[7864];
    poly[7871] = poly[28] * poly[570] - poly[7865];
    poly[7872] = poly[29] * poly[570] - poly[7866];
    poly[7873] = poly[1] * poly[572];
    poly[7874] = poly[1] * poly[573];
    poly[7875] = poly[2] * poly[573];
    poly[7876] = poly[1] * poly[574];
    poly[7877] = poly[2] * poly[574];
    poly[7878] = poly[3] * poly[574];
    poly[7879] = poly[1] * poly[575];
    poly[7880] = poly[2] * poly[575];
    poly[7881] = poly[3] * poly[575];
    poly[7882] = poly[4] * poly[575];
    poly[7883] = poly[1] * poly[576];
    poly[7884] = poly[2] * poly[576];
    poly[7885] = poly[3] * poly[576];
    poly[7886] = poly[4] * poly[576];
    poly[7887] = poly[5] * poly[576];
    poly[7888] = poly[1] * poly[577];
    poly[7889] = poly[2] * poly[577];
    poly[7890] = poly[3] * poly[577];
    poly[7891] = poly[4] * poly[577];
    poly[7892] = poly[5] * poly[577];
    poly[7893] = poly[6] * poly[577];
    poly[7894] = poly[54] * poly[31];
    poly[7895] = poly[1] * poly[578];
    poly[7896] = poly[2] * poly[578];
    poly[7897] = poly[3] * poly[578];
    poly[7898] = poly[4] * poly[578];
    poly[7899] = poly[5] * poly[578];
}

fn f_polynomials158(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7900] = poly[6] * poly[578];
    poly[7901] = poly[31] * poly[61];
    poly[7902] = poly[62] * poly[31];
    poly[7903] = poly[31] * poly[63];
    poly[7904] = poly[1] * poly[579];
    poly[7905] = poly[2] * poly[579];
    poly[7906] = poly[3] * poly[579];
    poly[7907] = poly[4] * poly[579];
    poly[7908] = poly[5] * poly[579];
    poly[7909] = poly[6] * poly[579];
    poly[7910] = poly[31] * poly[70];
    poly[7911] = poly[31] * poly[71];
    poly[7912] = poly[72] * poly[31];
    poly[7913] = poly[31] * poly[73];
    poly[7914] = poly[31] * poly[74];
    poly[7915] = poly[1] * poly[580];
    poly[7916] = poly[2] * poly[580];
    poly[7917] = poly[3] * poly[580];
    poly[7918] = poly[4] * poly[580];
    poly[7919] = poly[5] * poly[580];
    poly[7920] = poly[6] * poly[580];
    poly[7921] = poly[31] * poly[81];
    poly[7922] = poly[31] * poly[82];
    poly[7923] = poly[31] * poly[83];
    poly[7924] = poly[84] * poly[31];
    poly[7925] = poly[31] * poly[85];
    poly[7926] = poly[31] * poly[86];
    poly[7927] = poly[31] * poly[87];
    poly[7928] = poly[1] * poly[581];
    poly[7929] = poly[2] * poly[581];
    poly[7930] = poly[3] * poly[581];
    poly[7931] = poly[4] * poly[581];
    poly[7932] = poly[5] * poly[581];
    poly[7933] = poly[6] * poly[581];
    poly[7934] = poly[11] * poly[577];
    poly[7935] = poly[11] * poly[578];
    poly[7936] = poly[11] * poly[579];
    poly[7937] = poly[11] * poly[580];
    poly[7938] = poly[1] * poly[582];
    poly[7939] = poly[2] * poly[582];
    poly[7940] = poly[3] * poly[582];
    poly[7941] = poly[4] * poly[582];
    poly[7942] = poly[5] * poly[582];
    poly[7943] = poly[6] * poly[582];
    poly[7944] = poly[7] * poly[582];
    poly[7945] = poly[8] * poly[582];
    poly[7946] = poly[9] * poly[582];
    poly[7947] = poly[10] * poly[582];
    poly[7948] = poly[11] * poly[582];
    poly[7949] = poly[109] * poly[31];
}

fn f_polynomials159(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[7950] = poly[1] * poly[583];
    poly[7951] = poly[2] * poly[583];
    poly[7952] = poly[3] * poly[583];
    poly[7953] = poly[4] * poly[583];
    poly[7954] = poly[5] * poly[583];
    poly[7955] = poly[6] * poly[583];
    poly[7956] = poly[7] * poly[583];
    poly[7957] = poly[8] * poly[583];
    poly[7958] = poly[9] * poly[583];
    poly[7959] = poly[10] * poly[583];
    poly[7960] = poly[11] * poly[583];
    poly[7961] = poly[31] * poly[121];
    poly[7962] = poly[122] * poly[31];
    poly[7963] = poly[31] * poly[123];
    poly[7964] = poly[1] * poly[584];
    poly[7965] = poly[2] * poly[584];
    poly[7966] = poly[3] * poly[584];
    poly[7967] = poly[4] * poly[584];
    poly[7968] = poly[5] * poly[584];
    poly[7969] = poly[6] * poly[584];
    poly[7970] = poly[7] * poly[584];
    poly[7971] = poly[8] * poly[584];
    poly[7972] = poly[9] * poly[584];
    poly[7973] = poly[10] * poly[584];
    poly[7974] = poly[11] * poly[584];
    poly[7975] = poly[31] * poly[135];
    poly[7976] = poly[31] * poly[136];
    poly[7977] = poly[137] * poly[31];
    poly[7978] = poly[31] * poly[138];
    poly[7979] = poly[31] * poly[139];
    poly[7980] = poly[1] * poly[585];
    poly[7981] = poly[2] * poly[585];
    poly[7982] = poly[3] * poly[585];
    poly[7983] = poly[4] * poly[585];
    poly[7984] = poly[5] * poly[585];
    poly[7985] = poly[6] * poly[585];
    poly[7986] = poly[7] * poly[585];
    poly[7987] = poly[8] * poly[585];
    poly[7988] = poly[9] * poly[585];
    poly[7989] = poly[10] * poly[585];
    poly[7990] = poly[11] * poly[585];
    poly[7991] = poly[31] * poly[151];
    poly[7992] = poly[31] * poly[152];
    poly[7993] = poly[31] * poly[153];
    poly[7994] = poly[154] * poly[31];
    poly[7995] = poly[31] * poly[155];
    poly[7996] = poly[31] * poly[156];
    poly[7997] = poly[31] * poly[157];
    poly[7998] = poly[1] * poly[586];
    poly[7999] = poly[2] * poly[586];
}

fn f_polynomials160(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8000] = poly[3] * poly[586];
    poly[8001] = poly[4] * poly[586];
    poly[8002] = poly[5] * poly[586];
    poly[8003] = poly[6] * poly[586];
    poly[8004] = poly[31] * poly[164];
    poly[8005] = poly[31] * poly[165];
    poly[8006] = poly[31] * poly[166];
    poly[8007] = poly[31] * poly[167];
    poly[8008] = poly[31] * poly[168];
    poly[8009] = poly[31] * poly[169];
    poly[8010] = poly[31] * poly[170];
    poly[8011] = poly[31] * poly[171];
    poly[8012] = poly[11] * poly[586];
    poly[8013] = poly[31] * poly[173];
    poly[8014] = poly[31] * poly[174];
    poly[8015] = poly[31] * poly[175];
    poly[8016] = poly[31] * poly[176];
    poly[8017] = poly[31] * poly[177];
    poly[8018] = poly[31] * poly[178];
    poly[8019] = poly[31] * poly[179];
    poly[8020] = poly[31] * poly[180];
    poly[8021] = poly[31] * poly[181];
    poly[8022] = poly[31] * poly[182];
    poly[8023] = poly[31] * poly[183];
    poly[8024] = poly[1] * poly[587];
    poly[8025] = poly[2] * poly[587];
    poly[8026] = poly[3] * poly[587];
    poly[8027] = poly[4] * poly[587];
    poly[8028] = poly[5] * poly[587];
    poly[8029] = poly[6] * poly[587];
    poly[8030] = poly[17] * poly[577];
    poly[8031] = poly[17] * poly[578];
    poly[8032] = poly[17] * poly[579];
    poly[8033] = poly[17] * poly[580];
    poly[8034] = poly[11] * poly[587];
    poly[8035] = poly[17] * poly[582];
    poly[8036] = poly[17] * poly[583];
    poly[8037] = poly[17] * poly[584];
    poly[8038] = poly[17] * poly[585];
    poly[8039] = poly[17] * poly[586];
    poly[8040] = poly[1] * poly[588];
    poly[8041] = poly[2] * poly[588];
    poly[8042] = poly[3] * poly[588];
    poly[8043] = poly[4] * poly[588];
    poly[8044] = poly[5] * poly[588];
    poly[8045] = poly[6] * poly[588];
    poly[8046] = poly[7] * poly[588];
    poly[8047] = poly[8] * poly[588];
    poly[8048] = poly[9] * poly[588];
    poly[8049] = poly[10] * poly[588];
}

fn f_polynomials161(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8050] = poly[11] * poly[588];
    poly[8051] = poly[12] * poly[588];
    poly[8052] = poly[13] * poly[588];
    poly[8053] = poly[14] * poly[588];
    poly[8054] = poly[15] * poly[588];
    poly[8055] = poly[16] * poly[588];
    poly[8056] = poly[17] * poly[588];
    poly[8057] = poly[1] * poly[589];
    poly[8058] = poly[2] * poly[589];
    poly[8059] = poly[3] * poly[589];
    poly[8060] = poly[4] * poly[589];
    poly[8061] = poly[5] * poly[589];
    poly[8062] = poly[6] * poly[589];
    poly[8063] = poly[7] * poly[589];
    poly[8064] = poly[8] * poly[589];
    poly[8065] = poly[9] * poly[589];
    poly[8066] = poly[10] * poly[589];
    poly[8067] = poly[11] * poly[589];
    poly[8068] = poly[12] * poly[589];
    poly[8069] = poly[13] * poly[589];
    poly[8070] = poly[14] * poly[589];
    poly[8071] = poly[15] * poly[589];
    poly[8072] = poly[16] * poly[589];
    poly[8073] = poly[17] * poly[589];
    poly[8074] = mono[2171] + mono[2172] + mono[2173] + mono[2174];
    poly[8075] = poly[1] * poly[590];
    poly[8076] = poly[2] * poly[590];
    poly[8077] = poly[3] * poly[590];
    poly[8078] = poly[4] * poly[590];
    poly[8079] = poly[5] * poly[590];
    poly[8080] = poly[6] * poly[590];
    poly[8081] = poly[7] * poly[590];
    poly[8082] = poly[8] * poly[590];
    poly[8083] = poly[9] * poly[590];
    poly[8084] = poly[10] * poly[590];
    poly[8085] = poly[11] * poly[590];
    poly[8086] = poly[12] * poly[590];
    poly[8087] = poly[13] * poly[590];
    poly[8088] = poly[14] * poly[590];
    poly[8089] = poly[15] * poly[590];
    poly[8090] = poly[16] * poly[590];
    poly[8091] = poly[17] * poly[590];
    poly[8092] = mono[2175] + mono[2176] + mono[2177] + mono[2178];
    poly[8093] = mono[2179] + mono[2180] + mono[2181] + mono[2182];
    poly[8094] = poly[1] * poly[591];
    poly[8095] = poly[2] * poly[591];
    poly[8096] = poly[3] * poly[591];
    poly[8097] = poly[4] * poly[591];
    poly[8098] = poly[5] * poly[591];
    poly[8099] = poly[6] * poly[591];
}

fn f_polynomials162(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8100] = poly[7] * poly[591];
    poly[8101] = poly[8] * poly[591];
    poly[8102] = poly[9] * poly[591];
    poly[8103] = poly[10] * poly[591];
    poly[8104] = poly[11] * poly[591];
    poly[8105] = poly[12] * poly[591];
    poly[8106] = poly[13] * poly[591];
    poly[8107] = poly[14] * poly[591];
    poly[8108] = poly[15] * poly[591];
    poly[8109] = poly[16] * poly[591];
    poly[8110] = poly[17] * poly[591];
    poly[8111] = mono[2183] + mono[2184] + mono[2185] + mono[2186];
    poly[8112] = mono[2187] + mono[2188] + mono[2189] + mono[2190];
    poly[8113] = mono[2191] + mono[2192] + mono[2193] + mono[2194];
    poly[8114] = poly[1] * poly[592];
    poly[8115] = poly[2] * poly[592];
    poly[8116] = poly[3] * poly[592];
    poly[8117] = poly[4] * poly[592];
    poly[8118] = poly[5] * poly[592];
    poly[8119] = poly[6] * poly[592];
    poly[8120] = mono[2195]
        + mono[2196]
        + mono[2197]
        + mono[2198]
        + mono[2199]
        + mono[2200]
        + mono[2201]
        + mono[2202];
    poly[8121] = mono[2203]
        + mono[2204]
        + mono[2205]
        + mono[2206]
        + mono[2207]
        + mono[2208]
        + mono[2209]
        + mono[2210];
    poly[8122] = mono[2211]
        + mono[2212]
        + mono[2213]
        + mono[2214]
        + mono[2215]
        + mono[2216]
        + mono[2217]
        + mono[2218];
    poly[8123] = mono[2219]
        + mono[2220]
        + mono[2221]
        + mono[2222]
        + mono[2223]
        + mono[2224]
        + mono[2225]
        + mono[2226];
    poly[8124] = poly[7] * poly[592] - poly[8120];
    poly[8125] = poly[8] * poly[592] - poly[8121];
    poly[8126] = poly[9] * poly[592] - poly[8122];
    poly[8127] = poly[10] * poly[592] - poly[8123];
    poly[8128] = poly[11] * poly[592];
    poly[8129] = poly[12] * poly[592];
    poly[8130] = poly[13] * poly[592];
    poly[8131] = poly[14] * poly[592];
    poly[8132] = poly[15] * poly[592];
    poly[8133] = mono[2227]
        + mono[2228]
        + mono[2229]
        + mono[2230]
        + mono[2231]
        + mono[2232]
        + mono[2233]
        + mono[2234]
        + mono[2235]
        + mono[2236]
        + mono[2237]
        + mono[2238]
        + mono[2239]
        + mono[2240]
        + mono[2241]
        + mono[2242];
    poly[8134] = poly[16] * poly[592] - poly[8133];
    poly[8135] = poly[17] * poly[592];
    poly[8136] = mono[2243]
        + mono[2244]
        + mono[2245]
        + mono[2246]
        + mono[2247]
        + mono[2248]
        + mono[2249]
        + mono[2250];
    poly[8137] = mono[2251]
        + mono[2252]
        + mono[2253]
        + mono[2254]
        + mono[2255]
        + mono[2256]
        + mono[2257]
        + mono[2258];
    poly[8138] = mono[2259]
        + mono[2260]
        + mono[2261]
        + mono[2262]
        + mono[2263]
        + mono[2264]
        + mono[2265]
        + mono[2266];
    poly[8139] = mono[2267]
        + mono[2268]
        + mono[2269]
        + mono[2270]
        + mono[2271]
        + mono[2272]
        + mono[2273]
        + mono[2274];
    poly[8140] = mono[2275] + mono[2276] + mono[2277] + mono[2278];
    poly[8141] = poly[1] * poly[593];
    poly[8142] = poly[2] * poly[593];
    poly[8143] = poly[3] * poly[593];
    poly[8144] = poly[4] * poly[593];
    poly[8145] = poly[5] * poly[593];
    poly[8146] = poly[6] * poly[593];
    poly[8147] = poly[7] * poly[593];
    poly[8148] = poly[8] * poly[593];
    poly[8149] = poly[9] * poly[593];
}

fn f_polynomials163(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8150] = poly[10] * poly[593];
    poly[8151] = poly[11] * poly[593];
    poly[8152] = mono[2279]
        + mono[2280]
        + mono[2281]
        + mono[2282]
        + mono[2283]
        + mono[2284]
        + mono[2285]
        + mono[2286];
    poly[8153] = mono[2287]
        + mono[2288]
        + mono[2289]
        + mono[2290]
        + mono[2291]
        + mono[2292]
        + mono[2293]
        + mono[2294];
    poly[8154] = mono[2295]
        + mono[2296]
        + mono[2297]
        + mono[2298]
        + mono[2299]
        + mono[2300]
        + mono[2301]
        + mono[2302];
    poly[8155] = mono[2303]
        + mono[2304]
        + mono[2305]
        + mono[2306]
        + mono[2307]
        + mono[2308]
        + mono[2309]
        + mono[2310];
    poly[8156] = mono[2311]
        + mono[2312]
        + mono[2313]
        + mono[2314]
        + mono[2315]
        + mono[2316]
        + mono[2317]
        + mono[2318]
        + mono[2319]
        + mono[2320]
        + mono[2321]
        + mono[2322]
        + mono[2323]
        + mono[2324]
        + mono[2325]
        + mono[2326];
    poly[8157] = poly[12] * poly[593] - poly[8152];
    poly[8158] = poly[13] * poly[593] - poly[8153];
    poly[8159] = poly[14] * poly[593] - poly[8154];
    poly[8160] = poly[15] * poly[593] - poly[8155];
    poly[8161] = poly[16] * poly[593] - poly[8156];
    poly[8162] = poly[17] * poly[593];
    poly[8163] = mono[2327]
        + mono[2328]
        + mono[2329]
        + mono[2330]
        + mono[2331]
        + mono[2332]
        + mono[2333]
        + mono[2334];
    poly[8164] = mono[2335]
        + mono[2336]
        + mono[2337]
        + mono[2338]
        + mono[2339]
        + mono[2340]
        + mono[2341]
        + mono[2342];
    poly[8165] = mono[2343]
        + mono[2344]
        + mono[2345]
        + mono[2346]
        + mono[2347]
        + mono[2348]
        + mono[2349]
        + mono[2350];
    poly[8166] = mono[2351]
        + mono[2352]
        + mono[2353]
        + mono[2354]
        + mono[2355]
        + mono[2356]
        + mono[2357]
        + mono[2358];
    poly[8167] = mono[2359]
        + mono[2360]
        + mono[2361]
        + mono[2362]
        + mono[2363]
        + mono[2364]
        + mono[2365]
        + mono[2366]
        + mono[2367]
        + mono[2368]
        + mono[2369]
        + mono[2370]
        + mono[2371]
        + mono[2372]
        + mono[2373]
        + mono[2374];
    poly[8168] = mono[2375] + mono[2376] + mono[2377] + mono[2378];
    poly[8169] = poly[1] * poly[594];
    poly[8170] = poly[2] * poly[594];
    poly[8171] = poly[3] * poly[594];
    poly[8172] = poly[4] * poly[594];
    poly[8173] = poly[5] * poly[594];
    poly[8174] = poly[6] * poly[594];
    poly[8175] = poly[7] * poly[594];
    poly[8176] = poly[8] * poly[594];
    poly[8177] = poly[9] * poly[594];
    poly[8178] = poly[10] * poly[594];
    poly[8179] = poly[11] * poly[594];
    poly[8180] = poly[12] * poly[594];
    poly[8181] = poly[13] * poly[594];
    poly[8182] = poly[14] * poly[594];
    poly[8183] = poly[15] * poly[594];
    poly[8184] = poly[16] * poly[594];
    poly[8185] = poly[17] * poly[594];
    poly[8186] = poly[217] * poly[31];
    poly[8187] = poly[18] * poly[589] - poly[8074];
    poly[8188] = poly[18] * poly[590] - poly[8092];
    poly[8189] = poly[18] * poly[591] - poly[8111];
    poly[8190] = poly[18] * poly[592] - poly[8136];
    poly[8191] = poly[18] * poly[593] - poly[8163];
    poly[8192] = poly[1] * poly[595];
    poly[8193] = poly[2] * poly[595];
    poly[8194] = poly[3] * poly[595];
    poly[8195] = poly[4] * poly[595];
    poly[8196] = poly[5] * poly[595];
    poly[8197] = poly[6] * poly[595];
    poly[8198] = poly[7] * poly[595];
    poly[8199] = poly[8] * poly[595];
}

fn f_polynomials164(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8200] = poly[9] * poly[595];
    poly[8201] = poly[10] * poly[595];
    poly[8202] = poly[11] * poly[595];
    poly[8203] = poly[12] * poly[595];
    poly[8204] = poly[13] * poly[595];
    poly[8205] = poly[14] * poly[595];
    poly[8206] = poly[15] * poly[595];
    poly[8207] = poly[16] * poly[595];
    poly[8208] = poly[17] * poly[595];
    poly[8209] = poly[19] * poly[588] - poly[8074];
    poly[8210] = poly[236] * poly[31];
    poly[8211] = poly[19] * poly[590] - poly[8093];
    poly[8212] = poly[19] * poly[591] - poly[8112];
    poly[8213] = poly[19] * poly[592] - poly[8137];
    poly[8214] = poly[19] * poly[593] - poly[8164];
    poly[8215] = poly[18] * poly[595] - poly[8209];
    poly[8216] = poly[1] * poly[596];
    poly[8217] = poly[2] * poly[596];
    poly[8218] = poly[3] * poly[596];
    poly[8219] = poly[4] * poly[596];
    poly[8220] = poly[5] * poly[596];
    poly[8221] = poly[6] * poly[596];
    poly[8222] = poly[7] * poly[596];
    poly[8223] = poly[8] * poly[596];
    poly[8224] = poly[9] * poly[596];
    poly[8225] = poly[10] * poly[596];
    poly[8226] = poly[11] * poly[596];
    poly[8227] = poly[12] * poly[596];
    poly[8228] = poly[13] * poly[596];
    poly[8229] = poly[14] * poly[596];
    poly[8230] = poly[15] * poly[596];
    poly[8231] = poly[16] * poly[596];
    poly[8232] = poly[17] * poly[596];
    poly[8233] = poly[20] * poly[588] - poly[8092];
    poly[8234] = poly[20] * poly[589] - poly[8093];
    poly[8235] = poly[257] * poly[31];
    poly[8236] = poly[20] * poly[591] - poly[8113];
    poly[8237] = poly[20] * poly[592] - poly[8138];
    poly[8238] = poly[20] * poly[593] - poly[8165];
    poly[8239] = poly[18] * poly[596] - poly[8233];
    poly[8240] = poly[19] * poly[596] - poly[8234];
    poly[8241] = poly[1] * poly[597];
    poly[8242] = poly[2] * poly[597];
    poly[8243] = poly[3] * poly[597];
    poly[8244] = poly[4] * poly[597];
    poly[8245] = poly[5] * poly[597];
    poly[8246] = poly[6] * poly[597];
    poly[8247] = poly[7] * poly[597];
    poly[8248] = poly[8] * poly[597];
    poly[8249] = poly[9] * poly[597];
}

fn f_polynomials165(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8250] = poly[10] * poly[597];
    poly[8251] = poly[11] * poly[597];
    poly[8252] = poly[12] * poly[597];
    poly[8253] = poly[13] * poly[597];
    poly[8254] = poly[14] * poly[597];
    poly[8255] = poly[15] * poly[597];
    poly[8256] = poly[16] * poly[597];
    poly[8257] = poly[17] * poly[597];
    poly[8258] = poly[21] * poly[588] - poly[8111];
    poly[8259] = poly[21] * poly[589] - poly[8112];
    poly[8260] = poly[21] * poly[590] - poly[8113];
    poly[8261] = poly[280] * poly[31];
    poly[8262] = poly[21] * poly[592] - poly[8139];
    poly[8263] = poly[21] * poly[593] - poly[8166];
    poly[8264] = poly[18] * poly[597] - poly[8258];
    poly[8265] = poly[19] * poly[597] - poly[8259];
    poly[8266] = poly[20] * poly[597] - poly[8260];
    poly[8267] = poly[1] * poly[598];
    poly[8268] = poly[2] * poly[598];
    poly[8269] = poly[3] * poly[598];
    poly[8270] = poly[4] * poly[598];
    poly[8271] = poly[5] * poly[598];
    poly[8272] = poly[6] * poly[598];
    poly[8273] = poly[31] * poly[290] - poly[8120];
    poly[8274] = poly[31] * poly[291] - poly[8121];
    poly[8275] = poly[31] * poly[292] - poly[8122];
    poly[8276] = poly[31] * poly[293] - poly[8123];
    poly[8277] = poly[7] * poly[598] - poly[8273];
    poly[8278] = poly[8] * poly[598] - poly[8274];
    poly[8279] = poly[9] * poly[598] - poly[8275];
    poly[8280] = poly[10] * poly[598] - poly[8276];
    poly[8281] = poly[11] * poly[598];
    poly[8282] = poly[12] * poly[598];
    poly[8283] = poly[13] * poly[598];
    poly[8284] = poly[14] * poly[598];
    poly[8285] = poly[15] * poly[598];
    poly[8286] = poly[31] * poly[303] - poly[8133];
    poly[8287] = poly[16] * poly[598] - poly[8286];
    poly[8288] = poly[17] * poly[598];
    poly[8289] = poly[22] * poly[588] - poly[8136];
    poly[8290] = poly[22] * poly[589] - poly[8137];
    poly[8291] = poly[22] * poly[590] - poly[8138];
    poly[8292] = poly[22] * poly[591] - poly[8139];
    poly[8293] = poly[31] * poly[310];
    poly[8294] = poly[31] * poly[311];
    poly[8295] = poly[22] * poly[593] - poly[8167];
    poly[8296] = poly[18] * poly[598] - poly[8289];
    poly[8297] = poly[19] * poly[598] - poly[8290];
    poly[8298] = poly[20] * poly[598] - poly[8291];
    poly[8299] = poly[21] * poly[598] - poly[8292];
}

fn f_polynomials166(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8300] = poly[31] * poly[316] - poly[8140];
    poly[8301] = poly[1] * poly[599];
    poly[8302] = poly[2] * poly[599];
    poly[8303] = poly[3] * poly[599];
    poly[8304] = poly[4] * poly[599];
    poly[8305] = poly[5] * poly[599];
    poly[8306] = poly[6] * poly[599];
    poly[8307] = poly[7] * poly[599];
    poly[8308] = poly[8] * poly[599];
    poly[8309] = poly[9] * poly[599];
    poly[8310] = poly[10] * poly[599];
    poly[8311] = poly[11] * poly[599];
    poly[8312] = poly[31] * poly[328] - poly[8152];
    poly[8313] = poly[31] * poly[329] - poly[8153];
    poly[8314] = poly[31] * poly[330] - poly[8154];
    poly[8315] = poly[31] * poly[331] - poly[8155];
    poly[8316] = poly[31] * poly[332] - poly[8156];
    poly[8317] = poly[12] * poly[599] - poly[8312];
    poly[8318] = poly[13] * poly[599] - poly[8313];
    poly[8319] = poly[14] * poly[599] - poly[8314];
    poly[8320] = poly[15] * poly[599] - poly[8315];
    poly[8321] = poly[16] * poly[599] - poly[8316];
    poly[8322] = poly[17] * poly[599];
    poly[8323] = poly[23] * poly[588] - poly[8163];
    poly[8324] = poly[23] * poly[589] - poly[8164];
    poly[8325] = poly[23] * poly[590] - poly[8165];
    poly[8326] = poly[23] * poly[591] - poly[8166];
    poly[8327] = poly[23] * poly[592] - poly[8167];
    poly[8328] = poly[31] * poly[344];
    poly[8329] = poly[31] * poly[345];
    poly[8330] = poly[18] * poly[599] - poly[8323];
    poly[8331] = poly[19] * poly[599] - poly[8324];
    poly[8332] = poly[20] * poly[599] - poly[8325];
    poly[8333] = poly[21] * poly[599] - poly[8326];
    poly[8334] = poly[22] * poly[599] - poly[8327];
    poly[8335] = poly[31] * poly[351] - poly[8168];
    poly[8336] = poly[1] * poly[600];
    poly[8337] = poly[2] * poly[600];
    poly[8338] = poly[3] * poly[600];
    poly[8339] = poly[4] * poly[600];
    poly[8340] = poly[5] * poly[600];
    poly[8341] = poly[6] * poly[600];
    poly[8342] = poly[24] * poly[577];
    poly[8343] = poly[24] * poly[578];
    poly[8344] = poly[24] * poly[579];
    poly[8345] = poly[24] * poly[580];
    poly[8346] = poly[11] * poly[600];
    poly[8347] = poly[24] * poly[582];
    poly[8348] = poly[24] * poly[583];
    poly[8349] = poly[24] * poly[584];
}

fn f_polynomials167(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8350] = poly[24] * poly[585];
    poly[8351] = poly[24] * poly[586];
    poly[8352] = poly[17] * poly[600];
    poly[8353] = poly[24] * poly[588];
    poly[8354] = poly[24] * poly[589];
    poly[8355] = poly[24] * poly[590];
    poly[8356] = poly[24] * poly[591];
    poly[8357] = poly[24] * poly[592];
    poly[8358] = poly[24] * poly[593];
    poly[8359] = poly[24] * poly[594];
    poly[8360] = poly[24] * poly[595];
    poly[8361] = poly[24] * poly[596];
    poly[8362] = poly[24] * poly[597];
    poly[8363] = poly[24] * poly[598];
    poly[8364] = poly[24] * poly[599];
    poly[8365] = poly[1] * poly[601];
    poly[8366] = poly[2] * poly[601];
    poly[8367] = poly[3] * poly[601];
    poly[8368] = poly[4] * poly[601];
    poly[8369] = poly[5] * poly[601];
    poly[8370] = poly[6] * poly[601];
    poly[8371] = poly[7] * poly[601];
    poly[8372] = poly[8] * poly[601];
    poly[8373] = poly[9] * poly[601];
    poly[8374] = poly[10] * poly[601];
    poly[8375] = poly[11] * poly[601];
    poly[8376] = poly[12] * poly[601];
    poly[8377] = poly[13] * poly[601];
    poly[8378] = poly[14] * poly[601];
    poly[8379] = poly[15] * poly[601];
    poly[8380] = poly[16] * poly[601];
    poly[8381] = poly[17] * poly[601];
    poly[8382] = mono[2379] + mono[2380] + mono[2381] + mono[2382];
    poly[8383] = mono[2383] + mono[2384] + mono[2385] + mono[2386];
    poly[8384] = mono[2387] + mono[2388] + mono[2389] + mono[2390];
    poly[8385] = mono[2391] + mono[2392] + mono[2393] + mono[2394];
    poly[8386] = mono[2395]
        + mono[2396]
        + mono[2397]
        + mono[2398]
        + mono[2399]
        + mono[2400]
        + mono[2401]
        + mono[2402];
    poly[8387] = mono[2403]
        + mono[2404]
        + mono[2405]
        + mono[2406]
        + mono[2407]
        + mono[2408]
        + mono[2409]
        + mono[2410];
    poly[8388] = poly[18] * poly[601] - poly[8382];
    poly[8389] = poly[19] * poly[601] - poly[8383];
    poly[8390] = poly[20] * poly[601] - poly[8384];
    poly[8391] = poly[21] * poly[601] - poly[8385];
    poly[8392] = poly[22] * poly[601] - poly[8386];
    poly[8393] = poly[23] * poly[601] - poly[8387];
    poly[8394] = poly[24] * poly[601];
    poly[8395] = poly[1] * poly[602];
    poly[8396] = poly[2] * poly[602];
    poly[8397] = poly[3] * poly[602];
    poly[8398] = poly[4] * poly[602];
    poly[8399] = poly[5] * poly[602];
}

fn f_polynomials168(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8400] = poly[6] * poly[602];
    poly[8401] = poly[7] * poly[602];
    poly[8402] = poly[8] * poly[602];
    poly[8403] = poly[9] * poly[602];
    poly[8404] = poly[10] * poly[602];
    poly[8405] = poly[11] * poly[602];
    poly[8406] = poly[12] * poly[602];
    poly[8407] = poly[13] * poly[602];
    poly[8408] = poly[14] * poly[602];
    poly[8409] = poly[15] * poly[602];
    poly[8410] = poly[16] * poly[602];
    poly[8411] = poly[17] * poly[602];
    poly[8412] = mono[2411] + mono[2412] + mono[2413] + mono[2414];
    poly[8413] = mono[2415] + mono[2416] + mono[2417] + mono[2418];
    poly[8414] = mono[2419] + mono[2420] + mono[2421] + mono[2422];
    poly[8415] = mono[2423] + mono[2424] + mono[2425] + mono[2426];
    poly[8416] = mono[2427]
        + mono[2428]
        + mono[2429]
        + mono[2430]
        + mono[2431]
        + mono[2432]
        + mono[2433]
        + mono[2434];
    poly[8417] = mono[2435]
        + mono[2436]
        + mono[2437]
        + mono[2438]
        + mono[2439]
        + mono[2440]
        + mono[2441]
        + mono[2442];
    poly[8418] = poly[18] * poly[602] - poly[8412];
    poly[8419] = poly[19] * poly[602] - poly[8413];
    poly[8420] = poly[20] * poly[602] - poly[8414];
    poly[8421] = poly[21] * poly[602] - poly[8415];
    poly[8422] = poly[22] * poly[602] - poly[8416];
    poly[8423] = poly[23] * poly[602] - poly[8417];
    poly[8424] = poly[24] * poly[602];
    poly[8425] = mono[2443] + mono[2444] + mono[2445] + mono[2446];
    poly[8426] = poly[1] * poly[603];
    poly[8427] = poly[2] * poly[603];
    poly[8428] = poly[3] * poly[603];
    poly[8429] = poly[4] * poly[603];
    poly[8430] = poly[5] * poly[603];
    poly[8431] = poly[6] * poly[603];
    poly[8432] = poly[7] * poly[603];
    poly[8433] = poly[8] * poly[603];
    poly[8434] = poly[9] * poly[603];
    poly[8435] = poly[10] * poly[603];
    poly[8436] = poly[11] * poly[603];
    poly[8437] = poly[12] * poly[603];
    poly[8438] = poly[13] * poly[603];
    poly[8439] = poly[14] * poly[603];
    poly[8440] = poly[15] * poly[603];
    poly[8441] = poly[16] * poly[603];
    poly[8442] = poly[17] * poly[603];
    poly[8443] = mono[2447] + mono[2448] + mono[2449] + mono[2450];
    poly[8444] = mono[2451] + mono[2452] + mono[2453] + mono[2454];
    poly[8445] = mono[2455] + mono[2456] + mono[2457] + mono[2458];
    poly[8446] = mono[2459] + mono[2460] + mono[2461] + mono[2462];
    poly[8447] = mono[2463]
        + mono[2464]
        + mono[2465]
        + mono[2466]
        + mono[2467]
        + mono[2468]
        + mono[2469]
        + mono[2470];
    poly[8448] = mono[2471]
        + mono[2472]
        + mono[2473]
        + mono[2474]
        + mono[2475]
        + mono[2476]
        + mono[2477]
        + mono[2478];
    poly[8449] = poly[18] * poly[603] - poly[8443];
}

fn f_polynomials169(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8450] = poly[19] * poly[603] - poly[8444];
    poly[8451] = poly[20] * poly[603] - poly[8445];
    poly[8452] = poly[21] * poly[603] - poly[8446];
    poly[8453] = poly[22] * poly[603] - poly[8447];
    poly[8454] = poly[23] * poly[603] - poly[8448];
    poly[8455] = poly[24] * poly[603];
    poly[8456] = mono[2479] + mono[2480] + mono[2481] + mono[2482];
    poly[8457] = mono[2483] + mono[2484] + mono[2485] + mono[2486];
    poly[8458] = poly[1] * poly[604];
    poly[8459] = poly[2] * poly[604];
    poly[8460] = poly[3] * poly[604];
    poly[8461] = poly[4] * poly[604];
    poly[8462] = poly[5] * poly[604];
    poly[8463] = poly[6] * poly[604];
    poly[8464] = poly[7] * poly[604];
    poly[8465] = poly[8] * poly[604];
    poly[8466] = poly[9] * poly[604];
    poly[8467] = poly[10] * poly[604];
    poly[8468] = poly[11] * poly[604];
    poly[8469] = poly[12] * poly[604];
    poly[8470] = poly[13] * poly[604];
    poly[8471] = poly[14] * poly[604];
    poly[8472] = poly[15] * poly[604];
    poly[8473] = poly[16] * poly[604];
    poly[8474] = poly[17] * poly[604];
    poly[8475] = mono[2487] + mono[2488] + mono[2489] + mono[2490];
    poly[8476] = mono[2491] + mono[2492] + mono[2493] + mono[2494];
    poly[8477] = mono[2495] + mono[2496] + mono[2497] + mono[2498];
    poly[8478] = mono[2499] + mono[2500] + mono[2501] + mono[2502];
    poly[8479] = mono[2503]
        + mono[2504]
        + mono[2505]
        + mono[2506]
        + mono[2507]
        + mono[2508]
        + mono[2509]
        + mono[2510];
    poly[8480] = mono[2511]
        + mono[2512]
        + mono[2513]
        + mono[2514]
        + mono[2515]
        + mono[2516]
        + mono[2517]
        + mono[2518];
    poly[8481] = poly[18] * poly[604] - poly[8475];
    poly[8482] = poly[19] * poly[604] - poly[8476];
    poly[8483] = poly[20] * poly[604] - poly[8477];
    poly[8484] = poly[21] * poly[604] - poly[8478];
    poly[8485] = poly[22] * poly[604] - poly[8479];
    poly[8486] = poly[23] * poly[604] - poly[8480];
    poly[8487] = poly[24] * poly[604];
    poly[8488] = mono[2519] + mono[2520] + mono[2521] + mono[2522];
    poly[8489] = mono[2523] + mono[2524] + mono[2525] + mono[2526];
    poly[8490] = mono[2527] + mono[2528] + mono[2529] + mono[2530];
    poly[8491] = poly[1] * poly[605];
    poly[8492] = poly[2] * poly[605];
    poly[8493] = poly[3] * poly[605];
    poly[8494] = poly[4] * poly[605];
    poly[8495] = poly[5] * poly[605];
    poly[8496] = poly[6] * poly[605];
    poly[8497] = mono[2531]
        + mono[2532]
        + mono[2533]
        + mono[2534]
        + mono[2535]
        + mono[2536]
        + mono[2537]
        + mono[2538];
    poly[8498] = mono[2539]
        + mono[2540]
        + mono[2541]
        + mono[2542]
        + mono[2543]
        + mono[2544]
        + mono[2545]
        + mono[2546];
    poly[8499] = mono[2547]
        + mono[2548]
        + mono[2549]
        + mono[2550]
        + mono[2551]
        + mono[2552]
        + mono[2553]
        + mono[2554];
}

fn f_polynomials170(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8500] = mono[2555]
        + mono[2556]
        + mono[2557]
        + mono[2558]
        + mono[2559]
        + mono[2560]
        + mono[2561]
        + mono[2562];
    poly[8501] = poly[7] * poly[605] - poly[8497];
    poly[8502] = poly[8] * poly[605] - poly[8498];
    poly[8503] = poly[9] * poly[605] - poly[8499];
    poly[8504] = poly[10] * poly[605] - poly[8500];
    poly[8505] = poly[11] * poly[605];
    poly[8506] = poly[12] * poly[605];
    poly[8507] = poly[13] * poly[605];
    poly[8508] = poly[14] * poly[605];
    poly[8509] = poly[15] * poly[605];
    poly[8510] = mono[2563]
        + mono[2564]
        + mono[2565]
        + mono[2566]
        + mono[2567]
        + mono[2568]
        + mono[2569]
        + mono[2570]
        + mono[2571]
        + mono[2572]
        + mono[2573]
        + mono[2574]
        + mono[2575]
        + mono[2576]
        + mono[2577]
        + mono[2578];
    poly[8511] = poly[16] * poly[605] - poly[8510];
    poly[8512] = poly[17] * poly[605];
    poly[8513] = mono[2579]
        + mono[2580]
        + mono[2581]
        + mono[2582]
        + mono[2583]
        + mono[2584]
        + mono[2585]
        + mono[2586];
    poly[8514] = mono[2587]
        + mono[2588]
        + mono[2589]
        + mono[2590]
        + mono[2591]
        + mono[2592]
        + mono[2593]
        + mono[2594];
    poly[8515] = mono[2595]
        + mono[2596]
        + mono[2597]
        + mono[2598]
        + mono[2599]
        + mono[2600]
        + mono[2601]
        + mono[2602];
    poly[8516] = mono[2603]
        + mono[2604]
        + mono[2605]
        + mono[2606]
        + mono[2607]
        + mono[2608]
        + mono[2609]
        + mono[2610];
    poly[8517] = mono[2611]
        + mono[2612]
        + mono[2613]
        + mono[2614]
        + mono[2615]
        + mono[2616]
        + mono[2617]
        + mono[2618];
    poly[8518] = mono[2619]
        + mono[2620]
        + mono[2621]
        + mono[2622]
        + mono[2623]
        + mono[2624]
        + mono[2625]
        + mono[2626];
    poly[8519] = mono[2627]
        + mono[2628]
        + mono[2629]
        + mono[2630]
        + mono[2631]
        + mono[2632]
        + mono[2633]
        + mono[2634]
        + mono[2635]
        + mono[2636]
        + mono[2637]
        + mono[2638]
        + mono[2639]
        + mono[2640]
        + mono[2641]
        + mono[2642];
    poly[8520] = poly[18] * poly[605] - poly[8513];
    poly[8521] = poly[19] * poly[605] - poly[8514];
    poly[8522] = poly[20] * poly[605] - poly[8515];
    poly[8523] = poly[21] * poly[605] - poly[8516];
    poly[8524] = mono[2643]
        + mono[2644]
        + mono[2645]
        + mono[2646]
        + mono[2647]
        + mono[2648]
        + mono[2649]
        + mono[2650];
    poly[8525] = poly[22] * poly[605] - poly[8524] - poly[8518] - poly[8517];
    poly[8526] = poly[23] * poly[605] - poly[8519];
    poly[8527] = poly[24] * poly[605];
    poly[8528] = mono[2651]
        + mono[2652]
        + mono[2653]
        + mono[2654]
        + mono[2655]
        + mono[2656]
        + mono[2657]
        + mono[2658];
    poly[8529] = mono[2659]
        + mono[2660]
        + mono[2661]
        + mono[2662]
        + mono[2663]
        + mono[2664]
        + mono[2665]
        + mono[2666];
    poly[8530] = mono[2667]
        + mono[2668]
        + mono[2669]
        + mono[2670]
        + mono[2671]
        + mono[2672]
        + mono[2673]
        + mono[2674];
    poly[8531] = mono[2675]
        + mono[2676]
        + mono[2677]
        + mono[2678]
        + mono[2679]
        + mono[2680]
        + mono[2681]
        + mono[2682];
    poly[8532] = mono[2683] + mono[2684] + mono[2685] + mono[2686];
    poly[8533] = poly[1] * poly[606];
    poly[8534] = poly[2] * poly[606];
    poly[8535] = poly[3] * poly[606];
    poly[8536] = poly[4] * poly[606];
    poly[8537] = poly[5] * poly[606];
    poly[8538] = poly[6] * poly[606];
    poly[8539] = poly[7] * poly[606];
    poly[8540] = poly[8] * poly[606];
    poly[8541] = poly[9] * poly[606];
    poly[8542] = poly[10] * poly[606];
    poly[8543] = poly[11] * poly[606];
    poly[8544] = mono[2687]
        + mono[2688]
        + mono[2689]
        + mono[2690]
        + mono[2691]
        + mono[2692]
        + mono[2693]
        + mono[2694];
    poly[8545] = mono[2695]
        + mono[2696]
        + mono[2697]
        + mono[2698]
        + mono[2699]
        + mono[2700]
        + mono[2701]
        + mono[2702];
    poly[8546] = mono[2703]
        + mono[2704]
        + mono[2705]
        + mono[2706]
        + mono[2707]
        + mono[2708]
        + mono[2709]
        + mono[2710];
    poly[8547] = mono[2711]
        + mono[2712]
        + mono[2713]
        + mono[2714]
        + mono[2715]
        + mono[2716]
        + mono[2717]
        + mono[2718];
    poly[8548] = mono[2719]
        + mono[2720]
        + mono[2721]
        + mono[2722]
        + mono[2723]
        + mono[2724]
        + mono[2725]
        + mono[2726]
        + mono[2727]
        + mono[2728]
        + mono[2729]
        + mono[2730]
        + mono[2731]
        + mono[2732]
        + mono[2733]
        + mono[2734];
    poly[8549] = poly[12] * poly[606] - poly[8544];
}

fn f_polynomials171(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8550] = poly[13] * poly[606] - poly[8545];
    poly[8551] = poly[14] * poly[606] - poly[8546];
    poly[8552] = poly[15] * poly[606] - poly[8547];
    poly[8553] = poly[16] * poly[606] - poly[8548];
    poly[8554] = poly[17] * poly[606];
    poly[8555] = mono[2735]
        + mono[2736]
        + mono[2737]
        + mono[2738]
        + mono[2739]
        + mono[2740]
        + mono[2741]
        + mono[2742];
    poly[8556] = mono[2743]
        + mono[2744]
        + mono[2745]
        + mono[2746]
        + mono[2747]
        + mono[2748]
        + mono[2749]
        + mono[2750];
    poly[8557] = mono[2751]
        + mono[2752]
        + mono[2753]
        + mono[2754]
        + mono[2755]
        + mono[2756]
        + mono[2757]
        + mono[2758];
    poly[8558] = mono[2759]
        + mono[2760]
        + mono[2761]
        + mono[2762]
        + mono[2763]
        + mono[2764]
        + mono[2765]
        + mono[2766];
    poly[8559] = mono[2767]
        + mono[2768]
        + mono[2769]
        + mono[2770]
        + mono[2771]
        + mono[2772]
        + mono[2773]
        + mono[2774]
        + mono[2775]
        + mono[2776]
        + mono[2777]
        + mono[2778]
        + mono[2779]
        + mono[2780]
        + mono[2781]
        + mono[2782];
    poly[8560] = mono[2783]
        + mono[2784]
        + mono[2785]
        + mono[2786]
        + mono[2787]
        + mono[2788]
        + mono[2789]
        + mono[2790];
    poly[8561] = mono[2791]
        + mono[2792]
        + mono[2793]
        + mono[2794]
        + mono[2795]
        + mono[2796]
        + mono[2797]
        + mono[2798];
    poly[8562] = poly[18] * poly[606] - poly[8555];
    poly[8563] = poly[19] * poly[606] - poly[8556];
    poly[8564] = poly[20] * poly[606] - poly[8557];
    poly[8565] = poly[21] * poly[606] - poly[8558];
    poly[8566] = poly[22] * poly[606] - poly[8559];
    poly[8567] = mono[2799]
        + mono[2800]
        + mono[2801]
        + mono[2802]
        + mono[2803]
        + mono[2804]
        + mono[2805]
        + mono[2806];
    poly[8568] = poly[23] * poly[606] - poly[8567] - poly[8561] - poly[8560];
    poly[8569] = poly[24] * poly[606];
    poly[8570] = mono[2807]
        + mono[2808]
        + mono[2809]
        + mono[2810]
        + mono[2811]
        + mono[2812]
        + mono[2813]
        + mono[2814];
    poly[8571] = mono[2815]
        + mono[2816]
        + mono[2817]
        + mono[2818]
        + mono[2819]
        + mono[2820]
        + mono[2821]
        + mono[2822];
    poly[8572] = mono[2823]
        + mono[2824]
        + mono[2825]
        + mono[2826]
        + mono[2827]
        + mono[2828]
        + mono[2829]
        + mono[2830];
    poly[8573] = mono[2831]
        + mono[2832]
        + mono[2833]
        + mono[2834]
        + mono[2835]
        + mono[2836]
        + mono[2837]
        + mono[2838];
    poly[8574] = mono[2839]
        + mono[2840]
        + mono[2841]
        + mono[2842]
        + mono[2843]
        + mono[2844]
        + mono[2845]
        + mono[2846]
        + mono[2847]
        + mono[2848]
        + mono[2849]
        + mono[2850]
        + mono[2851]
        + mono[2852]
        + mono[2853]
        + mono[2854];
    poly[8575] = mono[2855] + mono[2856] + mono[2857] + mono[2858];
    poly[8576] = poly[1] * poly[607];
    poly[8577] = poly[2] * poly[607];
    poly[8578] = poly[3] * poly[607];
    poly[8579] = poly[4] * poly[607];
    poly[8580] = poly[5] * poly[607];
    poly[8581] = poly[6] * poly[607];
    poly[8582] = poly[7] * poly[607];
    poly[8583] = poly[8] * poly[607];
    poly[8584] = poly[9] * poly[607];
    poly[8585] = poly[10] * poly[607];
    poly[8586] = poly[11] * poly[607];
    poly[8587] = poly[12] * poly[607];
    poly[8588] = poly[13] * poly[607];
    poly[8589] = poly[14] * poly[607];
    poly[8590] = poly[15] * poly[607];
    poly[8591] = poly[16] * poly[607];
    poly[8592] = poly[17] * poly[607];
    poly[8593] = poly[18] * poly[607];
    poly[8594] = poly[19] * poly[607];
    poly[8595] = poly[20] * poly[607];
    poly[8596] = poly[21] * poly[607];
    poly[8597] = poly[22] * poly[607];
    poly[8598] = poly[23] * poly[607];
    poly[8599] = poly[24] * poly[607];
}

fn f_polynomials172(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8600] = poly[1] * poly[608];
    poly[8601] = poly[2] * poly[608];
    poly[8602] = poly[3] * poly[608];
    poly[8603] = poly[4] * poly[608];
    poly[8604] = poly[5] * poly[608];
    poly[8605] = poly[6] * poly[608];
    poly[8606] = poly[7] * poly[608];
    poly[8607] = poly[8] * poly[608];
    poly[8608] = poly[9] * poly[608];
    poly[8609] = poly[10] * poly[608];
    poly[8610] = poly[11] * poly[608];
    poly[8611] = poly[12] * poly[608];
    poly[8612] = poly[13] * poly[608];
    poly[8613] = poly[14] * poly[608];
    poly[8614] = poly[15] * poly[608];
    poly[8615] = poly[16] * poly[608];
    poly[8616] = poly[17] * poly[608];
    poly[8617] = mono[2859] + mono[2860];
    poly[8618] = mono[2861] + mono[2862];
    poly[8619] = mono[2863] + mono[2864];
    poly[8620] = mono[2865] + mono[2866];
    poly[8621] = mono[2867] + mono[2868] + mono[2869] + mono[2870];
    poly[8622] = mono[2871] + mono[2872] + mono[2873] + mono[2874];
    poly[8623] = poly[18] * poly[608] - poly[8617];
    poly[8624] = poly[19] * poly[608] - poly[8618];
    poly[8625] = poly[20] * poly[608] - poly[8619];
    poly[8626] = poly[21] * poly[608] - poly[8620];
    poly[8627] = poly[22] * poly[608] - poly[8621];
    poly[8628] = poly[23] * poly[608] - poly[8622];
    poly[8629] = poly[24] * poly[608];
    poly[8630] = poly[1] * poly[609];
    poly[8631] = poly[2] * poly[609];
    poly[8632] = poly[3] * poly[609];
    poly[8633] = poly[4] * poly[609];
    poly[8634] = poly[5] * poly[609];
    poly[8635] = poly[6] * poly[609];
    poly[8636] = poly[7] * poly[609];
    poly[8637] = poly[8] * poly[609];
    poly[8638] = poly[9] * poly[609];
    poly[8639] = poly[10] * poly[609];
    poly[8640] = poly[11] * poly[609];
    poly[8641] = poly[12] * poly[609];
    poly[8642] = poly[13] * poly[609];
    poly[8643] = poly[14] * poly[609];
    poly[8644] = poly[15] * poly[609];
    poly[8645] = poly[16] * poly[609];
    poly[8646] = poly[17] * poly[609];
    poly[8647] = poly[25] * poly[588] - poly[8382];
    poly[8648] = poly[25] * poly[589] - poly[8383];
    poly[8649] = poly[25] * poly[590] - poly[8384];
}

fn f_polynomials173(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8650] = poly[25] * poly[591] - poly[8385];
    poly[8651] = poly[25] * poly[592] - poly[8386];
    poly[8652] = poly[25] * poly[593] - poly[8387];
    poly[8653] = poly[18] * poly[609] - poly[8647];
    poly[8654] = poly[19] * poly[609] - poly[8648];
    poly[8655] = poly[20] * poly[609] - poly[8649];
    poly[8656] = poly[21] * poly[609] - poly[8650];
    poly[8657] = poly[22] * poly[609] - poly[8651];
    poly[8658] = poly[23] * poly[609] - poly[8652];
    poly[8659] = poly[24] * poly[609];
    poly[8660] = poly[399] * poly[31];
    poly[8661] = poly[25] * poly[602] - poly[8425];
    poly[8662] = poly[25] * poly[603] - poly[8456];
    poly[8663] = poly[25] * poly[604] - poly[8488];
    poly[8664] = poly[25] * poly[605] - poly[8528];
    poly[8665] = poly[25] * poly[606] - poly[8570];
    poly[8666] = poly[25] * poly[607];
    poly[8667] = poly[25] * poly[608];
    poly[8668] = poly[1] * poly[610];
    poly[8669] = poly[2] * poly[610];
    poly[8670] = poly[3] * poly[610];
    poly[8671] = poly[4] * poly[610];
    poly[8672] = poly[5] * poly[610];
    poly[8673] = poly[6] * poly[610];
    poly[8674] = poly[7] * poly[610];
    poly[8675] = poly[8] * poly[610];
    poly[8676] = poly[9] * poly[610];
    poly[8677] = poly[10] * poly[610];
    poly[8678] = poly[11] * poly[610];
    poly[8679] = poly[12] * poly[610];
    poly[8680] = poly[13] * poly[610];
    poly[8681] = poly[14] * poly[610];
    poly[8682] = poly[15] * poly[610];
    poly[8683] = poly[16] * poly[610];
    poly[8684] = poly[17] * poly[610];
    poly[8685] = poly[26] * poly[588] - poly[8412];
    poly[8686] = poly[26] * poly[589] - poly[8413];
    poly[8687] = poly[26] * poly[590] - poly[8414];
    poly[8688] = poly[26] * poly[591] - poly[8415];
    poly[8689] = poly[26] * poly[592] - poly[8416];
    poly[8690] = poly[26] * poly[593] - poly[8417];
    poly[8691] = poly[18] * poly[610] - poly[8685];
    poly[8692] = poly[19] * poly[610] - poly[8686];
    poly[8693] = poly[20] * poly[610] - poly[8687];
    poly[8694] = poly[21] * poly[610] - poly[8688];
    poly[8695] = poly[22] * poly[610] - poly[8689];
    poly[8696] = poly[23] * poly[610] - poly[8690];
    poly[8697] = poly[24] * poly[610];
    poly[8698] = poly[26] * poly[601] - poly[8425];
    poly[8699] = poly[425] * poly[31];
}

fn f_polynomials174(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8700] = poly[26] * poly[603] - poly[8457];
    poly[8701] = poly[26] * poly[604] - poly[8489];
    poly[8702] = poly[26] * poly[605] - poly[8529];
    poly[8703] = poly[26] * poly[606] - poly[8571];
    poly[8704] = poly[26] * poly[607];
    poly[8705] = poly[26] * poly[608];
    poly[8706] = poly[25] * poly[610] - poly[8698];
    poly[8707] = poly[1] * poly[611];
    poly[8708] = poly[2] * poly[611];
    poly[8709] = poly[3] * poly[611];
    poly[8710] = poly[4] * poly[611];
    poly[8711] = poly[5] * poly[611];
    poly[8712] = poly[6] * poly[611];
    poly[8713] = poly[7] * poly[611];
    poly[8714] = poly[8] * poly[611];
    poly[8715] = poly[9] * poly[611];
    poly[8716] = poly[10] * poly[611];
    poly[8717] = poly[11] * poly[611];
    poly[8718] = poly[12] * poly[611];
    poly[8719] = poly[13] * poly[611];
    poly[8720] = poly[14] * poly[611];
    poly[8721] = poly[15] * poly[611];
    poly[8722] = poly[16] * poly[611];
    poly[8723] = poly[17] * poly[611];
    poly[8724] = poly[27] * poly[588] - poly[8443];
    poly[8725] = poly[27] * poly[589] - poly[8444];
    poly[8726] = poly[27] * poly[590] - poly[8445];
    poly[8727] = poly[27] * poly[591] - poly[8446];
    poly[8728] = poly[27] * poly[592] - poly[8447];
    poly[8729] = poly[27] * poly[593] - poly[8448];
    poly[8730] = poly[18] * poly[611] - poly[8724];
    poly[8731] = poly[19] * poly[611] - poly[8725];
    poly[8732] = poly[20] * poly[611] - poly[8726];
    poly[8733] = poly[21] * poly[611] - poly[8727];
    poly[8734] = poly[22] * poly[611] - poly[8728];
    poly[8735] = poly[23] * poly[611] - poly[8729];
    poly[8736] = poly[24] * poly[611];
    poly[8737] = poly[27] * poly[601] - poly[8456];
    poly[8738] = poly[27] * poly[602] - poly[8457];
    poly[8739] = poly[453] * poly[31];
    poly[8740] = poly[27] * poly[604] - poly[8490];
    poly[8741] = poly[27] * poly[605] - poly[8530];
    poly[8742] = poly[27] * poly[606] - poly[8572];
    poly[8743] = poly[27] * poly[607];
    poly[8744] = poly[27] * poly[608];
    poly[8745] = poly[25] * poly[611] - poly[8737];
    poly[8746] = poly[26] * poly[611] - poly[8738];
    poly[8747] = poly[1] * poly[612];
    poly[8748] = poly[2] * poly[612];
    poly[8749] = poly[3] * poly[612];
}

fn f_polynomials175(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8750] = poly[4] * poly[612];
    poly[8751] = poly[5] * poly[612];
    poly[8752] = poly[6] * poly[612];
    poly[8753] = poly[7] * poly[612];
    poly[8754] = poly[8] * poly[612];
    poly[8755] = poly[9] * poly[612];
    poly[8756] = poly[10] * poly[612];
    poly[8757] = poly[11] * poly[612];
    poly[8758] = poly[12] * poly[612];
    poly[8759] = poly[13] * poly[612];
    poly[8760] = poly[14] * poly[612];
    poly[8761] = poly[15] * poly[612];
    poly[8762] = poly[16] * poly[612];
    poly[8763] = poly[17] * poly[612];
    poly[8764] = poly[28] * poly[588] - poly[8475];
    poly[8765] = poly[28] * poly[589] - poly[8476];
    poly[8766] = poly[28] * poly[590] - poly[8477];
    poly[8767] = poly[28] * poly[591] - poly[8478];
    poly[8768] = poly[28] * poly[592] - poly[8479];
    poly[8769] = poly[28] * poly[593] - poly[8480];
    poly[8770] = poly[18] * poly[612] - poly[8764];
    poly[8771] = poly[19] * poly[612] - poly[8765];
    poly[8772] = poly[20] * poly[612] - poly[8766];
    poly[8773] = poly[21] * poly[612] - poly[8767];
    poly[8774] = poly[22] * poly[612] - poly[8768];
    poly[8775] = poly[23] * poly[612] - poly[8769];
    poly[8776] = poly[24] * poly[612];
    poly[8777] = poly[28] * poly[601] - poly[8488];
    poly[8778] = poly[28] * poly[602] - poly[8489];
    poly[8779] = poly[28] * poly[603] - poly[8490];
    poly[8780] = poly[483] * poly[31];
    poly[8781] = poly[28] * poly[605] - poly[8531];
    poly[8782] = poly[28] * poly[606] - poly[8573];
    poly[8783] = poly[28] * poly[607];
    poly[8784] = poly[28] * poly[608];
    poly[8785] = poly[25] * poly[612] - poly[8777];
    poly[8786] = poly[26] * poly[612] - poly[8778];
    poly[8787] = poly[27] * poly[612] - poly[8779];
    poly[8788] = poly[1] * poly[613];
    poly[8789] = poly[2] * poly[613];
    poly[8790] = poly[3] * poly[613];
    poly[8791] = poly[4] * poly[613];
    poly[8792] = poly[5] * poly[613];
    poly[8793] = poly[6] * poly[613];
    poly[8794] = poly[31] * poly[493] - poly[8497];
    poly[8795] = poly[31] * poly[494] - poly[8498];
    poly[8796] = poly[31] * poly[495] - poly[8499];
    poly[8797] = poly[31] * poly[496] - poly[8500];
    poly[8798] = poly[7] * poly[613] - poly[8794];
    poly[8799] = poly[8] * poly[613] - poly[8795];
}

fn f_polynomials176(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8800] = poly[9] * poly[613] - poly[8796];
    poly[8801] = poly[10] * poly[613] - poly[8797];
    poly[8802] = poly[11] * poly[613];
    poly[8803] = poly[12] * poly[613];
    poly[8804] = poly[13] * poly[613];
    poly[8805] = poly[14] * poly[613];
    poly[8806] = poly[15] * poly[613];
    poly[8807] = poly[31] * poly[506] - poly[8510];
    poly[8808] = poly[16] * poly[613] - poly[8807];
    poly[8809] = poly[17] * poly[613];
    poly[8810] = poly[29] * poly[588] - poly[8513];
    poly[8811] = poly[29] * poly[589] - poly[8514];
    poly[8812] = poly[29] * poly[590] - poly[8515];
    poly[8813] = poly[29] * poly[591] - poly[8516];
    poly[8814] = mono[2875]
        + mono[2876]
        + mono[2877]
        + mono[2878]
        + mono[2879]
        + mono[2880]
        + mono[2881]
        + mono[2882];
    poly[8815] = poly[29] * poly[592] - poly[8814] - poly[8518] - poly[8517];
    poly[8816] = poly[29] * poly[593] - poly[8519];
    poly[8817] = poly[18] * poly[613] - poly[8810];
    poly[8818] = poly[19] * poly[613] - poly[8811];
    poly[8819] = poly[20] * poly[613] - poly[8812];
    poly[8820] = poly[21] * poly[613] - poly[8813];
    poly[8821] = poly[31] * poly[513] - poly[8814] - poly[8524] - poly[8517];
    poly[8822] = poly[22] * poly[613] - poly[8821] - poly[8815] - poly[8814];
    poly[8823] = poly[23] * poly[613] - poly[8816];
    poly[8824] = poly[24] * poly[613];
    poly[8825] = poly[29] * poly[601] - poly[8528];
    poly[8826] = poly[29] * poly[602] - poly[8529];
    poly[8827] = poly[29] * poly[603] - poly[8530];
    poly[8828] = poly[29] * poly[604] - poly[8531];
    poly[8829] = poly[31] * poly[521];
    poly[8830] = poly[31] * poly[522];
    poly[8831] = poly[29] * poly[606] - poly[8574];
    poly[8832] = poly[29] * poly[607];
    poly[8833] = poly[29] * poly[608];
    poly[8834] = poly[25] * poly[613] - poly[8825];
    poly[8835] = poly[26] * poly[613] - poly[8826];
    poly[8836] = poly[27] * poly[613] - poly[8827];
    poly[8837] = poly[28] * poly[613] - poly[8828];
    poly[8838] = poly[31] * poly[527] - poly[8532];
    poly[8839] = poly[1] * poly[614];
    poly[8840] = poly[2] * poly[614];
    poly[8841] = poly[3] * poly[614];
    poly[8842] = poly[4] * poly[614];
    poly[8843] = poly[5] * poly[614];
    poly[8844] = poly[6] * poly[614];
    poly[8845] = poly[7] * poly[614];
    poly[8846] = poly[8] * poly[614];
    poly[8847] = poly[9] * poly[614];
    poly[8848] = poly[10] * poly[614];
    poly[8849] = poly[11] * poly[614];
}

fn f_polynomials177(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8850] = poly[31] * poly[539] - poly[8544];
    poly[8851] = poly[31] * poly[540] - poly[8545];
    poly[8852] = poly[31] * poly[541] - poly[8546];
    poly[8853] = poly[31] * poly[542] - poly[8547];
    poly[8854] = poly[31] * poly[543] - poly[8548];
    poly[8855] = poly[12] * poly[614] - poly[8850];
    poly[8856] = poly[13] * poly[614] - poly[8851];
    poly[8857] = poly[14] * poly[614] - poly[8852];
    poly[8858] = poly[15] * poly[614] - poly[8853];
    poly[8859] = poly[16] * poly[614] - poly[8854];
    poly[8860] = poly[17] * poly[614];
    poly[8861] = poly[30] * poly[588] - poly[8555];
    poly[8862] = poly[30] * poly[589] - poly[8556];
    poly[8863] = poly[30] * poly[590] - poly[8557];
    poly[8864] = poly[30] * poly[591] - poly[8558];
    poly[8865] = poly[30] * poly[592] - poly[8559];
    poly[8866] = mono[2883]
        + mono[2884]
        + mono[2885]
        + mono[2886]
        + mono[2887]
        + mono[2888]
        + mono[2889]
        + mono[2890];
    poly[8867] = poly[30] * poly[593] - poly[8866] - poly[8561] - poly[8560];
    poly[8868] = poly[18] * poly[614] - poly[8861];
    poly[8869] = poly[19] * poly[614] - poly[8862];
    poly[8870] = poly[20] * poly[614] - poly[8863];
    poly[8871] = poly[21] * poly[614] - poly[8864];
    poly[8872] = poly[22] * poly[614] - poly[8865];
    poly[8873] = poly[31] * poly[555] - poly[8866] - poly[8567] - poly[8560];
    poly[8874] = poly[23] * poly[614] - poly[8873] - poly[8867] - poly[8866];
    poly[8875] = poly[24] * poly[614];
    poly[8876] = poly[30] * poly[601] - poly[8570];
    poly[8877] = poly[30] * poly[602] - poly[8571];
    poly[8878] = poly[30] * poly[603] - poly[8572];
    poly[8879] = poly[30] * poly[604] - poly[8573];
    poly[8880] = poly[30] * poly[605] - poly[8574];
    poly[8881] = poly[31] * poly[563];
    poly[8882] = poly[31] * poly[564];
    poly[8883] = poly[30] * poly[607];
    poly[8884] = poly[30] * poly[608];
    poly[8885] = poly[25] * poly[614] - poly[8876];
    poly[8886] = poly[26] * poly[614] - poly[8877];
    poly[8887] = poly[27] * poly[614] - poly[8878];
    poly[8888] = poly[28] * poly[614] - poly[8879];
    poly[8889] = poly[29] * poly[614] - poly[8880];
    poly[8890] = poly[31] * poly[570] - poly[8575];
    poly[8891] = poly[1] * poly[615];
    poly[8892] = poly[2] * poly[615];
    poly[8893] = poly[3] * poly[615];
    poly[8894] = poly[4] * poly[615];
    poly[8895] = poly[5] * poly[615];
    poly[8896] = poly[6] * poly[615];
    poly[8897] = poly[7] * poly[615];
    poly[8898] = poly[8] * poly[615];
    poly[8899] = poly[9] * poly[615];
}

fn f_polynomials178(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8900] = poly[10] * poly[615];
    poly[8901] = poly[11] * poly[615];
    poly[8902] = poly[12] * poly[615];
    poly[8903] = poly[13] * poly[615];
    poly[8904] = poly[14] * poly[615];
    poly[8905] = poly[15] * poly[615];
    poly[8906] = poly[16] * poly[615];
    poly[8907] = poly[17] * poly[615];
    poly[8908] = poly[18] * poly[615];
    poly[8909] = poly[19] * poly[615];
    poly[8910] = poly[20] * poly[615];
    poly[8911] = poly[21] * poly[615];
    poly[8912] = poly[22] * poly[615];
    poly[8913] = poly[23] * poly[615];
    poly[8914] = poly[24] * poly[615];
    poly[8915] = mono[2891] + mono[2892];
    poly[8916] = mono[2893] + mono[2894];
    poly[8917] = mono[2895] + mono[2896];
    poly[8918] = mono[2897] + mono[2898];
    poly[8919] = mono[2899] + mono[2900] + mono[2901] + mono[2902];
    poly[8920] = mono[2903] + mono[2904] + mono[2905] + mono[2906];
    poly[8921] = mono[2907] + mono[2908] + mono[2909] + mono[2910];
    poly[8922] = poly[25] * poly[615] - poly[8915];
    poly[8923] = poly[26] * poly[615] - poly[8916];
    poly[8924] = poly[27] * poly[615] - poly[8917];
    poly[8925] = poly[28] * poly[615] - poly[8918];
    poly[8926] = poly[29] * poly[615] - poly[8919];
    poly[8927] = poly[30] * poly[615] - poly[8920];
    poly[8928] = poly[1] * poly[617];
    poly[8929] = poly[1] * poly[618];
    poly[8930] = poly[2] * poly[618];
    poly[8931] = poly[1] * poly[619];
    poly[8932] = poly[2] * poly[619];
    poly[8933] = poly[3] * poly[619];
    poly[8934] = poly[1] * poly[620];
    poly[8935] = poly[2] * poly[620];
    poly[8936] = poly[3] * poly[620];
    poly[8937] = poly[4] * poly[620];
    poly[8938] = poly[1] * poly[621];
    poly[8939] = poly[2] * poly[621];
    poly[8940] = poly[3] * poly[621];
    poly[8941] = poly[4] * poly[621];
    poly[8942] = poly[5] * poly[621];
    poly[8943] = poly[1] * poly[622];
    poly[8944] = poly[2] * poly[622];
    poly[8945] = poly[3] * poly[622];
    poly[8946] = poly[4] * poly[622];
    poly[8947] = poly[5] * poly[622];
    poly[8948] = poly[6] * poly[622];
    poly[8949] = poly[32] * poly[54];
}

fn f_polynomials179(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[8950] = poly[1] * poly[623];
    poly[8951] = poly[2] * poly[623];
    poly[8952] = poly[3] * poly[623];
    poly[8953] = poly[4] * poly[623];
    poly[8954] = poly[5] * poly[623];
    poly[8955] = poly[6] * poly[623];
    poly[8956] = poly[32] * poly[61];
    poly[8957] = poly[32] * poly[62];
    poly[8958] = poly[32] * poly[63];
    poly[8959] = poly[1] * poly[624];
    poly[8960] = poly[2] * poly[624];
    poly[8961] = poly[3] * poly[624];
    poly[8962] = poly[4] * poly[624];
    poly[8963] = poly[5] * poly[624];
    poly[8964] = poly[6] * poly[624];
    poly[8965] = poly[32] * poly[70];
    poly[8966] = poly[32] * poly[71];
    poly[8967] = poly[32] * poly[72];
    poly[8968] = poly[32] * poly[73];
    poly[8969] = poly[32] * poly[74];
    poly[8970] = poly[1] * poly[625];
    poly[8971] = poly[2] * poly[625];
    poly[8972] = poly[3] * poly[625];
    poly[8973] = poly[4] * poly[625];
    poly[8974] = poly[5] * poly[625];
    poly[8975] = poly[6] * poly[625];
    poly[8976] = poly[32] * poly[81];
    poly[8977] = poly[32] * poly[82];
    poly[8978] = poly[32] * poly[83];
    poly[8979] = poly[32] * poly[84];
    poly[8980] = poly[32] * poly[85];
    poly[8981] = poly[32] * poly[86];
    poly[8982] = poly[32] * poly[87];
    poly[8983] = poly[1] * poly[626];
    poly[8984] = poly[2] * poly[626];
    poly[8985] = poly[3] * poly[626];
    poly[8986] = poly[4] * poly[626];
    poly[8987] = poly[5] * poly[626];
    poly[8988] = poly[6] * poly[626];
    poly[8989] = poly[11] * poly[622];
    poly[8990] = poly[11] * poly[623];
    poly[8991] = poly[11] * poly[624];
    poly[8992] = poly[11] * poly[625];
    poly[8993] = poly[1] * poly[627];
    poly[8994] = poly[2] * poly[627];
    poly[8995] = poly[3] * poly[627];
    poly[8996] = poly[4] * poly[627];
    poly[8997] = poly[5] * poly[627];
    poly[8998] = poly[6] * poly[627];
    poly[8999] = poly[32] * poly[104];
}

fn f_polynomials180(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9000] = poly[32] * poly[105];
    poly[9001] = poly[32] * poly[106];
    poly[9002] = poly[32] * poly[107];
    poly[9003] = poly[11] * poly[627];
    poly[9004] = poly[32] * poly[109];
    poly[9005] = poly[1] * poly[628];
    poly[9006] = poly[2] * poly[628];
    poly[9007] = poly[3] * poly[628];
    poly[9008] = poly[4] * poly[628];
    poly[9009] = poly[5] * poly[628];
    poly[9010] = poly[6] * poly[628];
    poly[9011] = poly[32] * poly[116];
    poly[9012] = poly[32] * poly[117];
    poly[9013] = poly[32] * poly[118];
    poly[9014] = poly[32] * poly[119];
    poly[9015] = poly[11] * poly[628];
    poly[9016] = poly[32] * poly[121];
    poly[9017] = poly[32] * poly[122];
    poly[9018] = poly[32] * poly[123];
    poly[9019] = poly[1] * poly[629];
    poly[9020] = poly[2] * poly[629];
    poly[9021] = poly[3] * poly[629];
    poly[9022] = poly[4] * poly[629];
    poly[9023] = poly[5] * poly[629];
    poly[9024] = poly[6] * poly[629];
    poly[9025] = poly[32] * poly[130];
    poly[9026] = poly[32] * poly[131];
    poly[9027] = poly[32] * poly[132];
    poly[9028] = poly[32] * poly[133];
    poly[9029] = poly[11] * poly[629];
    poly[9030] = poly[32] * poly[135];
    poly[9031] = poly[32] * poly[136];
    poly[9032] = poly[32] * poly[137];
    poly[9033] = poly[32] * poly[138];
    poly[9034] = poly[32] * poly[139];
    poly[9035] = poly[1] * poly[630];
    poly[9036] = poly[2] * poly[630];
    poly[9037] = poly[3] * poly[630];
    poly[9038] = poly[4] * poly[630];
    poly[9039] = poly[5] * poly[630];
    poly[9040] = poly[6] * poly[630];
    poly[9041] = poly[32] * poly[146];
    poly[9042] = poly[32] * poly[147];
    poly[9043] = poly[32] * poly[148];
    poly[9044] = poly[32] * poly[149];
    poly[9045] = poly[11] * poly[630];
    poly[9046] = poly[32] * poly[151];
    poly[9047] = poly[32] * poly[152];
    poly[9048] = poly[32] * poly[153];
    poly[9049] = poly[32] * poly[154];
}

fn f_polynomials181(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9050] = poly[32] * poly[155];
    poly[9051] = poly[32] * poly[156];
    poly[9052] = poly[32] * poly[157];
    poly[9053] = poly[1] * poly[631];
    poly[9054] = poly[2] * poly[631];
    poly[9055] = poly[3] * poly[631];
    poly[9056] = poly[4] * poly[631];
    poly[9057] = poly[5] * poly[631];
    poly[9058] = poly[6] * poly[631];
    poly[9059] = poly[32] * poly[164];
    poly[9060] = poly[32] * poly[165];
    poly[9061] = poly[32] * poly[166];
    poly[9062] = poly[32] * poly[167];
    poly[9063] = poly[32] * poly[168];
    poly[9064] = poly[32] * poly[169];
    poly[9065] = poly[32] * poly[170];
    poly[9066] = poly[32] * poly[171];
    poly[9067] = poly[11] * poly[631];
    poly[9068] = poly[32] * poly[173];
    poly[9069] = poly[32] * poly[174];
    poly[9070] = poly[32] * poly[175];
    poly[9071] = poly[32] * poly[176];
    poly[9072] = poly[32] * poly[177];
    poly[9073] = poly[32] * poly[178];
    poly[9074] = poly[32] * poly[179];
    poly[9075] = poly[32] * poly[180];
    poly[9076] = poly[32] * poly[181];
    poly[9077] = poly[32] * poly[182];
    poly[9078] = poly[32] * poly[183];
    poly[9079] = poly[1] * poly[632];
    poly[9080] = poly[2] * poly[632];
    poly[9081] = poly[3] * poly[632];
    poly[9082] = poly[4] * poly[632];
    poly[9083] = poly[5] * poly[632];
    poly[9084] = poly[6] * poly[632];
    poly[9085] = poly[17] * poly[622];
    poly[9086] = poly[17] * poly[623];
    poly[9087] = poly[17] * poly[624];
    poly[9088] = poly[17] * poly[625];
    poly[9089] = poly[11] * poly[632];
    poly[9090] = poly[17] * poly[627];
    poly[9091] = poly[17] * poly[628];
    poly[9092] = poly[17] * poly[629];
    poly[9093] = poly[17] * poly[630];
    poly[9094] = poly[17] * poly[631];
    poly[9095] = poly[1] * poly[633];
    poly[9096] = poly[2] * poly[633];
    poly[9097] = poly[3] * poly[633];
    poly[9098] = poly[4] * poly[633];
    poly[9099] = poly[5] * poly[633];
}

fn f_polynomials182(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9100] = poly[6] * poly[633];
    poly[9101] = poly[32] * poly[206];
    poly[9102] = poly[32] * poly[207];
    poly[9103] = poly[32] * poly[208];
    poly[9104] = poly[32] * poly[209];
    poly[9105] = poly[11] * poly[633];
    poly[9106] = poly[32] * poly[211];
    poly[9107] = poly[32] * poly[212];
    poly[9108] = poly[32] * poly[213];
    poly[9109] = poly[32] * poly[214];
    poly[9110] = poly[32] * poly[215];
    poly[9111] = poly[17] * poly[633];
    poly[9112] = poly[32] * poly[217];
    poly[9113] = poly[1] * poly[634];
    poly[9114] = poly[2] * poly[634];
    poly[9115] = poly[3] * poly[634];
    poly[9116] = poly[4] * poly[634];
    poly[9117] = poly[5] * poly[634];
    poly[9118] = poly[6] * poly[634];
    poly[9119] = poly[32] * poly[224];
    poly[9120] = poly[32] * poly[225];
    poly[9121] = poly[32] * poly[226];
    poly[9122] = poly[32] * poly[227];
    poly[9123] = poly[11] * poly[634];
    poly[9124] = poly[32] * poly[229];
    poly[9125] = poly[32] * poly[230];
    poly[9126] = poly[32] * poly[231];
    poly[9127] = poly[32] * poly[232];
    poly[9128] = poly[32] * poly[233];
    poly[9129] = poly[17] * poly[634];
    poly[9130] = poly[32] * poly[235];
    poly[9131] = poly[32] * poly[236];
    poly[9132] = poly[32] * poly[237];
    poly[9133] = poly[1] * poly[635];
    poly[9134] = poly[2] * poly[635];
    poly[9135] = poly[3] * poly[635];
    poly[9136] = poly[4] * poly[635];
    poly[9137] = poly[5] * poly[635];
    poly[9138] = poly[6] * poly[635];
    poly[9139] = poly[32] * poly[244];
    poly[9140] = poly[32] * poly[245];
    poly[9141] = poly[32] * poly[246];
    poly[9142] = poly[32] * poly[247];
    poly[9143] = poly[11] * poly[635];
    poly[9144] = poly[32] * poly[249];
    poly[9145] = poly[32] * poly[250];
    poly[9146] = poly[32] * poly[251];
    poly[9147] = poly[32] * poly[252];
    poly[9148] = poly[32] * poly[253];
    poly[9149] = poly[17] * poly[635];
}

fn f_polynomials183(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9150] = poly[32] * poly[255];
    poly[9151] = poly[32] * poly[256];
    poly[9152] = poly[32] * poly[257];
    poly[9153] = poly[32] * poly[258];
    poly[9154] = poly[32] * poly[259];
    poly[9155] = poly[1] * poly[636];
    poly[9156] = poly[2] * poly[636];
    poly[9157] = poly[3] * poly[636];
    poly[9158] = poly[4] * poly[636];
    poly[9159] = poly[5] * poly[636];
    poly[9160] = poly[6] * poly[636];
    poly[9161] = poly[32] * poly[266];
    poly[9162] = poly[32] * poly[267];
    poly[9163] = poly[32] * poly[268];
    poly[9164] = poly[32] * poly[269];
    poly[9165] = poly[11] * poly[636];
    poly[9166] = poly[32] * poly[271];
    poly[9167] = poly[32] * poly[272];
    poly[9168] = poly[32] * poly[273];
    poly[9169] = poly[32] * poly[274];
    poly[9170] = poly[32] * poly[275];
    poly[9171] = poly[17] * poly[636];
    poly[9172] = poly[32] * poly[277];
    poly[9173] = poly[32] * poly[278];
    poly[9174] = poly[32] * poly[279];
    poly[9175] = poly[32] * poly[280];
    poly[9176] = poly[32] * poly[281];
    poly[9177] = poly[32] * poly[282];
    poly[9178] = poly[32] * poly[283];
    poly[9179] = poly[1] * poly[637];
    poly[9180] = poly[2] * poly[637];
    poly[9181] = poly[3] * poly[637];
    poly[9182] = poly[4] * poly[637];
    poly[9183] = poly[5] * poly[637];
    poly[9184] = poly[6] * poly[637];
    poly[9185] = poly[32] * poly[290];
    poly[9186] = poly[32] * poly[291];
    poly[9187] = poly[32] * poly[292];
    poly[9188] = poly[32] * poly[293];
    poly[9189] = poly[32] * poly[294];
    poly[9190] = poly[32] * poly[295];
    poly[9191] = poly[32] * poly[296];
    poly[9192] = poly[32] * poly[297];
    poly[9193] = poly[11] * poly[637];
    poly[9194] = poly[32] * poly[299];
    poly[9195] = poly[32] * poly[300];
    poly[9196] = poly[32] * poly[301];
    poly[9197] = poly[32] * poly[302];
    poly[9198] = poly[32] * poly[303];
    poly[9199] = poly[32] * poly[304];
}

fn f_polynomials184(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9200] = poly[17] * poly[637];
    poly[9201] = poly[32] * poly[306];
    poly[9202] = poly[32] * poly[307];
    poly[9203] = poly[32] * poly[308];
    poly[9204] = poly[32] * poly[309];
    poly[9205] = poly[32] * poly[310];
    poly[9206] = poly[32] * poly[311];
    poly[9207] = poly[32] * poly[312];
    poly[9208] = poly[32] * poly[313];
    poly[9209] = poly[32] * poly[314];
    poly[9210] = poly[32] * poly[315];
    poly[9211] = poly[32] * poly[316];
    poly[9212] = poly[1] * poly[638];
    poly[9213] = poly[2] * poly[638];
    poly[9214] = poly[3] * poly[638];
    poly[9215] = poly[4] * poly[638];
    poly[9216] = poly[5] * poly[638];
    poly[9217] = poly[6] * poly[638];
    poly[9218] = poly[32] * poly[323];
    poly[9219] = poly[32] * poly[324];
    poly[9220] = poly[32] * poly[325];
    poly[9221] = poly[32] * poly[326];
    poly[9222] = poly[11] * poly[638];
    poly[9223] = poly[32] * poly[328];
    poly[9224] = poly[32] * poly[329];
    poly[9225] = poly[32] * poly[330];
    poly[9226] = poly[32] * poly[331];
    poly[9227] = poly[32] * poly[332];
    poly[9228] = poly[32] * poly[333];
    poly[9229] = poly[32] * poly[334];
    poly[9230] = poly[32] * poly[335];
    poly[9231] = poly[32] * poly[336];
    poly[9232] = poly[32] * poly[337];
    poly[9233] = poly[17] * poly[638];
    poly[9234] = poly[32] * poly[339];
    poly[9235] = poly[32] * poly[340];
    poly[9236] = poly[32] * poly[341];
    poly[9237] = poly[32] * poly[342];
    poly[9238] = poly[32] * poly[343];
    poly[9239] = poly[32] * poly[344];
    poly[9240] = poly[32] * poly[345];
    poly[9241] = poly[32] * poly[346];
    poly[9242] = poly[32] * poly[347];
    poly[9243] = poly[32] * poly[348];
    poly[9244] = poly[32] * poly[349];
    poly[9245] = poly[32] * poly[350];
    poly[9246] = poly[32] * poly[351];
    poly[9247] = poly[1] * poly[639];
    poly[9248] = poly[2] * poly[639];
    poly[9249] = poly[3] * poly[639];
}

fn f_polynomials185(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9250] = poly[4] * poly[639];
    poly[9251] = poly[5] * poly[639];
    poly[9252] = poly[6] * poly[639];
    poly[9253] = poly[24] * poly[622];
    poly[9254] = poly[24] * poly[623];
    poly[9255] = poly[24] * poly[624];
    poly[9256] = poly[24] * poly[625];
    poly[9257] = poly[11] * poly[639];
    poly[9258] = poly[24] * poly[627];
    poly[9259] = poly[24] * poly[628];
    poly[9260] = poly[24] * poly[629];
    poly[9261] = poly[24] * poly[630];
    poly[9262] = poly[24] * poly[631];
    poly[9263] = poly[17] * poly[639];
    poly[9264] = poly[24] * poly[633];
    poly[9265] = poly[24] * poly[634];
    poly[9266] = poly[24] * poly[635];
    poly[9267] = poly[24] * poly[636];
    poly[9268] = poly[24] * poly[637];
    poly[9269] = poly[24] * poly[638];
    poly[9270] = poly[1] * poly[640];
    poly[9271] = poly[2] * poly[640];
    poly[9272] = poly[3] * poly[640];
    poly[9273] = poly[4] * poly[640];
    poly[9274] = poly[5] * poly[640];
    poly[9275] = poly[6] * poly[640];
    poly[9276] = poly[32] * poly[381];
    poly[9277] = poly[32] * poly[382];
    poly[9278] = poly[32] * poly[383];
    poly[9279] = poly[32] * poly[384];
    poly[9280] = poly[11] * poly[640];
    poly[9281] = poly[32] * poly[386];
    poly[9282] = poly[32] * poly[387];
    poly[9283] = poly[32] * poly[388];
    poly[9284] = poly[32] * poly[389];
    poly[9285] = poly[32] * poly[390];
    poly[9286] = poly[17] * poly[640];
    poly[9287] = poly[32] * poly[392];
    poly[9288] = poly[32] * poly[393];
    poly[9289] = poly[32] * poly[394];
    poly[9290] = poly[32] * poly[395];
    poly[9291] = poly[32] * poly[396];
    poly[9292] = poly[32] * poly[397];
    poly[9293] = poly[24] * poly[640];
    poly[9294] = poly[32] * poly[399];
    poly[9295] = poly[1] * poly[641];
    poly[9296] = poly[2] * poly[641];
    poly[9297] = poly[3] * poly[641];
    poly[9298] = poly[4] * poly[641];
    poly[9299] = poly[5] * poly[641];
}

fn f_polynomials186(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9300] = poly[6] * poly[641];
    poly[9301] = poly[32] * poly[406];
    poly[9302] = poly[32] * poly[407];
    poly[9303] = poly[32] * poly[408];
    poly[9304] = poly[32] * poly[409];
    poly[9305] = poly[11] * poly[641];
    poly[9306] = poly[32] * poly[411];
    poly[9307] = poly[32] * poly[412];
    poly[9308] = poly[32] * poly[413];
    poly[9309] = poly[32] * poly[414];
    poly[9310] = poly[32] * poly[415];
    poly[9311] = poly[17] * poly[641];
    poly[9312] = poly[32] * poly[417];
    poly[9313] = poly[32] * poly[418];
    poly[9314] = poly[32] * poly[419];
    poly[9315] = poly[32] * poly[420];
    poly[9316] = poly[32] * poly[421];
    poly[9317] = poly[32] * poly[422];
    poly[9318] = poly[24] * poly[641];
    poly[9319] = poly[32] * poly[424];
    poly[9320] = poly[32] * poly[425];
    poly[9321] = poly[32] * poly[426];
    poly[9322] = poly[1] * poly[642];
    poly[9323] = poly[2] * poly[642];
    poly[9324] = poly[3] * poly[642];
    poly[9325] = poly[4] * poly[642];
    poly[9326] = poly[5] * poly[642];
    poly[9327] = poly[6] * poly[642];
    poly[9328] = poly[32] * poly[433];
    poly[9329] = poly[32] * poly[434];
    poly[9330] = poly[32] * poly[435];
    poly[9331] = poly[32] * poly[436];
    poly[9332] = poly[11] * poly[642];
    poly[9333] = poly[32] * poly[438];
    poly[9334] = poly[32] * poly[439];
    poly[9335] = poly[32] * poly[440];
    poly[9336] = poly[32] * poly[441];
    poly[9337] = poly[32] * poly[442];
    poly[9338] = poly[17] * poly[642];
    poly[9339] = poly[32] * poly[444];
    poly[9340] = poly[32] * poly[445];
    poly[9341] = poly[32] * poly[446];
    poly[9342] = poly[32] * poly[447];
    poly[9343] = poly[32] * poly[448];
    poly[9344] = poly[32] * poly[449];
    poly[9345] = poly[24] * poly[642];
    poly[9346] = poly[32] * poly[451];
    poly[9347] = poly[32] * poly[452];
    poly[9348] = poly[32] * poly[453];
    poly[9349] = poly[32] * poly[454];
}

fn f_polynomials187(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9350] = poly[32] * poly[455];
    poly[9351] = poly[1] * poly[643];
    poly[9352] = poly[2] * poly[643];
    poly[9353] = poly[3] * poly[643];
    poly[9354] = poly[4] * poly[643];
    poly[9355] = poly[5] * poly[643];
    poly[9356] = poly[6] * poly[643];
    poly[9357] = poly[32] * poly[462];
    poly[9358] = poly[32] * poly[463];
    poly[9359] = poly[32] * poly[464];
    poly[9360] = poly[32] * poly[465];
    poly[9361] = poly[11] * poly[643];
    poly[9362] = poly[32] * poly[467];
    poly[9363] = poly[32] * poly[468];
    poly[9364] = poly[32] * poly[469];
    poly[9365] = poly[32] * poly[470];
    poly[9366] = poly[32] * poly[471];
    poly[9367] = poly[17] * poly[643];
    poly[9368] = poly[32] * poly[473];
    poly[9369] = poly[32] * poly[474];
    poly[9370] = poly[32] * poly[475];
    poly[9371] = poly[32] * poly[476];
    poly[9372] = poly[32] * poly[477];
    poly[9373] = poly[32] * poly[478];
    poly[9374] = poly[24] * poly[643];
    poly[9375] = poly[32] * poly[480];
    poly[9376] = poly[32] * poly[481];
    poly[9377] = poly[32] * poly[482];
    poly[9378] = poly[32] * poly[483];
    poly[9379] = poly[32] * poly[484];
    poly[9380] = poly[32] * poly[485];
    poly[9381] = poly[32] * poly[486];
    poly[9382] = poly[1] * poly[644];
    poly[9383] = poly[2] * poly[644];
    poly[9384] = poly[3] * poly[644];
    poly[9385] = poly[4] * poly[644];
    poly[9386] = poly[5] * poly[644];
    poly[9387] = poly[6] * poly[644];
    poly[9388] = poly[32] * poly[493];
    poly[9389] = poly[32] * poly[494];
    poly[9390] = poly[32] * poly[495];
    poly[9391] = poly[32] * poly[496];
    poly[9392] = poly[32] * poly[497];
    poly[9393] = poly[32] * poly[498];
    poly[9394] = poly[32] * poly[499];
    poly[9395] = poly[32] * poly[500];
    poly[9396] = poly[11] * poly[644];
    poly[9397] = poly[32] * poly[502];
    poly[9398] = poly[32] * poly[503];
    poly[9399] = poly[32] * poly[504];
}

fn f_polynomials188(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9400] = poly[32] * poly[505];
    poly[9401] = poly[32] * poly[506];
    poly[9402] = poly[32] * poly[507];
    poly[9403] = poly[17] * poly[644];
    poly[9404] = poly[32] * poly[509];
    poly[9405] = poly[32] * poly[510];
    poly[9406] = poly[32] * poly[511];
    poly[9407] = poly[32] * poly[512];
    poly[9408] = poly[32] * poly[513];
    poly[9409] = poly[32] * poly[514];
    poly[9410] = poly[32] * poly[515];
    poly[9411] = poly[24] * poly[644];
    poly[9412] = poly[32] * poly[517];
    poly[9413] = poly[32] * poly[518];
    poly[9414] = poly[32] * poly[519];
    poly[9415] = poly[32] * poly[520];
    poly[9416] = poly[32] * poly[521];
    poly[9417] = poly[32] * poly[522];
    poly[9418] = poly[32] * poly[523];
    poly[9419] = poly[32] * poly[524];
    poly[9420] = poly[32] * poly[525];
    poly[9421] = poly[32] * poly[526];
    poly[9422] = poly[32] * poly[527];
    poly[9423] = poly[1] * poly[645];
    poly[9424] = poly[2] * poly[645];
    poly[9425] = poly[3] * poly[645];
    poly[9426] = poly[4] * poly[645];
    poly[9427] = poly[5] * poly[645];
    poly[9428] = poly[6] * poly[645];
    poly[9429] = poly[32] * poly[534];
    poly[9430] = poly[32] * poly[535];
    poly[9431] = poly[32] * poly[536];
    poly[9432] = poly[32] * poly[537];
    poly[9433] = poly[11] * poly[645];
    poly[9434] = poly[32] * poly[539];
    poly[9435] = poly[32] * poly[540];
    poly[9436] = poly[32] * poly[541];
    poly[9437] = poly[32] * poly[542];
    poly[9438] = poly[32] * poly[543];
    poly[9439] = poly[32] * poly[544];
    poly[9440] = poly[32] * poly[545];
    poly[9441] = poly[32] * poly[546];
    poly[9442] = poly[32] * poly[547];
    poly[9443] = poly[32] * poly[548];
    poly[9444] = poly[17] * poly[645];
    poly[9445] = poly[32] * poly[550];
    poly[9446] = poly[32] * poly[551];
    poly[9447] = poly[32] * poly[552];
    poly[9448] = poly[32] * poly[553];
    poly[9449] = poly[32] * poly[554];
}

fn f_polynomials189(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9450] = poly[32] * poly[555];
    poly[9451] = poly[32] * poly[556];
    poly[9452] = poly[24] * poly[645];
    poly[9453] = poly[32] * poly[558];
    poly[9454] = poly[32] * poly[559];
    poly[9455] = poly[32] * poly[560];
    poly[9456] = poly[32] * poly[561];
    poly[9457] = poly[32] * poly[562];
    poly[9458] = poly[32] * poly[563];
    poly[9459] = poly[32] * poly[564];
    poly[9460] = poly[32] * poly[565];
    poly[9461] = poly[32] * poly[566];
    poly[9462] = poly[32] * poly[567];
    poly[9463] = poly[32] * poly[568];
    poly[9464] = poly[32] * poly[569];
    poly[9465] = poly[32] * poly[570];
    poly[9466] = poly[1] * poly[646];
    poly[9467] = poly[2] * poly[646];
    poly[9468] = poly[3] * poly[646];
    poly[9469] = poly[4] * poly[646];
    poly[9470] = poly[5] * poly[646];
    poly[9471] = poly[6] * poly[646];
    poly[9472] = poly[32] * poly[577];
    poly[9473] = poly[32] * poly[578];
    poly[9474] = poly[32] * poly[579];
    poly[9475] = poly[32] * poly[580];
    poly[9476] = poly[11] * poly[646];
    poly[9477] = poly[32] * poly[582];
    poly[9478] = poly[32] * poly[583];
    poly[9479] = poly[32] * poly[584];
    poly[9480] = poly[32] * poly[585];
    poly[9481] = poly[32] * poly[586];
    poly[9482] = poly[17] * poly[646];
    poly[9483] = poly[32] * poly[588];
    poly[9484] = poly[32] * poly[589];
    poly[9485] = poly[32] * poly[590];
    poly[9486] = poly[32] * poly[591];
    poly[9487] = poly[32] * poly[592];
    poly[9488] = poly[32] * poly[593];
    poly[9489] = poly[32] * poly[594];
    poly[9490] = poly[32] * poly[595];
    poly[9491] = poly[32] * poly[596];
    poly[9492] = poly[32] * poly[597];
    poly[9493] = poly[32] * poly[598];
    poly[9494] = poly[32] * poly[599];
    poly[9495] = poly[24] * poly[646];
    poly[9496] = poly[32] * poly[601];
    poly[9497] = poly[32] * poly[602];
    poly[9498] = poly[32] * poly[603];
    poly[9499] = poly[32] * poly[604];
}

fn f_polynomials190(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9500] = poly[32] * poly[605];
    poly[9501] = poly[32] * poly[606];
    poly[9502] = poly[32] * poly[607];
    poly[9503] = poly[32] * poly[608];
    poly[9504] = poly[32] * poly[609];
    poly[9505] = poly[32] * poly[610];
    poly[9506] = poly[32] * poly[611];
    poly[9507] = poly[32] * poly[612];
    poly[9508] = poly[32] * poly[613];
    poly[9509] = poly[32] * poly[614];
    poly[9510] = poly[32] * poly[615];
    poly[9511] = poly[1] * poly[33];
    poly[9512] = poly[1] * poly[648];
    poly[9513] = poly[1] * poly[34];
    poly[9514] = poly[2] * poly[35];
    poly[9515] = poly[1] * poly[649];
    poly[9516] = poly[2] * poly[649];
    poly[9517] = poly[1] * poly[36];
    poly[9518] = poly[2] * poly[37];
    poly[9519] = poly[3] * poly[38];
    poly[9520] = poly[1] * poly[650];
    poly[9521] = poly[2] * poly[650];
    poly[9522] = poly[3] * poly[650];
    poly[9523] = poly[1] * poly[39];
    poly[9524] = poly[2] * poly[40];
    poly[9525] = poly[3] * poly[41];
    poly[9526] = poly[4] * poly[42];
    poly[9527] = poly[1] * poly[651];
    poly[9528] = poly[2] * poly[651];
    poly[9529] = poly[3] * poly[651];
    poly[9530] = poly[4] * poly[651];
    poly[9531] = poly[1] * poly[43];
    poly[9532] = poly[2] * poly[44];
    poly[9533] = poly[3] * poly[45];
    poly[9534] = poly[4] * poly[46];
    poly[9535] = poly[5] * poly[47];
    poly[9536] = poly[1] * poly[652];
    poly[9537] = poly[2] * poly[652];
    poly[9538] = poly[3] * poly[652];
    poly[9539] = poly[4] * poly[652];
    poly[9540] = poly[5] * poly[652];
    poly[9541] = poly[1] * poly[48];
    poly[9542] = poly[2] * poly[49];
    poly[9543] = poly[3] * poly[50];
    poly[9544] = poly[4] * poly[51];
    poly[9545] = poly[5] * poly[52];
    poly[9546] = poly[6] * poly[53];
    poly[9547] = poly[1] * poly[653];
    poly[9548] = poly[2] * poly[653];
    poly[9549] = poly[3] * poly[653];
}

fn f_polynomials191(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9550] = poly[4] * poly[653];
    poly[9551] = poly[5] * poly[653];
    poly[9552] = poly[6] * poly[653];
    poly[9553] = poly[54] * poly[7];
    poly[9554] = poly[1] * poly[55];
    poly[9555] = poly[2] * poly[56];
    poly[9556] = poly[3] * poly[57];
    poly[9557] = poly[4] * poly[58];
    poly[9558] = poly[5] * poly[59];
    poly[9559] = poly[6] * poly[60];
    poly[9560] = poly[7] * poly[61] - poly[753];
    poly[9561] = poly[7] * poly[63] - poly[753];
    poly[9562] = poly[1] * poly[654];
    poly[9563] = poly[2] * poly[654];
    poly[9564] = poly[3] * poly[654];
    poly[9565] = poly[4] * poly[654];
    poly[9566] = poly[5] * poly[654];
    poly[9567] = poly[6] * poly[654];
    poly[9568] = poly[8] * poly[61] - poly[754];
    poly[9569] = poly[62] * poly[8];
    poly[9570] = poly[7] * poly[654] - poly[9568];
    poly[9571] = poly[1] * poly[64];
    poly[9572] = poly[2] * poly[65];
    poly[9573] = poly[3] * poly[66];
    poly[9574] = poly[4] * poly[67];
    poly[9575] = poly[5] * poly[68];
    poly[9576] = poly[6] * poly[69];
    poly[9577] = poly[7] * poly[70] - poly[795];
    poly[9578] = poly[8] * poly[71] - poly[805];
    poly[9579] = poly[7] * poly[73] - poly[795];
    poly[9580] = poly[8] * poly[74] - poly[805];
    poly[9581] = poly[1] * poly[655];
    poly[9582] = poly[2] * poly[655];
    poly[9583] = poly[3] * poly[655];
    poly[9584] = poly[4] * poly[655];
    poly[9585] = poly[5] * poly[655];
    poly[9586] = poly[6] * poly[655];
    poly[9587] = poly[9] * poly[70] - poly[797];
    poly[9588] = poly[9] * poly[71] - poly[806];
    poly[9589] = poly[72] * poly[9];
    poly[9590] = poly[7] * poly[655] - poly[9587];
    poly[9591] = poly[8] * poly[655] - poly[9588];
    poly[9592] = poly[1] * poly[75];
    poly[9593] = poly[2] * poly[76];
    poly[9594] = poly[3] * poly[77];
    poly[9595] = poly[4] * poly[78];
    poly[9596] = poly[5] * poly[79];
    poly[9597] = poly[6] * poly[80];
    poly[9598] = poly[7] * poly[81] - poly[856];
    poly[9599] = poly[8] * poly[82] - poly[867];
}

fn f_polynomials192(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9600] = poly[9] * poly[83] - poly[879];
    poly[9601] = poly[7] * poly[85] - poly[856];
    poly[9602] = poly[8] * poly[86] - poly[867];
    poly[9603] = poly[9] * poly[87] - poly[879];
    poly[9604] = poly[1] * poly[656];
    poly[9605] = poly[2] * poly[656];
    poly[9606] = poly[3] * poly[656];
    poly[9607] = poly[4] * poly[656];
    poly[9608] = poly[5] * poly[656];
    poly[9609] = poly[6] * poly[656];
    poly[9610] = poly[10] * poly[81] - poly[859];
    poly[9611] = poly[10] * poly[82] - poly[869];
    poly[9612] = poly[10] * poly[83] - poly[880];
    poly[9613] = poly[84] * poly[10];
    poly[9614] = poly[7] * poly[656] - poly[9610];
    poly[9615] = poly[8] * poly[656] - poly[9611];
    poly[9616] = poly[9] * poly[656] - poly[9612];
    poly[9617] = poly[1] * poly[88];
    poly[9618] = poly[2] * poly[89];
    poly[9619] = poly[3] * poly[90];
    poly[9620] = poly[4] * poly[91];
    poly[9621] = poly[5] * poly[92];
    poly[9622] = poly[6] * poly[93];
    poly[9623] = poly[11] * poly[653];
    poly[9624] = poly[11] * poly[654];
    poly[9625] = poly[11] * poly[655];
    poly[9626] = poly[11] * poly[656];
    poly[9627] = poly[1] * poly[657];
    poly[9628] = poly[2] * poly[657];
    poly[9629] = poly[3] * poly[657];
    poly[9630] = poly[4] * poly[657];
    poly[9631] = poly[5] * poly[657];
    poly[9632] = poly[6] * poly[657];
    poly[9633] = poly[11] * poly[94];
    poly[9634] = poly[11] * poly[95];
    poly[9635] = poly[11] * poly[96];
    poly[9636] = poly[11] * poly[97];
    poly[9637] = poly[1] * poly[98];
    poly[9638] = poly[2] * poly[99];
    poly[9639] = poly[3] * poly[100];
    poly[9640] = poly[4] * poly[101];
    poly[9641] = poly[5] * poly[102];
    poly[9642] = poly[6] * poly[103];
    poly[9643] = poly[12] * poly[653];
    poly[9644] = poly[12] * poly[654];
    poly[9645] = poly[12] * poly[655];
    poly[9646] = poly[12] * poly[656];
    poly[9647] = poly[11] * poly[108];
    poly[9648] = poly[1] * poly[658];
    poly[9649] = poly[2] * poly[658];
}

fn f_polynomials193(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9650] = poly[3] * poly[658];
    poly[9651] = poly[4] * poly[658];
    poly[9652] = poly[5] * poly[658];
    poly[9653] = poly[6] * poly[658];
    poly[9654] = poly[7] * poly[658];
    poly[9655] = poly[8] * poly[658];
    poly[9656] = poly[9] * poly[658];
    poly[9657] = poly[10] * poly[658];
    poly[9658] = poly[11] * poly[658];
    poly[9659] = poly[109] * poly[12];
    poly[9660] = poly[1] * poly[110];
    poly[9661] = poly[2] * poly[111];
    poly[9662] = poly[3] * poly[112];
    poly[9663] = poly[4] * poly[113];
    poly[9664] = poly[5] * poly[114];
    poly[9665] = poly[6] * poly[115];
    poly[9666] = poly[13] * poly[653];
    poly[9667] = poly[13] * poly[654];
    poly[9668] = poly[13] * poly[655];
    poly[9669] = poly[13] * poly[656];
    poly[9670] = poly[11] * poly[120];
    poly[9671] = poly[12] * poly[121] - poly[1112];
    poly[9672] = poly[12] * poly[123] - poly[1112];
    poly[9673] = poly[1] * poly[659];
    poly[9674] = poly[2] * poly[659];
    poly[9675] = poly[3] * poly[659];
    poly[9676] = poly[4] * poly[659];
    poly[9677] = poly[5] * poly[659];
    poly[9678] = poly[6] * poly[659];
    poly[9679] = poly[7] * poly[659];
    poly[9680] = poly[8] * poly[659];
    poly[9681] = poly[9] * poly[659];
    poly[9682] = poly[10] * poly[659];
    poly[9683] = poly[11] * poly[659];
    poly[9684] = poly[13] * poly[121] - poly[1113];
    poly[9685] = poly[122] * poly[13];
    poly[9686] = poly[12] * poly[659] - poly[9684];
    poly[9687] = poly[1] * poly[124];
    poly[9688] = poly[2] * poly[125];
    poly[9689] = poly[3] * poly[126];
    poly[9690] = poly[4] * poly[127];
    poly[9691] = poly[5] * poly[128];
    poly[9692] = poly[6] * poly[129];
    poly[9693] = poly[14] * poly[653];
    poly[9694] = poly[14] * poly[654];
    poly[9695] = poly[14] * poly[655];
    poly[9696] = poly[14] * poly[656];
    poly[9697] = poly[11] * poly[134];
    poly[9698] = poly[12] * poly[135] - poly[1224];
    poly[9699] = poly[13] * poly[136] - poly[1239];
}

fn f_polynomials194(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9700] = poly[12] * poly[138] - poly[1224];
    poly[9701] = poly[13] * poly[139] - poly[1239];
    poly[9702] = poly[1] * poly[660];
    poly[9703] = poly[2] * poly[660];
    poly[9704] = poly[3] * poly[660];
    poly[9705] = poly[4] * poly[660];
    poly[9706] = poly[5] * poly[660];
    poly[9707] = poly[6] * poly[660];
    poly[9708] = poly[7] * poly[660];
    poly[9709] = poly[8] * poly[660];
    poly[9710] = poly[9] * poly[660];
    poly[9711] = poly[10] * poly[660];
    poly[9712] = poly[11] * poly[660];
    poly[9713] = poly[14] * poly[135] - poly[1226];
    poly[9714] = poly[14] * poly[136] - poly[1240];
    poly[9715] = poly[137] * poly[14];
    poly[9716] = poly[12] * poly[660] - poly[9713];
    poly[9717] = poly[13] * poly[660] - poly[9714];
    poly[9718] = poly[1] * poly[140];
    poly[9719] = poly[2] * poly[141];
    poly[9720] = poly[3] * poly[142];
    poly[9721] = poly[4] * poly[143];
    poly[9722] = poly[5] * poly[144];
    poly[9723] = poly[6] * poly[145];
    poly[9724] = poly[15] * poly[653];
    poly[9725] = poly[15] * poly[654];
    poly[9726] = poly[15] * poly[655];
    poly[9727] = poly[15] * poly[656];
    poly[9728] = poly[11] * poly[150];
    poly[9729] = poly[12] * poly[151] - poly[1365];
    poly[9730] = poly[13] * poly[152] - poly[1381];
    poly[9731] = poly[14] * poly[153] - poly[1398];
    poly[9732] = poly[12] * poly[155] - poly[1365];
    poly[9733] = poly[13] * poly[156] - poly[1381];
    poly[9734] = poly[14] * poly[157] - poly[1398];
    poly[9735] = poly[1] * poly[661];
    poly[9736] = poly[2] * poly[661];
    poly[9737] = poly[3] * poly[661];
    poly[9738] = poly[4] * poly[661];
    poly[9739] = poly[5] * poly[661];
    poly[9740] = poly[6] * poly[661];
    poly[9741] = poly[7] * poly[661];
    poly[9742] = poly[8] * poly[661];
    poly[9743] = poly[9] * poly[661];
    poly[9744] = poly[10] * poly[661];
    poly[9745] = poly[11] * poly[661];
    poly[9746] = poly[15] * poly[151] - poly[1368];
    poly[9747] = poly[15] * poly[152] - poly[1383];
    poly[9748] = poly[15] * poly[153] - poly[1399];
    poly[9749] = poly[154] * poly[15];
}

fn f_polynomials195(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9750] = poly[12] * poly[661] - poly[9746];
    poly[9751] = poly[13] * poly[661] - poly[9747];
    poly[9752] = poly[14] * poly[661] - poly[9748];
    poly[9753] = poly[1] * poly[158];
    poly[9754] = poly[2] * poly[159];
    poly[9755] = poly[3] * poly[160];
    poly[9756] = poly[4] * poly[161];
    poly[9757] = poly[5] * poly[162];
    poly[9758] = poly[6] * poly[163];
    poly[9759] = poly[7] * poly[164] - poly[1453];
    poly[9760] = poly[8] * poly[165] - poly[1464];
    poly[9761] = poly[9] * poly[166] - poly[1476];
    poly[9762] = poly[10] * poly[167] - poly[1489];
    poly[9763] = poly[7] * poly[168] - poly[1453];
    poly[9764] = poly[8] * poly[169] - poly[1464];
    poly[9765] = poly[9] * poly[170] - poly[1476];
    poly[9766] = poly[10] * poly[171] - poly[1489];
    poly[9767] = poly[11] * poly[172];
    poly[9768] = poly[12] * poly[173] - poly[1614];
    poly[9769] = poly[13] * poly[174] - poly[1636];
    poly[9770] = poly[14] * poly[175] - poly[1659];
    poly[9771] = poly[15] * poly[176] - poly[1683];
    poly[9772] = poly[12] * poly[179] - poly[1614];
    poly[9773] = poly[13] * poly[180] - poly[1636];
    poly[9774] = poly[14] * poly[181] - poly[1659];
    poly[9775] = poly[15] * poly[182] - poly[1683];
    poly[9776] = poly[1] * poly[662];
    poly[9777] = poly[2] * poly[662];
    poly[9778] = poly[3] * poly[662];
    poly[9779] = poly[4] * poly[662];
    poly[9780] = poly[5] * poly[662];
    poly[9781] = poly[6] * poly[662];
    poly[9782] = poly[16] * poly[164] - poly[1695] - poly[1590] - poly[1579] - poly[1590];
    poly[9783] = poly[16] * poly[165] - poly[1696] - poly[1591] - poly[1580] - poly[1591];
    poly[9784] = poly[16] * poly[166] - poly[1697] - poly[1592] - poly[1581] - poly[1592];
    poly[9785] = poly[16] * poly[167] - poly[1698] - poly[1593] - poly[1582] - poly[1593];
    poly[9786] = poly[7] * poly[662] - poly[9782];
    poly[9787] = poly[8] * poly[662] - poly[9783];
    poly[9788] = poly[9] * poly[662] - poly[9784];
    poly[9789] = poly[10] * poly[662] - poly[9785];
    poly[9790] = poly[11] * poly[662];
    poly[9791] = poly[16] * poly[173] - poly[1700] - poly[1619] - poly[1618] - poly[1700];
    poly[9792] = poly[16] * poly[174] - poly[1701] - poly[1640] - poly[1639] - poly[1701];
    poly[9793] = poly[16] * poly[175] - poly[1702] - poly[1662] - poly[1661] - poly[1702];
    poly[9794] = poly[16] * poly[176] - poly[1703] - poly[1685] - poly[1684] - poly[1703];
    poly[9795] = poly[16] * poly[177] - poly[1704];
    poly[9796] = poly[16] * poly[178] - poly[1704];
    poly[9797] = poly[12] * poly[662] - poly[9791];
    poly[9798] = poly[13] * poly[662] - poly[9792];
    poly[9799] = poly[14] * poly[662] - poly[9793];
}

fn f_polynomials196(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9800] = poly[15] * poly[662] - poly[9794];
    poly[9801] = poly[16] * poly[183] - poly[1704];
    poly[9802] = poly[1] * poly[184];
    poly[9803] = poly[2] * poly[185];
    poly[9804] = poly[3] * poly[186];
    poly[9805] = poly[4] * poly[187];
    poly[9806] = poly[5] * poly[188];
    poly[9807] = poly[6] * poly[189];
    poly[9808] = poly[17] * poly[653];
    poly[9809] = poly[17] * poly[654];
    poly[9810] = poly[17] * poly[655];
    poly[9811] = poly[17] * poly[656];
    poly[9812] = poly[11] * poly[194];
    poly[9813] = poly[17] * poly[658];
    poly[9814] = poly[17] * poly[659];
    poly[9815] = poly[17] * poly[660];
    poly[9816] = poly[17] * poly[661];
    poly[9817] = poly[17] * poly[662];
    poly[9818] = poly[1] * poly[663];
    poly[9819] = poly[2] * poly[663];
    poly[9820] = poly[3] * poly[663];
    poly[9821] = poly[4] * poly[663];
    poly[9822] = poly[5] * poly[663];
    poly[9823] = poly[6] * poly[663];
    poly[9824] = poly[17] * poly[190];
    poly[9825] = poly[17] * poly[191];
    poly[9826] = poly[17] * poly[192];
    poly[9827] = poly[17] * poly[193];
    poly[9828] = poly[11] * poly[663];
    poly[9829] = poly[17] * poly[195];
    poly[9830] = poly[17] * poly[196];
    poly[9831] = poly[17] * poly[197];
    poly[9832] = poly[17] * poly[198];
    poly[9833] = poly[17] * poly[199];
    poly[9834] = poly[1] * poly[200];
    poly[9835] = poly[2] * poly[201];
    poly[9836] = poly[3] * poly[202];
    poly[9837] = poly[4] * poly[203];
    poly[9838] = poly[5] * poly[204];
    poly[9839] = poly[6] * poly[205];
    poly[9840] = poly[18] * poly[653];
    poly[9841] = poly[18] * poly[654];
    poly[9842] = poly[18] * poly[655];
    poly[9843] = poly[18] * poly[656];
    poly[9844] = poly[11] * poly[210];
    poly[9845] = poly[18] * poly[658];
    poly[9846] = poly[18] * poly[659];
    poly[9847] = poly[18] * poly[660];
    poly[9848] = poly[18] * poly[661];
    poly[9849] = poly[18] * poly[662];
}

fn f_polynomials197(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9850] = poly[17] * poly[216];
    poly[9851] = poly[1] * poly[664];
    poly[9852] = poly[2] * poly[664];
    poly[9853] = poly[3] * poly[664];
    poly[9854] = poly[4] * poly[664];
    poly[9855] = poly[5] * poly[664];
    poly[9856] = poly[6] * poly[664];
    poly[9857] = poly[7] * poly[664];
    poly[9858] = poly[8] * poly[664];
    poly[9859] = poly[9] * poly[664];
    poly[9860] = poly[10] * poly[664];
    poly[9861] = poly[11] * poly[664];
    poly[9862] = poly[12] * poly[664];
    poly[9863] = poly[13] * poly[664];
    poly[9864] = poly[14] * poly[664];
    poly[9865] = poly[15] * poly[664];
    poly[9866] = poly[16] * poly[664];
    poly[9867] = poly[17] * poly[664];
    poly[9868] = poly[217] * poly[18];
    poly[9869] = poly[1] * poly[218];
    poly[9870] = poly[2] * poly[219];
    poly[9871] = poly[3] * poly[220];
    poly[9872] = poly[4] * poly[221];
    poly[9873] = poly[5] * poly[222];
    poly[9874] = poly[6] * poly[223];
    poly[9875] = poly[19] * poly[653];
    poly[9876] = poly[19] * poly[654];
    poly[9877] = poly[19] * poly[655];
    poly[9878] = poly[19] * poly[656];
    poly[9879] = poly[11] * poly[228];
    poly[9880] = poly[19] * poly[658];
    poly[9881] = poly[19] * poly[659];
    poly[9882] = poly[19] * poly[660];
    poly[9883] = poly[19] * poly[661];
    poly[9884] = poly[19] * poly[662];
    poly[9885] = poly[17] * poly[234];
    poly[9886] = poly[18] * poly[235] - poly[2262];
    poly[9887] = poly[18] * poly[237] - poly[2262];
    poly[9888] = poly[1] * poly[665];
    poly[9889] = poly[2] * poly[665];
    poly[9890] = poly[3] * poly[665];
    poly[9891] = poly[4] * poly[665];
    poly[9892] = poly[5] * poly[665];
    poly[9893] = poly[6] * poly[665];
    poly[9894] = poly[7] * poly[665];
    poly[9895] = poly[8] * poly[665];
    poly[9896] = poly[9] * poly[665];
    poly[9897] = poly[10] * poly[665];
    poly[9898] = poly[11] * poly[665];
    poly[9899] = poly[12] * poly[665];
}

fn f_polynomials198(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9900] = poly[13] * poly[665];
    poly[9901] = poly[14] * poly[665];
    poly[9902] = poly[15] * poly[665];
    poly[9903] = poly[16] * poly[665];
    poly[9904] = poly[17] * poly[665];
    poly[9905] = poly[19] * poly[235] - poly[2263];
    poly[9906] = poly[236] * poly[19];
    poly[9907] = poly[18] * poly[665] - poly[9905];
    poly[9908] = poly[1] * poly[238];
    poly[9909] = poly[2] * poly[239];
    poly[9910] = poly[3] * poly[240];
    poly[9911] = poly[4] * poly[241];
    poly[9912] = poly[5] * poly[242];
    poly[9913] = poly[6] * poly[243];
    poly[9914] = poly[20] * poly[653];
    poly[9915] = poly[20] * poly[654];
    poly[9916] = poly[20] * poly[655];
    poly[9917] = poly[20] * poly[656];
    poly[9918] = poly[11] * poly[248];
    poly[9919] = poly[20] * poly[658];
    poly[9920] = poly[20] * poly[659];
    poly[9921] = poly[20] * poly[660];
    poly[9922] = poly[20] * poly[661];
    poly[9923] = poly[20] * poly[662];
    poly[9924] = poly[17] * poly[254];
    poly[9925] = poly[18] * poly[255] - poly[2500];
    poly[9926] = poly[19] * poly[256] - poly[2521];
    poly[9927] = poly[18] * poly[258] - poly[2500];
    poly[9928] = poly[19] * poly[259] - poly[2521];
    poly[9929] = poly[1] * poly[666];
    poly[9930] = poly[2] * poly[666];
    poly[9931] = poly[3] * poly[666];
    poly[9932] = poly[4] * poly[666];
    poly[9933] = poly[5] * poly[666];
    poly[9934] = poly[6] * poly[666];
    poly[9935] = poly[7] * poly[666];
    poly[9936] = poly[8] * poly[666];
    poly[9937] = poly[9] * poly[666];
    poly[9938] = poly[10] * poly[666];
    poly[9939] = poly[11] * poly[666];
    poly[9940] = poly[12] * poly[666];
    poly[9941] = poly[13] * poly[666];
    poly[9942] = poly[14] * poly[666];
    poly[9943] = poly[15] * poly[666];
    poly[9944] = poly[16] * poly[666];
    poly[9945] = poly[17] * poly[666];
    poly[9946] = poly[20] * poly[255] - poly[2502];
    poly[9947] = poly[20] * poly[256] - poly[2522];
    poly[9948] = poly[257] * poly[20];
    poly[9949] = poly[18] * poly[666] - poly[9946];
}

fn f_polynomials199(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[9950] = poly[19] * poly[666] - poly[9947];
    poly[9951] = poly[1] * poly[260];
    poly[9952] = poly[2] * poly[261];
    poly[9953] = poly[3] * poly[262];
    poly[9954] = poly[4] * poly[263];
    poly[9955] = poly[5] * poly[264];
    poly[9956] = poly[6] * poly[265];
    poly[9957] = poly[21] * poly[653];
    poly[9958] = poly[21] * poly[654];
    poly[9959] = poly[21] * poly[655];
    poly[9960] = poly[21] * poly[656];
    poly[9961] = poly[11] * poly[270];
    poly[9962] = poly[21] * poly[658];
    poly[9963] = poly[21] * poly[659];
    poly[9964] = poly[21] * poly[660];
    poly[9965] = poly[21] * poly[661];
    poly[9966] = poly[21] * poly[662];
    poly[9967] = poly[17] * poly[276];
    poly[9968] = poly[18] * poly[277] - poly[2779];
    poly[9969] = poly[19] * poly[278] - poly[2801];
    poly[9970] = poly[20] * poly[279] - poly[2824];
    poly[9971] = poly[18] * poly[281] - poly[2779];
    poly[9972] = poly[19] * poly[282] - poly[2801];
    poly[9973] = poly[20] * poly[283] - poly[2824];
    poly[9974] = poly[1] * poly[667];
    poly[9975] = poly[2] * poly[667];
    poly[9976] = poly[3] * poly[667];
    poly[9977] = poly[4] * poly[667];
    poly[9978] = poly[5] * poly[667];
    poly[9979] = poly[6] * poly[667];
    poly[9980] = poly[7] * poly[667];
    poly[9981] = poly[8] * poly[667];
    poly[9982] = poly[9] * poly[667];
    poly[9983] = poly[10] * poly[667];
    poly[9984] = poly[11] * poly[667];
    poly[9985] = poly[12] * poly[667];
    poly[9986] = poly[13] * poly[667];
    poly[9987] = poly[14] * poly[667];
    poly[9988] = poly[15] * poly[667];
    poly[9989] = poly[16] * poly[667];
    poly[9990] = poly[17] * poly[667];
    poly[9991] = poly[21] * poly[277] - poly[2782];
    poly[9992] = poly[21] * poly[278] - poly[2803];
    poly[9993] = poly[21] * poly[279] - poly[2825];
    poly[9994] = poly[280] * poly[21];
    poly[9995] = poly[18] * poly[667] - poly[9991];
    poly[9996] = poly[19] * poly[667] - poly[9992];
    poly[9997] = poly[20] * poly[667] - poly[9993];
    poly[9998] = poly[1] * poly[284];
    poly[9999] = poly[2] * poly[285];
}

fn f_polynomials200(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10000] = poly[3] * poly[286];
    poly[10001] = poly[4] * poly[287];
    poly[10002] = poly[5] * poly[288];
    poly[10003] = poly[6] * poly[289];
    poly[10004] = poly[7] * poly[290] - poly[2879];
    poly[10005] = poly[8] * poly[291] - poly[2890];
    poly[10006] = poly[9] * poly[292] - poly[2902];
    poly[10007] = poly[10] * poly[293] - poly[2915];
    poly[10008] = poly[7] * poly[294] - poly[2879];
    poly[10009] = poly[8] * poly[295] - poly[2890];
    poly[10010] = poly[9] * poly[296] - poly[2902];
    poly[10011] = poly[10] * poly[297] - poly[2915];
    poly[10012] = poly[11] * poly[298];
    poly[10013] = poly[22] * poly[658];
    poly[10014] = poly[22] * poly[659];
    poly[10015] = poly[22] * poly[660];
    poly[10016] = poly[22] * poly[661];
    poly[10017] = poly[16] * poly[303] - poly[3058] - poly[3052] - poly[3028] - poly[3028];
    poly[10018] = poly[22] * poly[662] - poly[10017];
    poly[10019] = poly[17] * poly[305];
    poly[10020] = poly[18] * poly[306] - poly[3235];
    poly[10021] = poly[19] * poly[307] - poly[3264];
    poly[10022] = poly[20] * poly[308] - poly[3294];
    poly[10023] = poly[21] * poly[309] - poly[3325];
    poly[10024] = poly[18] * poly[312] - poly[3235];
    poly[10025] = poly[19] * poly[313] - poly[3264];
    poly[10026] = poly[20] * poly[314] - poly[3294];
    poly[10027] = poly[21] * poly[315] - poly[3325];
    poly[10028] = poly[1] * poly[668];
    poly[10029] = poly[2] * poly[668];
    poly[10030] = poly[3] * poly[668];
    poly[10031] = poly[4] * poly[668];
    poly[10032] = poly[5] * poly[668];
    poly[10033] = poly[6] * poly[668];
    poly[10034] = poly[22] * poly[290] - poly[3337] - poly[3197] - poly[3180] - poly[3197];
    poly[10035] = poly[22] * poly[291] - poly[3338] - poly[3198] - poly[3181] - poly[3198];
    poly[10036] = poly[22] * poly[292] - poly[3339] - poly[3199] - poly[3182] - poly[3199];
    poly[10037] = poly[22] * poly[293] - poly[3340] - poly[3200] - poly[3183] - poly[3200];
    poly[10038] = poly[7] * poly[668] - poly[10034];
    poly[10039] = poly[8] * poly[668] - poly[10035];
    poly[10040] = poly[9] * poly[668] - poly[10036];
    poly[10041] = poly[10] * poly[668] - poly[10037];
    poly[10042] = poly[11] * poly[668];
    poly[10043] = poly[12] * poly[668];
    poly[10044] = poly[13] * poly[668];
    poly[10045] = poly[14] * poly[668];
    poly[10046] = poly[15] * poly[668];
    poly[10047] = poly[22] * poly[303] - poly[3346] - poly[3210] - poly[3189] - poly[3210];
    poly[10048] = poly[16] * poly[668] - poly[10047];
    poly[10049] = poly[17] * poly[668];
}

fn f_polynomials201(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10050] = poly[22] * poly[306] - poly[3348] - poly[3240] - poly[3239] - poly[3348];
    poly[10051] = poly[22] * poly[307] - poly[3349] - poly[3268] - poly[3267] - poly[3349];
    poly[10052] = poly[22] * poly[308] - poly[3350] - poly[3297] - poly[3296] - poly[3350];
    poly[10053] = poly[22] * poly[309] - poly[3351] - poly[3327] - poly[3326] - poly[3351];
    poly[10054] = poly[22] * poly[310] - poly[3352];
    poly[10055] = poly[22] * poly[311] - poly[3352];
    poly[10056] = poly[18] * poly[668] - poly[10050];
    poly[10057] = poly[19] * poly[668] - poly[10051];
    poly[10058] = poly[20] * poly[668] - poly[10052];
    poly[10059] = poly[21] * poly[668] - poly[10053];
    poly[10060] = poly[22] * poly[316] - poly[3352];
    poly[10061] = poly[1] * poly[317];
    poly[10062] = poly[2] * poly[318];
    poly[10063] = poly[3] * poly[319];
    poly[10064] = poly[4] * poly[320];
    poly[10065] = poly[5] * poly[321];
    poly[10066] = poly[6] * poly[322];
    poly[10067] = poly[23] * poly[653];
    poly[10068] = poly[23] * poly[654];
    poly[10069] = poly[23] * poly[655];
    poly[10070] = poly[23] * poly[656];
    poly[10071] = poly[11] * poly[327];
    poly[10072] = poly[12] * poly[328] - poly[3503];
    poly[10073] = poly[13] * poly[329] - poly[3520];
    poly[10074] = poly[14] * poly[330] - poly[3538];
    poly[10075] = poly[15] * poly[331] - poly[3557];
    poly[10076] = poly[16] * poly[332] - poly[3582] - poly[3581] - poly[3491] - poly[3491];
    poly[10077] = poly[12] * poly[333] - poly[3503];
    poly[10078] = poly[13] * poly[334] - poly[3520];
    poly[10079] = poly[14] * poly[335] - poly[3538];
    poly[10080] = poly[15] * poly[336] - poly[3557];
    poly[10081] = poly[23] * poly[662] - poly[10076];
    poly[10082] = poly[17] * poly[338];
    poly[10083] = poly[18] * poly[339] - poly[3797];
    poly[10084] = poly[19] * poly[340] - poly[3827];
    poly[10085] = poly[20] * poly[341] - poly[3858];
    poly[10086] = poly[21] * poly[342] - poly[3890];
    poly[10087] = poly[22] * poly[343] - poly[3930] - poly[3929] - poly[3735] - poly[3735];
    poly[10088] = poly[18] * poly[346] - poly[3797];
    poly[10089] = poly[19] * poly[347] - poly[3827];
    poly[10090] = poly[20] * poly[348] - poly[3858];
    poly[10091] = poly[21] * poly[349] - poly[3890];
    poly[10092] = poly[23] * poly[668] - poly[10087];
    poly[10093] = poly[1] * poly[669];
    poly[10094] = poly[2] * poly[669];
    poly[10095] = poly[3] * poly[669];
    poly[10096] = poly[4] * poly[669];
    poly[10097] = poly[5] * poly[669];
    poly[10098] = poly[6] * poly[669];
    poly[10099] = poly[7] * poly[669];
}

fn f_polynomials202(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10100] = poly[8] * poly[669];
    poly[10101] = poly[9] * poly[669];
    poly[10102] = poly[10] * poly[669];
    poly[10103] = poly[11] * poly[669];
    poly[10104] = poly[23] * poly[328] - poly[3949] - poly[3764] - poly[3747] - poly[3764];
    poly[10105] = poly[23] * poly[329] - poly[3950] - poly[3765] - poly[3748] - poly[3765];
    poly[10106] = poly[23] * poly[330] - poly[3951] - poly[3766] - poly[3749] - poly[3766];
    poly[10107] = poly[23] * poly[331] - poly[3952] - poly[3767] - poly[3750] - poly[3767];
    poly[10108] = poly[23] * poly[332] - poly[3953] - poly[3768] - poly[3751] - poly[3768];
    poly[10109] = poly[12] * poly[669] - poly[10104];
    poly[10110] = poly[13] * poly[669] - poly[10105];
    poly[10111] = poly[14] * poly[669] - poly[10106];
    poly[10112] = poly[15] * poly[669] - poly[10107];
    poly[10113] = poly[16] * poly[669] - poly[10108];
    poly[10114] = poly[17] * poly[669];
    poly[10115] = poly[23] * poly[339] - poly[3955] - poly[3803] - poly[3802] - poly[3955];
    poly[10116] = poly[23] * poly[340] - poly[3956] - poly[3832] - poly[3831] - poly[3956];
    poly[10117] = poly[23] * poly[341] - poly[3957] - poly[3862] - poly[3861] - poly[3957];
    poly[10118] = poly[23] * poly[342] - poly[3958] - poly[3893] - poly[3892] - poly[3958];
    poly[10119] = poly[23] * poly[343] - poly[3959] - poly[3932] - poly[3931] - poly[3959];
    poly[10120] = poly[23] * poly[344] - poly[3960];
    poly[10121] = poly[23] * poly[345] - poly[3960];
    poly[10122] = poly[18] * poly[669] - poly[10115];
    poly[10123] = poly[19] * poly[669] - poly[10116];
    poly[10124] = poly[20] * poly[669] - poly[10117];
    poly[10125] = poly[21] * poly[669] - poly[10118];
    poly[10126] = poly[22] * poly[669] - poly[10119];
    poly[10127] = poly[23] * poly[351] - poly[3960];
    poly[10128] = poly[1] * poly[352];
    poly[10129] = poly[2] * poly[353];
    poly[10130] = poly[3] * poly[354];
    poly[10131] = poly[4] * poly[355];
    poly[10132] = poly[5] * poly[356];
    poly[10133] = poly[6] * poly[357];
    poly[10134] = poly[24] * poly[653];
    poly[10135] = poly[24] * poly[654];
    poly[10136] = poly[24] * poly[655];
    poly[10137] = poly[24] * poly[656];
    poly[10138] = poly[11] * poly[362];
    poly[10139] = poly[24] * poly[658];
    poly[10140] = poly[24] * poly[659];
    poly[10141] = poly[24] * poly[660];
    poly[10142] = poly[24] * poly[661];
    poly[10143] = poly[24] * poly[662];
    poly[10144] = poly[17] * poly[368];
    poly[10145] = poly[24] * poly[664];
    poly[10146] = poly[24] * poly[665];
    poly[10147] = poly[24] * poly[666];
    poly[10148] = poly[24] * poly[667];
    poly[10149] = poly[24] * poly[668];
}

fn f_polynomials203(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10150] = poly[24] * poly[669];
    poly[10151] = poly[1] * poly[670];
    poly[10152] = poly[2] * poly[670];
    poly[10153] = poly[3] * poly[670];
    poly[10154] = poly[4] * poly[670];
    poly[10155] = poly[5] * poly[670];
    poly[10156] = poly[6] * poly[670];
    poly[10157] = poly[24] * poly[358];
    poly[10158] = poly[24] * poly[359];
    poly[10159] = poly[24] * poly[360];
    poly[10160] = poly[24] * poly[361];
    poly[10161] = poly[11] * poly[670];
    poly[10162] = poly[24] * poly[363];
    poly[10163] = poly[24] * poly[364];
    poly[10164] = poly[24] * poly[365];
    poly[10165] = poly[24] * poly[366];
    poly[10166] = poly[24] * poly[367];
    poly[10167] = poly[17] * poly[670];
    poly[10168] = poly[24] * poly[369];
    poly[10169] = poly[24] * poly[370];
    poly[10170] = poly[24] * poly[371];
    poly[10171] = poly[24] * poly[372];
    poly[10172] = poly[24] * poly[373];
    poly[10173] = poly[24] * poly[374];
    poly[10174] = poly[1] * poly[375];
    poly[10175] = poly[2] * poly[376];
    poly[10176] = poly[3] * poly[377];
    poly[10177] = poly[4] * poly[378];
    poly[10178] = poly[5] * poly[379];
    poly[10179] = poly[6] * poly[380];
    poly[10180] = poly[25] * poly[653];
    poly[10181] = poly[25] * poly[654];
    poly[10182] = poly[25] * poly[655];
    poly[10183] = poly[25] * poly[656];
    poly[10184] = poly[11] * poly[385];
    poly[10185] = poly[25] * poly[658];
    poly[10186] = poly[25] * poly[659];
    poly[10187] = poly[25] * poly[660];
    poly[10188] = poly[25] * poly[661];
    poly[10189] = poly[25] * poly[662];
    poly[10190] = poly[17] * poly[391];
    poly[10191] = poly[25] * poly[664];
    poly[10192] = poly[25] * poly[665];
    poly[10193] = poly[25] * poly[666];
    poly[10194] = poly[25] * poly[667];
    poly[10195] = poly[25] * poly[668];
    poly[10196] = poly[25] * poly[669];
    poly[10197] = poly[24] * poly[398];
    poly[10198] = poly[1] * poly[671];
    poly[10199] = poly[2] * poly[671];
}

fn f_polynomials204(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10200] = poly[3] * poly[671];
    poly[10201] = poly[4] * poly[671];
    poly[10202] = poly[5] * poly[671];
    poly[10203] = poly[6] * poly[671];
    poly[10204] = poly[7] * poly[671];
    poly[10205] = poly[8] * poly[671];
    poly[10206] = poly[9] * poly[671];
    poly[10207] = poly[10] * poly[671];
    poly[10208] = poly[11] * poly[671];
    poly[10209] = poly[12] * poly[671];
    poly[10210] = poly[13] * poly[671];
    poly[10211] = poly[14] * poly[671];
    poly[10212] = poly[15] * poly[671];
    poly[10213] = poly[16] * poly[671];
    poly[10214] = poly[17] * poly[671];
    poly[10215] = poly[18] * poly[671];
    poly[10216] = poly[19] * poly[671];
    poly[10217] = poly[20] * poly[671];
    poly[10218] = poly[21] * poly[671];
    poly[10219] = poly[22] * poly[671];
    poly[10220] = poly[23] * poly[671];
    poly[10221] = poly[24] * poly[671];
    poly[10222] = poly[399] * poly[25];
    poly[10223] = poly[1] * poly[400];
    poly[10224] = poly[2] * poly[401];
    poly[10225] = poly[3] * poly[402];
    poly[10226] = poly[4] * poly[403];
    poly[10227] = poly[5] * poly[404];
    poly[10228] = poly[6] * poly[405];
    poly[10229] = poly[26] * poly[653];
    poly[10230] = poly[26] * poly[654];
    poly[10231] = poly[26] * poly[655];
    poly[10232] = poly[26] * poly[656];
    poly[10233] = poly[11] * poly[410];
    poly[10234] = poly[26] * poly[658];
    poly[10235] = poly[26] * poly[659];
    poly[10236] = poly[26] * poly[660];
    poly[10237] = poly[26] * poly[661];
    poly[10238] = poly[26] * poly[662];
    poly[10239] = poly[17] * poly[416];
    poly[10240] = poly[26] * poly[664];
    poly[10241] = poly[26] * poly[665];
    poly[10242] = poly[26] * poly[666];
    poly[10243] = poly[26] * poly[667];
    poly[10244] = poly[26] * poly[668];
    poly[10245] = poly[26] * poly[669];
    poly[10246] = poly[24] * poly[423];
    poly[10247] = poly[25] * poly[424] - poly[5065];
    poly[10248] = poly[25] * poly[426] - poly[5065];
    poly[10249] = poly[1] * poly[672];
}

fn f_polynomials205(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10250] = poly[2] * poly[672];
    poly[10251] = poly[3] * poly[672];
    poly[10252] = poly[4] * poly[672];
    poly[10253] = poly[5] * poly[672];
    poly[10254] = poly[6] * poly[672];
    poly[10255] = poly[7] * poly[672];
    poly[10256] = poly[8] * poly[672];
    poly[10257] = poly[9] * poly[672];
    poly[10258] = poly[10] * poly[672];
    poly[10259] = poly[11] * poly[672];
    poly[10260] = poly[12] * poly[672];
    poly[10261] = poly[13] * poly[672];
    poly[10262] = poly[14] * poly[672];
    poly[10263] = poly[15] * poly[672];
    poly[10264] = poly[16] * poly[672];
    poly[10265] = poly[17] * poly[672];
    poly[10266] = poly[18] * poly[672];
    poly[10267] = poly[19] * poly[672];
    poly[10268] = poly[20] * poly[672];
    poly[10269] = poly[21] * poly[672];
    poly[10270] = poly[22] * poly[672];
    poly[10271] = poly[23] * poly[672];
    poly[10272] = poly[24] * poly[672];
    poly[10273] = poly[26] * poly[424] - poly[5066];
    poly[10274] = poly[425] * poly[26];
    poly[10275] = poly[25] * poly[672] - poly[10273];
    poly[10276] = poly[1] * poly[427];
    poly[10277] = poly[2] * poly[428];
    poly[10278] = poly[3] * poly[429];
    poly[10279] = poly[4] * poly[430];
    poly[10280] = poly[5] * poly[431];
    poly[10281] = poly[6] * poly[432];
    poly[10282] = poly[27] * poly[653];
    poly[10283] = poly[27] * poly[654];
    poly[10284] = poly[27] * poly[655];
    poly[10285] = poly[27] * poly[656];
    poly[10286] = poly[11] * poly[437];
    poly[10287] = poly[27] * poly[658];
    poly[10288] = poly[27] * poly[659];
    poly[10289] = poly[27] * poly[660];
    poly[10290] = poly[27] * poly[661];
    poly[10291] = poly[27] * poly[662];
    poly[10292] = poly[17] * poly[443];
    poly[10293] = poly[27] * poly[664];
    poly[10294] = poly[27] * poly[665];
    poly[10295] = poly[27] * poly[666];
    poly[10296] = poly[27] * poly[667];
    poly[10297] = poly[27] * poly[668];
    poly[10298] = poly[27] * poly[669];
    poly[10299] = poly[24] * poly[450];
}

fn f_polynomials206(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10300] = poly[25] * poly[451] - poly[5506];
    poly[10301] = poly[26] * poly[452] - poly[5534];
    poly[10302] = poly[25] * poly[454] - poly[5506];
    poly[10303] = poly[26] * poly[455] - poly[5534];
    poly[10304] = poly[1] * poly[673];
    poly[10305] = poly[2] * poly[673];
    poly[10306] = poly[3] * poly[673];
    poly[10307] = poly[4] * poly[673];
    poly[10308] = poly[5] * poly[673];
    poly[10309] = poly[6] * poly[673];
    poly[10310] = poly[7] * poly[673];
    poly[10311] = poly[8] * poly[673];
    poly[10312] = poly[9] * poly[673];
    poly[10313] = poly[10] * poly[673];
    poly[10314] = poly[11] * poly[673];
    poly[10315] = poly[12] * poly[673];
    poly[10316] = poly[13] * poly[673];
    poly[10317] = poly[14] * poly[673];
    poly[10318] = poly[15] * poly[673];
    poly[10319] = poly[16] * poly[673];
    poly[10320] = poly[17] * poly[673];
    poly[10321] = poly[18] * poly[673];
    poly[10322] = poly[19] * poly[673];
    poly[10323] = poly[20] * poly[673];
    poly[10324] = poly[21] * poly[673];
    poly[10325] = poly[22] * poly[673];
    poly[10326] = poly[23] * poly[673];
    poly[10327] = poly[24] * poly[673];
    poly[10328] = poly[27] * poly[451] - poly[5508];
    poly[10329] = poly[27] * poly[452] - poly[5535];
    poly[10330] = poly[453] * poly[27];
    poly[10331] = poly[25] * poly[673] - poly[10328];
    poly[10332] = poly[26] * poly[673] - poly[10329];
    poly[10333] = poly[1] * poly[456];
    poly[10334] = poly[2] * poly[457];
    poly[10335] = poly[3] * poly[458];
    poly[10336] = poly[4] * poly[459];
    poly[10337] = poly[5] * poly[460];
    poly[10338] = poly[6] * poly[461];
    poly[10339] = poly[28] * poly[653];
    poly[10340] = poly[28] * poly[654];
    poly[10341] = poly[28] * poly[655];
    poly[10342] = poly[28] * poly[656];
    poly[10343] = poly[11] * poly[466];
    poly[10344] = poly[28] * poly[658];
    poly[10345] = poly[28] * poly[659];
    poly[10346] = poly[28] * poly[660];
    poly[10347] = poly[28] * poly[661];
    poly[10348] = poly[28] * poly[662];
    poly[10349] = poly[17] * poly[472];
}

fn f_polynomials207(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10350] = poly[28] * poly[664];
    poly[10351] = poly[28] * poly[665];
    poly[10352] = poly[28] * poly[666];
    poly[10353] = poly[28] * poly[667];
    poly[10354] = poly[28] * poly[668];
    poly[10355] = poly[28] * poly[669];
    poly[10356] = poly[24] * poly[479];
    poly[10357] = poly[25] * poly[480] - poly[6002];
    poly[10358] = poly[26] * poly[481] - poly[6031];
    poly[10359] = poly[27] * poly[482] - poly[6061];
    poly[10360] = poly[25] * poly[484] - poly[6002];
    poly[10361] = poly[26] * poly[485] - poly[6031];
    poly[10362] = poly[27] * poly[486] - poly[6061];
    poly[10363] = poly[1] * poly[674];
    poly[10364] = poly[2] * poly[674];
    poly[10365] = poly[3] * poly[674];
    poly[10366] = poly[4] * poly[674];
    poly[10367] = poly[5] * poly[674];
    poly[10368] = poly[6] * poly[674];
    poly[10369] = poly[7] * poly[674];
    poly[10370] = poly[8] * poly[674];
    poly[10371] = poly[9] * poly[674];
    poly[10372] = poly[10] * poly[674];
    poly[10373] = poly[11] * poly[674];
    poly[10374] = poly[12] * poly[674];
    poly[10375] = poly[13] * poly[674];
    poly[10376] = poly[14] * poly[674];
    poly[10377] = poly[15] * poly[674];
    poly[10378] = poly[16] * poly[674];
    poly[10379] = poly[17] * poly[674];
    poly[10380] = poly[18] * poly[674];
    poly[10381] = poly[19] * poly[674];
    poly[10382] = poly[20] * poly[674];
    poly[10383] = poly[21] * poly[674];
    poly[10384] = poly[22] * poly[674];
    poly[10385] = poly[23] * poly[674];
    poly[10386] = poly[24] * poly[674];
    poly[10387] = poly[28] * poly[480] - poly[6005];
    poly[10388] = poly[28] * poly[481] - poly[6033];
    poly[10389] = poly[28] * poly[482] - poly[6062];
    poly[10390] = poly[483] * poly[28];
    poly[10391] = poly[25] * poly[674] - poly[10387];
    poly[10392] = poly[26] * poly[674] - poly[10388];
    poly[10393] = poly[27] * poly[674] - poly[10389];
    poly[10394] = poly[1] * poly[487];
    poly[10395] = poly[2] * poly[488];
    poly[10396] = poly[3] * poly[489];
    poly[10397] = poly[4] * poly[490];
    poly[10398] = poly[5] * poly[491];
    poly[10399] = poly[6] * poly[492];
}

fn f_polynomials208(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10400] = poly[7] * poly[493] - poly[6116];
    poly[10401] = poly[8] * poly[494] - poly[6127];
    poly[10402] = poly[9] * poly[495] - poly[6139];
    poly[10403] = poly[10] * poly[496] - poly[6152];
    poly[10404] = poly[7] * poly[497] - poly[6116];
    poly[10405] = poly[8] * poly[498] - poly[6127];
    poly[10406] = poly[9] * poly[499] - poly[6139];
    poly[10407] = poly[10] * poly[500] - poly[6152];
    poly[10408] = poly[11] * poly[501];
    poly[10409] = poly[29] * poly[658];
    poly[10410] = poly[29] * poly[659];
    poly[10411] = poly[29] * poly[660];
    poly[10412] = poly[29] * poly[661];
    poly[10413] = poly[16] * poly[506] - poly[6295] - poly[6289] - poly[6265] - poly[6265];
    poly[10414] = poly[29] * poly[662] - poly[10413];
    poly[10415] = poly[17] * poly[508];
    poly[10416] = poly[29] * poly[664];
    poly[10417] = poly[29] * poly[665];
    poly[10418] = poly[29] * poly[666];
    poly[10419] = poly[29] * poly[667];
    poly[10420] = poly[22] * poly[513] - poly[6484] - poly[6478] - poly[6447] - poly[6447];
    poly[10421] = poly[29] * poly[668] - poly[10420];
    poly[10422] = poly[29] * poly[669];
    poly[10423] = poly[24] * poly[516];
    poly[10424] = poly[25] * poly[517] - poly[6767];
    poly[10425] = poly[26] * poly[518] - poly[6804];
    poly[10426] = poly[27] * poly[519] - poly[6842];
    poly[10427] = poly[28] * poly[520] - poly[6881];
    poly[10428] = poly[25] * poly[523] - poly[6767];
    poly[10429] = poly[26] * poly[524] - poly[6804];
    poly[10430] = poly[27] * poly[525] - poly[6842];
    poly[10431] = poly[28] * poly[526] - poly[6881];
    poly[10432] = poly[1] * poly[675];
    poly[10433] = poly[2] * poly[675];
    poly[10434] = poly[3] * poly[675];
    poly[10435] = poly[4] * poly[675];
    poly[10436] = poly[5] * poly[675];
    poly[10437] = poly[6] * poly[675];
    poly[10438] = poly[29] * poly[493] - poly[6893] - poly[6713] - poly[6689] - poly[6713];
    poly[10439] = poly[29] * poly[494] - poly[6894] - poly[6714] - poly[6690] - poly[6714];
    poly[10440] = poly[29] * poly[495] - poly[6895] - poly[6715] - poly[6691] - poly[6715];
    poly[10441] = poly[29] * poly[496] - poly[6896] - poly[6716] - poly[6692] - poly[6716];
    poly[10442] = poly[7] * poly[675] - poly[10438];
    poly[10443] = poly[8] * poly[675] - poly[10439];
    poly[10444] = poly[9] * poly[675] - poly[10440];
    poly[10445] = poly[10] * poly[675] - poly[10441];
    poly[10446] = poly[11] * poly[675];
    poly[10447] = poly[12] * poly[675];
    poly[10448] = poly[13] * poly[675];
    poly[10449] = poly[14] * poly[675];
}

fn f_polynomials209(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10450] = poly[15] * poly[675];
    poly[10451] = poly[29] * poly[506] - poly[6902] - poly[6726] - poly[6698] - poly[6726];
    poly[10452] = poly[16] * poly[675] - poly[10451];
    poly[10453] = poly[17] * poly[675];
    poly[10454] = poly[18] * poly[675];
    poly[10455] = poly[19] * poly[675];
    poly[10456] = poly[20] * poly[675];
    poly[10457] = poly[21] * poly[675];
    poly[10458] = poly[29] * poly[513] - poly[6908] - poly[6733] - poly[6704] - poly[6733];
    poly[10459] = poly[22] * poly[675] - poly[10458];
    poly[10460] = poly[23] * poly[675];
    poly[10461] = poly[24] * poly[675];
    poly[10462] = poly[29] * poly[517] - poly[6911] - poly[6772] - poly[6771] - poly[6911];
    poly[10463] = poly[29] * poly[518] - poly[6912] - poly[6808] - poly[6807] - poly[6912];
    poly[10464] = poly[29] * poly[519] - poly[6913] - poly[6845] - poly[6844] - poly[6913];
    poly[10465] = poly[29] * poly[520] - poly[6914] - poly[6883] - poly[6882] - poly[6914];
    poly[10466] = poly[29] * poly[521] - poly[6915];
    poly[10467] = poly[29] * poly[522] - poly[6915];
    poly[10468] = poly[25] * poly[675] - poly[10462];
    poly[10469] = poly[26] * poly[675] - poly[10463];
    poly[10470] = poly[27] * poly[675] - poly[10464];
    poly[10471] = poly[28] * poly[675] - poly[10465];
    poly[10472] = poly[29] * poly[527] - poly[6915];
    poly[10473] = poly[1] * poly[528];
    poly[10474] = poly[2] * poly[529];
    poly[10475] = poly[3] * poly[530];
    poly[10476] = poly[4] * poly[531];
    poly[10477] = poly[5] * poly[532];
    poly[10478] = poly[6] * poly[533];
    poly[10479] = poly[30] * poly[653];
    poly[10480] = poly[30] * poly[654];
    poly[10481] = poly[30] * poly[655];
    poly[10482] = poly[30] * poly[656];
    poly[10483] = poly[11] * poly[538];
    poly[10484] = poly[12] * poly[539] - poly[7066];
    poly[10485] = poly[13] * poly[540] - poly[7083];
    poly[10486] = poly[14] * poly[541] - poly[7101];
    poly[10487] = poly[15] * poly[542] - poly[7120];
    poly[10488] = poly[16] * poly[543] - poly[7145] - poly[7144] - poly[7054] - poly[7054];
    poly[10489] = poly[12] * poly[544] - poly[7066];
    poly[10490] = poly[13] * poly[545] - poly[7083];
    poly[10491] = poly[14] * poly[546] - poly[7101];
    poly[10492] = poly[15] * poly[547] - poly[7120];
    poly[10493] = poly[30] * poly[662] - poly[10488];
    poly[10494] = poly[17] * poly[549];
    poly[10495] = poly[30] * poly[664];
    poly[10496] = poly[30] * poly[665];
    poly[10497] = poly[30] * poly[666];
    poly[10498] = poly[30] * poly[667];
    poly[10499] = poly[30] * poly[668];
}

fn f_polynomials210(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10500] = poly[23] * poly[555] - poly[7382] - poly[7375] - poly[7342] - poly[7342];
    poly[10501] = poly[30] * poly[669] - poly[10500];
    poly[10502] = poly[24] * poly[557];
    poly[10503] = poly[25] * poly[558] - poly[7664];
    poly[10504] = poly[26] * poly[559] - poly[7702];
    poly[10505] = poly[27] * poly[560] - poly[7741];
    poly[10506] = poly[28] * poly[561] - poly[7781];
    poly[10507] = poly[29] * poly[562] - poly[7830] - poly[7829] - poly[7579] - poly[7579];
    poly[10508] = poly[25] * poly[565] - poly[7664];
    poly[10509] = poly[26] * poly[566] - poly[7702];
    poly[10510] = poly[27] * poly[567] - poly[7741];
    poly[10511] = poly[28] * poly[568] - poly[7781];
    poly[10512] = poly[30] * poly[675] - poly[10507];
    poly[10513] = poly[1] * poly[676];
    poly[10514] = poly[2] * poly[676];
    poly[10515] = poly[3] * poly[676];
    poly[10516] = poly[4] * poly[676];
    poly[10517] = poly[5] * poly[676];
    poly[10518] = poly[6] * poly[676];
    poly[10519] = poly[7] * poly[676];
    poly[10520] = poly[8] * poly[676];
    poly[10521] = poly[9] * poly[676];
    poly[10522] = poly[10] * poly[676];
    poly[10523] = poly[11] * poly[676];
    poly[10524] = poly[30] * poly[539] - poly[7849] - poly[7615] - poly[7591] - poly[7615];
    poly[10525] = poly[30] * poly[540] - poly[7850] - poly[7616] - poly[7592] - poly[7616];
    poly[10526] = poly[30] * poly[541] - poly[7851] - poly[7617] - poly[7593] - poly[7617];
    poly[10527] = poly[30] * poly[542] - poly[7852] - poly[7618] - poly[7594] - poly[7618];
    poly[10528] = poly[30] * poly[543] - poly[7853] - poly[7619] - poly[7595] - poly[7619];
    poly[10529] = poly[12] * poly[676] - poly[10524];
    poly[10530] = poly[13] * poly[676] - poly[10525];
    poly[10531] = poly[14] * poly[676] - poly[10526];
    poly[10532] = poly[15] * poly[676] - poly[10527];
    poly[10533] = poly[16] * poly[676] - poly[10528];
    poly[10534] = poly[17] * poly[676];
    poly[10535] = poly[18] * poly[676];
    poly[10536] = poly[19] * poly[676];
    poly[10537] = poly[20] * poly[676];
    poly[10538] = poly[21] * poly[676];
    poly[10539] = poly[22] * poly[676];
    poly[10540] = poly[30] * poly[555] - poly[7860] - poly[7631] - poly[7602] - poly[7631];
    poly[10541] = poly[23] * poly[676] - poly[10540];
    poly[10542] = poly[24] * poly[676];
    poly[10543] = poly[30] * poly[558] - poly[7862] - poly[7670] - poly[7669] - poly[7862];
    poly[10544] = poly[30] * poly[559] - poly[7863] - poly[7707] - poly[7706] - poly[7863];
    poly[10545] = poly[30] * poly[560] - poly[7864] - poly[7745] - poly[7744] - poly[7864];
    poly[10546] = poly[30] * poly[561] - poly[7865] - poly[7784] - poly[7783] - poly[7865];
    poly[10547] = poly[30] * poly[562] - poly[7866] - poly[7832] - poly[7831] - poly[7866];
    poly[10548] = poly[30] * poly[563] - poly[7867];
    poly[10549] = poly[30] * poly[564] - poly[7867];
}

fn f_polynomials211(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10550] = poly[25] * poly[676] - poly[10543];
    poly[10551] = poly[26] * poly[676] - poly[10544];
    poly[10552] = poly[27] * poly[676] - poly[10545];
    poly[10553] = poly[28] * poly[676] - poly[10546];
    poly[10554] = poly[29] * poly[676] - poly[10547];
    poly[10555] = poly[30] * poly[570] - poly[7867];
    poly[10556] = poly[1] * poly[571];
    poly[10557] = poly[2] * poly[572];
    poly[10558] = poly[3] * poly[573];
    poly[10559] = poly[4] * poly[574];
    poly[10560] = poly[5] * poly[575];
    poly[10561] = poly[6] * poly[576];
    poly[10562] = poly[31] * poly[653];
    poly[10563] = poly[31] * poly[654];
    poly[10564] = poly[31] * poly[655];
    poly[10565] = poly[31] * poly[656];
    poly[10566] = poly[11] * poly[581];
    poly[10567] = poly[31] * poly[658];
    poly[10568] = poly[31] * poly[659];
    poly[10569] = poly[31] * poly[660];
    poly[10570] = poly[31] * poly[661];
    poly[10571] = poly[31] * poly[662];
    poly[10572] = poly[17] * poly[587];
    poly[10573] = poly[18] * poly[588] - poly[8186];
    poly[10574] = poly[19] * poly[589] - poly[8210];
    poly[10575] = poly[20] * poly[590] - poly[8235];
    poly[10576] = poly[21] * poly[591] - poly[8261];
    poly[10577] = poly[22] * poly[592] - poly[8294] - poly[8293] - poly[8140] - poly[8140];
    poly[10578] = poly[23] * poly[593] - poly[8329] - poly[8328] - poly[8168] - poly[8168];
    poly[10579] = poly[18] * poly[594] - poly[8186];
    poly[10580] = poly[19] * poly[595] - poly[8210];
    poly[10581] = poly[20] * poly[596] - poly[8235];
    poly[10582] = poly[21] * poly[597] - poly[8261];
    poly[10583] = poly[31] * poly[668] - poly[10577];
    poly[10584] = poly[31] * poly[669] - poly[10578];
    poly[10585] = poly[24] * poly[600];
    poly[10586] = poly[25] * poly[601] - poly[8660];
    poly[10587] = poly[26] * poly[602] - poly[8699];
    poly[10588] = poly[27] * poly[603] - poly[8739];
    poly[10589] = poly[28] * poly[604] - poly[8780];
    poly[10590] = poly[29] * poly[605] - poly[8830] - poly[8829] - poly[8532] - poly[8532];
    poly[10591] = poly[30] * poly[606] - poly[8882] - poly[8881] - poly[8575] - poly[8575];
    poly[10592] = poly[25] * poly[609] - poly[8660];
    poly[10593] = poly[26] * poly[610] - poly[8699];
    poly[10594] = poly[27] * poly[611] - poly[8739];
    poly[10595] = poly[28] * poly[612] - poly[8780];
    poly[10596] = poly[31] * poly[675] - poly[10590];
    poly[10597] = poly[31] * poly[676] - poly[10591];
    poly[10598] = poly[1] * poly[677];
    poly[10599] = poly[2] * poly[677];
}

fn f_polynomials212(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10600] = poly[3] * poly[677];
    poly[10601] = poly[4] * poly[677];
    poly[10602] = poly[5] * poly[677];
    poly[10603] = poly[6] * poly[677];
    poly[10604] = poly[7] * poly[677];
    poly[10605] = poly[8] * poly[677];
    poly[10606] = poly[9] * poly[677];
    poly[10607] = poly[10] * poly[677];
    poly[10608] = poly[11] * poly[677];
    poly[10609] = poly[12] * poly[677];
    poly[10610] = poly[13] * poly[677];
    poly[10611] = poly[14] * poly[677];
    poly[10612] = poly[15] * poly[677];
    poly[10613] = poly[16] * poly[677];
    poly[10614] = poly[17] * poly[677];
    poly[10615] = poly[31] * poly[588] - poly[8908] - poly[8617] - poly[8593] - poly[8617];
    poly[10616] = poly[31] * poly[589] - poly[8909] - poly[8618] - poly[8594] - poly[8618];
    poly[10617] = poly[31] * poly[590] - poly[8910] - poly[8619] - poly[8595] - poly[8619];
    poly[10618] = poly[31] * poly[591] - poly[8911] - poly[8620] - poly[8596] - poly[8620];
    poly[10619] = poly[31] * poly[592] - poly[8912] - poly[8621] - poly[8597] - poly[8621];
    poly[10620] = poly[31] * poly[593] - poly[8913] - poly[8622] - poly[8598] - poly[8622];
    poly[10621] = poly[18] * poly[677] - poly[10615];
    poly[10622] = poly[19] * poly[677] - poly[10616];
    poly[10623] = poly[20] * poly[677] - poly[10617];
    poly[10624] = poly[21] * poly[677] - poly[10618];
    poly[10625] = poly[22] * poly[677] - poly[10619];
    poly[10626] = poly[23] * poly[677] - poly[10620];
    poly[10627] = poly[24] * poly[677];
    poly[10628] = poly[31] * poly[601] - poly[8915] - poly[8667] - poly[8666] - poly[8915];
    poly[10629] = poly[31] * poly[602] - poly[8916] - poly[8705] - poly[8704] - poly[8916];
    poly[10630] = poly[31] * poly[603] - poly[8917] - poly[8744] - poly[8743] - poly[8917];
    poly[10631] = poly[31] * poly[604] - poly[8918] - poly[8784] - poly[8783] - poly[8918];
    poly[10632] = poly[31] * poly[605] - poly[8919] - poly[8833] - poly[8832] - poly[8919];
    poly[10633] = poly[31] * poly[606] - poly[8920] - poly[8884] - poly[8883] - poly[8920];
    poly[10634] = poly[31] * poly[607] - poly[8921];
    poly[10635] = poly[31] * poly[608] - poly[8921];
    poly[10636] = poly[25] * poly[677] - poly[10628];
    poly[10637] = poly[26] * poly[677] - poly[10629];
    poly[10638] = poly[27] * poly[677] - poly[10630];
    poly[10639] = poly[28] * poly[677] - poly[10631];
    poly[10640] = poly[29] * poly[677] - poly[10632];
    poly[10641] = poly[30] * poly[677] - poly[10633];
    poly[10642] = poly[31] * poly[615] - poly[8921];
    poly[10643] = poly[1] * poly[616];
    poly[10644] = poly[2] * poly[617];
    poly[10645] = poly[3] * poly[618];
    poly[10646] = poly[4] * poly[619];
    poly[10647] = poly[5] * poly[620];
    poly[10648] = poly[6] * poly[621];
    poly[10649] = poly[32] * poly[653];
}

fn f_polynomials213(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10650] = poly[32] * poly[654];
    poly[10651] = poly[32] * poly[655];
    poly[10652] = poly[32] * poly[656];
    poly[10653] = poly[11] * poly[626];
    poly[10654] = poly[32] * poly[658];
    poly[10655] = poly[32] * poly[659];
    poly[10656] = poly[32] * poly[660];
    poly[10657] = poly[32] * poly[661];
    poly[10658] = poly[32] * poly[662];
    poly[10659] = poly[17] * poly[632];
    poly[10660] = poly[32] * poly[664];
    poly[10661] = poly[32] * poly[665];
    poly[10662] = poly[32] * poly[666];
    poly[10663] = poly[32] * poly[667];
    poly[10664] = poly[32] * poly[668];
    poly[10665] = poly[32] * poly[669];
    poly[10666] = poly[24] * poly[639];
    poly[10667] = poly[32] * poly[671];
    poly[10668] = poly[32] * poly[672];
    poly[10669] = poly[32] * poly[673];
    poly[10670] = poly[32] * poly[674];
    poly[10671] = poly[32] * poly[675];
    poly[10672] = poly[32] * poly[676];
    poly[10673] = poly[32] * poly[677];
    poly[10674] = poly[1] * poly[678];
    poly[10675] = poly[2] * poly[678];
    poly[10676] = poly[3] * poly[678];
    poly[10677] = poly[4] * poly[678];
    poly[10678] = poly[5] * poly[678];
    poly[10679] = poly[6] * poly[678];
    poly[10680] = poly[32] * poly[622];
    poly[10681] = poly[32] * poly[623];
    poly[10682] = poly[32] * poly[624];
    poly[10683] = poly[32] * poly[625];
    poly[10684] = poly[11] * poly[678];
    poly[10685] = poly[32] * poly[627];
    poly[10686] = poly[32] * poly[628];
    poly[10687] = poly[32] * poly[629];
    poly[10688] = poly[32] * poly[630];
    poly[10689] = poly[32] * poly[631];
    poly[10690] = poly[17] * poly[678];
    poly[10691] = poly[32] * poly[633];
    poly[10692] = poly[32] * poly[634];
    poly[10693] = poly[32] * poly[635];
    poly[10694] = poly[32] * poly[636];
    poly[10695] = poly[32] * poly[637];
    poly[10696] = poly[32] * poly[638];
    poly[10697] = poly[24] * poly[678];
    poly[10698] = poly[32] * poly[640];
    poly[10699] = poly[32] * poly[641];
}

fn f_polynomials214(poly: &mut [f32; N_POLYS], mono: &[f32; N_MONOS]) {
    poly[10700] = poly[32] * poly[642];
    poly[10701] = poly[32] * poly[643];
    poly[10702] = poly[32] * poly[644];
    poly[10703] = poly[32] * poly[645];
    poly[10704] = poly[32] * poly[646];
    poly[10705] = poly[1] * poly[647];
    poly[10706] = poly[2] * poly[648];
    poly[10707] = poly[3] * poly[649];
    poly[10708] = poly[4] * poly[650];
    poly[10709] = poly[5] * poly[651];
    poly[10710] = poly[6] * poly[652];
    poly[10711] = poly[7] * poly[653] - poly[9553];
    poly[10712] = poly[8] * poly[654] - poly[9569];
    poly[10713] = poly[9] * poly[655] - poly[9589];
    poly[10714] = poly[10] * poly[656] - poly[9613];
    poly[10715] = poly[11] * poly[657];
    poly[10716] = poly[12] * poly[658] - poly[9659];
    poly[10717] = poly[13] * poly[659] - poly[9685];
    poly[10718] = poly[14] * poly[660] - poly[9715];
    poly[10719] = poly[15] * poly[661] - poly[9749];
    poly[10720] = poly[16] * poly[662] - poly[9801] - poly[9796] - poly[9795];
    poly[10721] = poly[17] * poly[663];
    poly[10722] = poly[18] * poly[664] - poly[9868];
    poly[10723] = poly[19] * poly[665] - poly[9906];
    poly[10724] = poly[20] * poly[666] - poly[9948];
    poly[10725] = poly[21] * poly[667] - poly[9994];
    poly[10726] = poly[22] * poly[668] - poly[10060] - poly[10055] - poly[10054];
    poly[10727] = poly[23] * poly[669] - poly[10127] - poly[10121] - poly[10120];
    poly[10728] = poly[24] * poly[670];
    poly[10729] = poly[25] * poly[671] - poly[10222];
    poly[10730] = poly[26] * poly[672] - poly[10274];
    poly[10731] = poly[27] * poly[673] - poly[10330];
    poly[10732] = poly[28] * poly[674] - poly[10390];
    poly[10733] = poly[29] * poly[675] - poly[10472] - poly[10467] - poly[10466];
    poly[10734] = poly[30] * poly[676] - poly[10555] - poly[10549] - poly[10548];
    poly[10735] = poly[31] * poly[677] - poly[10642] - poly[10635] - poly[10634];
    poly[10736] = poly[32] * poly[678];
}

// Total number of monomials = 10737

pub fn f_polynomials(r: &[f32; N_DISTANCES]) -> [f32; N_POLYS] {
    let mono: [f32; N_MONOS] = f_monomials(r);
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
    f_polynomials12(&mut poly, &mono);
    f_polynomials13(&mut poly, &mono);
    f_polynomials14(&mut poly, &mono);
    f_polynomials15(&mut poly, &mono);
    f_polynomials16(&mut poly, &mono);
    f_polynomials17(&mut poly, &mono);
    f_polynomials18(&mut poly, &mono);
    f_polynomials19(&mut poly, &mono);
    f_polynomials20(&mut poly, &mono);
    f_polynomials21(&mut poly, &mono);
    f_polynomials22(&mut poly, &mono);
    f_polynomials23(&mut poly, &mono);
    f_polynomials24(&mut poly, &mono);
    f_polynomials25(&mut poly, &mono);
    f_polynomials26(&mut poly, &mono);
    f_polynomials27(&mut poly, &mono);
    f_polynomials28(&mut poly, &mono);
    f_polynomials29(&mut poly, &mono);
    f_polynomials30(&mut poly, &mono);
    f_polynomials31(&mut poly, &mono);
    f_polynomials32(&mut poly, &mono);
    f_polynomials33(&mut poly, &mono);
    f_polynomials34(&mut poly, &mono);
    f_polynomials35(&mut poly, &mono);
    f_polynomials36(&mut poly, &mono);
    f_polynomials37(&mut poly, &mono);
    f_polynomials38(&mut poly, &mono);
    f_polynomials39(&mut poly, &mono);
    f_polynomials40(&mut poly, &mono);
    f_polynomials41(&mut poly, &mono);
    f_polynomials42(&mut poly, &mono);
    f_polynomials43(&mut poly, &mono);
    f_polynomials44(&mut poly, &mono);
    f_polynomials45(&mut poly, &mono);
    f_polynomials46(&mut poly, &mono);
    f_polynomials47(&mut poly, &mono);
    f_polynomials48(&mut poly, &mono);
    f_polynomials49(&mut poly, &mono);
    f_polynomials50(&mut poly, &mono);
    f_polynomials51(&mut poly, &mono);
    f_polynomials52(&mut poly, &mono);
    f_polynomials53(&mut poly, &mono);
    f_polynomials54(&mut poly, &mono);
    f_polynomials55(&mut poly, &mono);
    f_polynomials56(&mut poly, &mono);
    f_polynomials57(&mut poly, &mono);
    f_polynomials58(&mut poly, &mono);
    f_polynomials59(&mut poly, &mono);
    f_polynomials60(&mut poly, &mono);
    f_polynomials61(&mut poly, &mono);
    f_polynomials62(&mut poly, &mono);
    f_polynomials63(&mut poly, &mono);
    f_polynomials64(&mut poly, &mono);
    f_polynomials65(&mut poly, &mono);
    f_polynomials66(&mut poly, &mono);
    f_polynomials67(&mut poly, &mono);
    f_polynomials68(&mut poly, &mono);
    f_polynomials69(&mut poly, &mono);
    f_polynomials70(&mut poly, &mono);
    f_polynomials71(&mut poly, &mono);
    f_polynomials72(&mut poly, &mono);
    f_polynomials73(&mut poly, &mono);
    f_polynomials74(&mut poly, &mono);
    f_polynomials75(&mut poly, &mono);
    f_polynomials76(&mut poly, &mono);
    f_polynomials77(&mut poly, &mono);
    f_polynomials78(&mut poly, &mono);
    f_polynomials79(&mut poly, &mono);
    f_polynomials80(&mut poly, &mono);
    f_polynomials81(&mut poly, &mono);
    f_polynomials82(&mut poly, &mono);
    f_polynomials83(&mut poly, &mono);
    f_polynomials84(&mut poly, &mono);
    f_polynomials85(&mut poly, &mono);
    f_polynomials86(&mut poly, &mono);
    f_polynomials87(&mut poly, &mono);
    f_polynomials88(&mut poly, &mono);
    f_polynomials89(&mut poly, &mono);
    f_polynomials90(&mut poly, &mono);
    f_polynomials91(&mut poly, &mono);
    f_polynomials92(&mut poly, &mono);
    f_polynomials93(&mut poly, &mono);
    f_polynomials94(&mut poly, &mono);
    f_polynomials95(&mut poly, &mono);
    f_polynomials96(&mut poly, &mono);
    f_polynomials97(&mut poly, &mono);
    f_polynomials98(&mut poly, &mono);
    f_polynomials99(&mut poly, &mono);
    f_polynomials100(&mut poly, &mono);
    f_polynomials101(&mut poly, &mono);
    f_polynomials102(&mut poly, &mono);
    f_polynomials103(&mut poly, &mono);
    f_polynomials104(&mut poly, &mono);
    f_polynomials105(&mut poly, &mono);
    f_polynomials106(&mut poly, &mono);
    f_polynomials107(&mut poly, &mono);
    f_polynomials108(&mut poly, &mono);
    f_polynomials109(&mut poly, &mono);
    f_polynomials110(&mut poly, &mono);
    f_polynomials111(&mut poly, &mono);
    f_polynomials112(&mut poly, &mono);
    f_polynomials113(&mut poly, &mono);
    f_polynomials114(&mut poly, &mono);
    f_polynomials115(&mut poly, &mono);
    f_polynomials116(&mut poly, &mono);
    f_polynomials117(&mut poly, &mono);
    f_polynomials118(&mut poly, &mono);
    f_polynomials119(&mut poly, &mono);
    f_polynomials120(&mut poly, &mono);
    f_polynomials121(&mut poly, &mono);
    f_polynomials122(&mut poly, &mono);
    f_polynomials123(&mut poly, &mono);
    f_polynomials124(&mut poly, &mono);
    f_polynomials125(&mut poly, &mono);
    f_polynomials126(&mut poly, &mono);
    f_polynomials127(&mut poly, &mono);
    f_polynomials128(&mut poly, &mono);
    f_polynomials129(&mut poly, &mono);
    f_polynomials130(&mut poly, &mono);
    f_polynomials131(&mut poly, &mono);
    f_polynomials132(&mut poly, &mono);
    f_polynomials133(&mut poly, &mono);
    f_polynomials134(&mut poly, &mono);
    f_polynomials135(&mut poly, &mono);
    f_polynomials136(&mut poly, &mono);
    f_polynomials137(&mut poly, &mono);
    f_polynomials138(&mut poly, &mono);
    f_polynomials139(&mut poly, &mono);
    f_polynomials140(&mut poly, &mono);
    f_polynomials141(&mut poly, &mono);
    f_polynomials142(&mut poly, &mono);
    f_polynomials143(&mut poly, &mono);
    f_polynomials144(&mut poly, &mono);
    f_polynomials145(&mut poly, &mono);
    f_polynomials146(&mut poly, &mono);
    f_polynomials147(&mut poly, &mono);
    f_polynomials148(&mut poly, &mono);
    f_polynomials149(&mut poly, &mono);
    f_polynomials150(&mut poly, &mono);
    f_polynomials151(&mut poly, &mono);
    f_polynomials152(&mut poly, &mono);
    f_polynomials153(&mut poly, &mono);
    f_polynomials154(&mut poly, &mono);
    f_polynomials155(&mut poly, &mono);
    f_polynomials156(&mut poly, &mono);
    f_polynomials157(&mut poly, &mono);
    f_polynomials158(&mut poly, &mono);
    f_polynomials159(&mut poly, &mono);
    f_polynomials160(&mut poly, &mono);
    f_polynomials161(&mut poly, &mono);
    f_polynomials162(&mut poly, &mono);
    f_polynomials163(&mut poly, &mono);
    f_polynomials164(&mut poly, &mono);
    f_polynomials165(&mut poly, &mono);
    f_polynomials166(&mut poly, &mono);
    f_polynomials167(&mut poly, &mono);
    f_polynomials168(&mut poly, &mono);
    f_polynomials169(&mut poly, &mono);
    f_polynomials170(&mut poly, &mono);
    f_polynomials171(&mut poly, &mono);
    f_polynomials172(&mut poly, &mono);
    f_polynomials173(&mut poly, &mono);
    f_polynomials174(&mut poly, &mono);
    f_polynomials175(&mut poly, &mono);
    f_polynomials176(&mut poly, &mono);
    f_polynomials177(&mut poly, &mono);
    f_polynomials178(&mut poly, &mono);
    f_polynomials179(&mut poly, &mono);
    f_polynomials180(&mut poly, &mono);
    f_polynomials181(&mut poly, &mono);
    f_polynomials182(&mut poly, &mono);
    f_polynomials183(&mut poly, &mono);
    f_polynomials184(&mut poly, &mono);
    f_polynomials185(&mut poly, &mono);
    f_polynomials186(&mut poly, &mono);
    f_polynomials187(&mut poly, &mono);
    f_polynomials188(&mut poly, &mono);
    f_polynomials189(&mut poly, &mono);
    f_polynomials190(&mut poly, &mono);
    f_polynomials191(&mut poly, &mono);
    f_polynomials192(&mut poly, &mono);
    f_polynomials193(&mut poly, &mono);
    f_polynomials194(&mut poly, &mono);
    f_polynomials195(&mut poly, &mono);
    f_polynomials196(&mut poly, &mono);
    f_polynomials197(&mut poly, &mono);
    f_polynomials198(&mut poly, &mono);
    f_polynomials199(&mut poly, &mono);
    f_polynomials200(&mut poly, &mono);
    f_polynomials201(&mut poly, &mono);
    f_polynomials202(&mut poly, &mono);
    f_polynomials203(&mut poly, &mono);
    f_polynomials204(&mut poly, &mono);
    f_polynomials205(&mut poly, &mono);
    f_polynomials206(&mut poly, &mono);
    f_polynomials207(&mut poly, &mono);
    f_polynomials208(&mut poly, &mono);
    f_polynomials209(&mut poly, &mono);
    f_polynomials210(&mut poly, &mono);
    f_polynomials211(&mut poly, &mono);
    f_polynomials212(&mut poly, &mono);
    f_polynomials213(&mut poly, &mono);
    f_polynomials214(&mut poly, &mono);

    return poly;
}
