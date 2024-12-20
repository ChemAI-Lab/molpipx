#![allow(unused_variables)]
use super::monomials_1_1_1_1_1_4::*;


pub const N_POLYS: usize = 1001;

// File created from data/molecule_ABCDE/MOL_1_1_1_1_1_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4];
    poly[5] = mono[5];
    poly[6] = mono[6];
    poly[7] = mono[7];
    poly[8] = mono[8];
    poly[9] = mono[9];
    poly[10] = mono[10];
    poly[11] = poly[1] * poly[2];
    poly[12] = poly[1] * poly[3];
    poly[13] = poly[2] * poly[3];
    poly[14] = poly[1] * poly[4];
    poly[15] = poly[2] * poly[4];
    poly[16] = poly[3] * poly[4];
    poly[17] = poly[1] * poly[5];
    poly[18] = poly[2] * poly[5];
    poly[19] = poly[3] * poly[5];
    poly[20] = poly[4] * poly[5];
    poly[21] = poly[1] * poly[6];
    poly[22] = poly[2] * poly[6];
    poly[23] = poly[3] * poly[6];
    poly[24] = poly[4] * poly[6];
    poly[25] = poly[5] * poly[6];
    poly[26] = poly[1] * poly[7];
    poly[27] = poly[2] * poly[7];
    poly[28] = poly[3] * poly[7];
    poly[29] = poly[4] * poly[7];
    poly[30] = poly[5] * poly[7];
    poly[31] = poly[6] * poly[7];
    poly[32] = poly[1] * poly[8];
    poly[33] = poly[2] * poly[8];
    poly[34] = poly[3] * poly[8];
    poly[35] = poly[4] * poly[8];
    poly[36] = poly[5] * poly[8];
    poly[37] = poly[6] * poly[8];
    poly[38] = poly[7] * poly[8];
    poly[39] = poly[1] * poly[9];
    poly[40] = poly[2] * poly[9];
    poly[41] = poly[3] * poly[9];
    poly[42] = poly[4] * poly[9];
    poly[43] = poly[5] * poly[9];
    poly[44] = poly[6] * poly[9];
    poly[45] = poly[7] * poly[9];
    poly[46] = poly[8] * poly[9];
    poly[47] = poly[1] * poly[10];
    poly[48] = poly[2] * poly[10];
    poly[49] = poly[3] * poly[10];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[4] * poly[10];
    poly[51] = poly[5] * poly[10];
    poly[52] = poly[6] * poly[10];
    poly[53] = poly[7] * poly[10];
    poly[54] = poly[8] * poly[10];
    poly[55] = poly[9] * poly[10];
    poly[56] = poly[1] * poly[1];
    poly[57] = poly[2] * poly[2];
    poly[58] = poly[3] * poly[3];
    poly[59] = poly[4] * poly[4];
    poly[60] = poly[5] * poly[5];
    poly[61] = poly[6] * poly[6];
    poly[62] = poly[7] * poly[7];
    poly[63] = poly[8] * poly[8];
    poly[64] = poly[9] * poly[9];
    poly[65] = poly[10] * poly[10];
    poly[66] = poly[1] * poly[13];
    poly[67] = poly[1] * poly[15];
    poly[68] = poly[1] * poly[16];
    poly[69] = poly[2] * poly[16];
    poly[70] = poly[1] * poly[18];
    poly[71] = poly[1] * poly[19];
    poly[72] = poly[2] * poly[19];
    poly[73] = poly[1] * poly[20];
    poly[74] = poly[2] * poly[20];
    poly[75] = poly[3] * poly[20];
    poly[76] = poly[1] * poly[22];
    poly[77] = poly[1] * poly[23];
    poly[78] = poly[2] * poly[23];
    poly[79] = poly[1] * poly[24];
    poly[80] = poly[2] * poly[24];
    poly[81] = poly[3] * poly[24];
    poly[82] = poly[1] * poly[25];
    poly[83] = poly[2] * poly[25];
    poly[84] = poly[3] * poly[25];
    poly[85] = poly[4] * poly[25];
    poly[86] = poly[1] * poly[27];
    poly[87] = poly[1] * poly[28];
    poly[88] = poly[2] * poly[28];
    poly[89] = poly[1] * poly[29];
    poly[90] = poly[2] * poly[29];
    poly[91] = poly[3] * poly[29];
    poly[92] = poly[1] * poly[30];
    poly[93] = poly[2] * poly[30];
    poly[94] = poly[3] * poly[30];
    poly[95] = poly[4] * poly[30];
    poly[96] = poly[1] * poly[31];
    poly[97] = poly[2] * poly[31];
    poly[98] = poly[3] * poly[31];
    poly[99] = poly[4] * poly[31];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[5] * poly[31];
    poly[101] = poly[1] * poly[33];
    poly[102] = poly[1] * poly[34];
    poly[103] = poly[2] * poly[34];
    poly[104] = poly[1] * poly[35];
    poly[105] = poly[2] * poly[35];
    poly[106] = poly[3] * poly[35];
    poly[107] = poly[1] * poly[36];
    poly[108] = poly[2] * poly[36];
    poly[109] = poly[3] * poly[36];
    poly[110] = poly[4] * poly[36];
    poly[111] = poly[1] * poly[37];
    poly[112] = poly[2] * poly[37];
    poly[113] = poly[3] * poly[37];
    poly[114] = poly[4] * poly[37];
    poly[115] = poly[5] * poly[37];
    poly[116] = poly[1] * poly[38];
    poly[117] = poly[2] * poly[38];
    poly[118] = poly[3] * poly[38];
    poly[119] = poly[4] * poly[38];
    poly[120] = poly[5] * poly[38];
    poly[121] = poly[6] * poly[38];
    poly[122] = poly[1] * poly[40];
    poly[123] = poly[1] * poly[41];
    poly[124] = poly[2] * poly[41];
    poly[125] = poly[1] * poly[42];
    poly[126] = poly[2] * poly[42];
    poly[127] = poly[3] * poly[42];
    poly[128] = poly[1] * poly[43];
    poly[129] = poly[2] * poly[43];
    poly[130] = poly[3] * poly[43];
    poly[131] = poly[4] * poly[43];
    poly[132] = poly[1] * poly[44];
    poly[133] = poly[2] * poly[44];
    poly[134] = poly[3] * poly[44];
    poly[135] = poly[4] * poly[44];
    poly[136] = poly[5] * poly[44];
    poly[137] = poly[1] * poly[45];
    poly[138] = poly[2] * poly[45];
    poly[139] = poly[3] * poly[45];
    poly[140] = poly[4] * poly[45];
    poly[141] = poly[5] * poly[45];
    poly[142] = poly[6] * poly[45];
    poly[143] = poly[1] * poly[46];
    poly[144] = poly[2] * poly[46];
    poly[145] = poly[3] * poly[46];
    poly[146] = poly[4] * poly[46];
    poly[147] = poly[5] * poly[46];
    poly[148] = poly[6] * poly[46];
    poly[149] = poly[7] * poly[46];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[48];
    poly[151] = poly[1] * poly[49];
    poly[152] = poly[2] * poly[49];
    poly[153] = poly[1] * poly[50];
    poly[154] = poly[2] * poly[50];
    poly[155] = poly[3] * poly[50];
    poly[156] = poly[1] * poly[51];
    poly[157] = poly[2] * poly[51];
    poly[158] = poly[3] * poly[51];
    poly[159] = poly[4] * poly[51];
    poly[160] = poly[1] * poly[52];
    poly[161] = poly[2] * poly[52];
    poly[162] = poly[3] * poly[52];
    poly[163] = poly[4] * poly[52];
    poly[164] = poly[5] * poly[52];
    poly[165] = poly[1] * poly[53];
    poly[166] = poly[2] * poly[53];
    poly[167] = poly[3] * poly[53];
    poly[168] = poly[4] * poly[53];
    poly[169] = poly[5] * poly[53];
    poly[170] = poly[6] * poly[53];
    poly[171] = poly[1] * poly[54];
    poly[172] = poly[2] * poly[54];
    poly[173] = poly[3] * poly[54];
    poly[174] = poly[4] * poly[54];
    poly[175] = poly[5] * poly[54];
    poly[176] = poly[6] * poly[54];
    poly[177] = poly[7] * poly[54];
    poly[178] = poly[1] * poly[55];
    poly[179] = poly[2] * poly[55];
    poly[180] = poly[3] * poly[55];
    poly[181] = poly[4] * poly[55];
    poly[182] = poly[5] * poly[55];
    poly[183] = poly[6] * poly[55];
    poly[184] = poly[7] * poly[55];
    poly[185] = poly[8] * poly[55];
    poly[186] = poly[1] * poly[11];
    poly[187] = poly[1] * poly[57];
    poly[188] = poly[1] * poly[12];
    poly[189] = poly[2] * poly[13];
    poly[190] = poly[1] * poly[58];
    poly[191] = poly[2] * poly[58];
    poly[192] = poly[1] * poly[14];
    poly[193] = poly[2] * poly[15];
    poly[194] = poly[3] * poly[16];
    poly[195] = poly[1] * poly[59];
    poly[196] = poly[2] * poly[59];
    poly[197] = poly[3] * poly[59];
    poly[198] = poly[1] * poly[17];
    poly[199] = poly[2] * poly[18];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[3] * poly[19];
    poly[201] = poly[4] * poly[20];
    poly[202] = poly[1] * poly[60];
    poly[203] = poly[2] * poly[60];
    poly[204] = poly[3] * poly[60];
    poly[205] = poly[4] * poly[60];
    poly[206] = poly[1] * poly[21];
    poly[207] = poly[2] * poly[22];
    poly[208] = poly[3] * poly[23];
    poly[209] = poly[4] * poly[24];
    poly[210] = poly[5] * poly[25];
    poly[211] = poly[1] * poly[61];
    poly[212] = poly[2] * poly[61];
    poly[213] = poly[3] * poly[61];
    poly[214] = poly[4] * poly[61];
    poly[215] = poly[5] * poly[61];
    poly[216] = poly[1] * poly[26];
    poly[217] = poly[2] * poly[27];
    poly[218] = poly[3] * poly[28];
    poly[219] = poly[4] * poly[29];
    poly[220] = poly[5] * poly[30];
    poly[221] = poly[6] * poly[31];
    poly[222] = poly[1] * poly[62];
    poly[223] = poly[2] * poly[62];
    poly[224] = poly[3] * poly[62];
    poly[225] = poly[4] * poly[62];
    poly[226] = poly[5] * poly[62];
    poly[227] = poly[6] * poly[62];
    poly[228] = poly[1] * poly[32];
    poly[229] = poly[2] * poly[33];
    poly[230] = poly[3] * poly[34];
    poly[231] = poly[4] * poly[35];
    poly[232] = poly[5] * poly[36];
    poly[233] = poly[6] * poly[37];
    poly[234] = poly[7] * poly[38];
    poly[235] = poly[1] * poly[63];
    poly[236] = poly[2] * poly[63];
    poly[237] = poly[3] * poly[63];
    poly[238] = poly[4] * poly[63];
    poly[239] = poly[5] * poly[63];
    poly[240] = poly[6] * poly[63];
    poly[241] = poly[7] * poly[63];
    poly[242] = poly[1] * poly[39];
    poly[243] = poly[2] * poly[40];
    poly[244] = poly[3] * poly[41];
    poly[245] = poly[4] * poly[42];
    poly[246] = poly[5] * poly[43];
    poly[247] = poly[6] * poly[44];
    poly[248] = poly[7] * poly[45];
    poly[249] = poly[8] * poly[46];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[1] * poly[64];
    poly[251] = poly[2] * poly[64];
    poly[252] = poly[3] * poly[64];
    poly[253] = poly[4] * poly[64];
    poly[254] = poly[5] * poly[64];
    poly[255] = poly[6] * poly[64];
    poly[256] = poly[7] * poly[64];
    poly[257] = poly[8] * poly[64];
    poly[258] = poly[1] * poly[47];
    poly[259] = poly[2] * poly[48];
    poly[260] = poly[3] * poly[49];
    poly[261] = poly[4] * poly[50];
    poly[262] = poly[5] * poly[51];
    poly[263] = poly[6] * poly[52];
    poly[264] = poly[7] * poly[53];
    poly[265] = poly[8] * poly[54];
    poly[266] = poly[9] * poly[55];
    poly[267] = poly[1] * poly[65];
    poly[268] = poly[2] * poly[65];
    poly[269] = poly[3] * poly[65];
    poly[270] = poly[4] * poly[65];
    poly[271] = poly[5] * poly[65];
    poly[272] = poly[6] * poly[65];
    poly[273] = poly[7] * poly[65];
    poly[274] = poly[8] * poly[65];
    poly[275] = poly[9] * poly[65];
    poly[276] = poly[1] * poly[56];
    poly[277] = poly[2] * poly[57];
    poly[278] = poly[3] * poly[58];
    poly[279] = poly[4] * poly[59];
    poly[280] = poly[5] * poly[60];
    poly[281] = poly[6] * poly[61];
    poly[282] = poly[7] * poly[62];
    poly[283] = poly[8] * poly[63];
    poly[284] = poly[9] * poly[64];
    poly[285] = poly[10] * poly[65];
    poly[286] = poly[1] * poly[69];
    poly[287] = poly[1] * poly[72];
    poly[288] = poly[1] * poly[74];
    poly[289] = poly[1] * poly[75];
    poly[290] = poly[2] * poly[75];
    poly[291] = poly[1] * poly[78];
    poly[292] = poly[1] * poly[80];
    poly[293] = poly[1] * poly[81];
    poly[294] = poly[2] * poly[81];
    poly[295] = poly[1] * poly[83];
    poly[296] = poly[1] * poly[84];
    poly[297] = poly[2] * poly[84];
    poly[298] = poly[1] * poly[85];
    poly[299] = poly[2] * poly[85];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[3] * poly[85];
    poly[301] = poly[1] * poly[88];
    poly[302] = poly[1] * poly[90];
    poly[303] = poly[1] * poly[91];
    poly[304] = poly[2] * poly[91];
    poly[305] = poly[1] * poly[93];
    poly[306] = poly[1] * poly[94];
    poly[307] = poly[2] * poly[94];
    poly[308] = poly[1] * poly[95];
    poly[309] = poly[2] * poly[95];
    poly[310] = poly[3] * poly[95];
    poly[311] = poly[1] * poly[97];
    poly[312] = poly[1] * poly[98];
    poly[313] = poly[2] * poly[98];
    poly[314] = poly[1] * poly[99];
    poly[315] = poly[2] * poly[99];
    poly[316] = poly[3] * poly[99];
    poly[317] = poly[1] * poly[100];
    poly[318] = poly[2] * poly[100];
    poly[319] = poly[3] * poly[100];
    poly[320] = poly[4] * poly[100];
    poly[321] = poly[1] * poly[103];
    poly[322] = poly[1] * poly[105];
    poly[323] = poly[1] * poly[106];
    poly[324] = poly[2] * poly[106];
    poly[325] = poly[1] * poly[108];
    poly[326] = poly[1] * poly[109];
    poly[327] = poly[2] * poly[109];
    poly[328] = poly[1] * poly[110];
    poly[329] = poly[2] * poly[110];
    poly[330] = poly[3] * poly[110];
    poly[331] = poly[1] * poly[112];
    poly[332] = poly[1] * poly[113];
    poly[333] = poly[2] * poly[113];
    poly[334] = poly[1] * poly[114];
    poly[335] = poly[2] * poly[114];
    poly[336] = poly[3] * poly[114];
    poly[337] = poly[1] * poly[115];
    poly[338] = poly[2] * poly[115];
    poly[339] = poly[3] * poly[115];
    poly[340] = poly[4] * poly[115];
    poly[341] = poly[1] * poly[117];
    poly[342] = poly[1] * poly[118];
    poly[343] = poly[2] * poly[118];
    poly[344] = poly[1] * poly[119];
    poly[345] = poly[2] * poly[119];
    poly[346] = poly[3] * poly[119];
    poly[347] = poly[1] * poly[120];
    poly[348] = poly[2] * poly[120];
    poly[349] = poly[3] * poly[120];
}

#[inline(never)]
fn f_polynomials7(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[350] = poly[4] * poly[120];
    poly[351] = poly[1] * poly[121];
    poly[352] = poly[2] * poly[121];
    poly[353] = poly[3] * poly[121];
    poly[354] = poly[4] * poly[121];
    poly[355] = poly[5] * poly[121];
    poly[356] = poly[1] * poly[124];
    poly[357] = poly[1] * poly[126];
    poly[358] = poly[1] * poly[127];
    poly[359] = poly[2] * poly[127];
    poly[360] = poly[1] * poly[129];
    poly[361] = poly[1] * poly[130];
    poly[362] = poly[2] * poly[130];
    poly[363] = poly[1] * poly[131];
    poly[364] = poly[2] * poly[131];
    poly[365] = poly[3] * poly[131];
    poly[366] = poly[1] * poly[133];
    poly[367] = poly[1] * poly[134];
    poly[368] = poly[2] * poly[134];
    poly[369] = poly[1] * poly[135];
    poly[370] = poly[2] * poly[135];
    poly[371] = poly[3] * poly[135];
    poly[372] = poly[1] * poly[136];
    poly[373] = poly[2] * poly[136];
    poly[374] = poly[3] * poly[136];
    poly[375] = poly[4] * poly[136];
    poly[376] = poly[1] * poly[138];
    poly[377] = poly[1] * poly[139];
    poly[378] = poly[2] * poly[139];
    poly[379] = poly[1] * poly[140];
    poly[380] = poly[2] * poly[140];
    poly[381] = poly[3] * poly[140];
    poly[382] = poly[1] * poly[141];
    poly[383] = poly[2] * poly[141];
    poly[384] = poly[3] * poly[141];
    poly[385] = poly[4] * poly[141];
    poly[386] = poly[1] * poly[142];
    poly[387] = poly[2] * poly[142];
    poly[388] = poly[3] * poly[142];
    poly[389] = poly[4] * poly[142];
    poly[390] = poly[5] * poly[142];
    poly[391] = poly[1] * poly[144];
    poly[392] = poly[1] * poly[145];
    poly[393] = poly[2] * poly[145];
    poly[394] = poly[1] * poly[146];
    poly[395] = poly[2] * poly[146];
    poly[396] = poly[3] * poly[146];
    poly[397] = poly[1] * poly[147];
    poly[398] = poly[2] * poly[147];
    poly[399] = poly[3] * poly[147];
}

#[inline(never)]
fn f_polynomials8(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[400] = poly[4] * poly[147];
    poly[401] = poly[1] * poly[148];
    poly[402] = poly[2] * poly[148];
    poly[403] = poly[3] * poly[148];
    poly[404] = poly[4] * poly[148];
    poly[405] = poly[5] * poly[148];
    poly[406] = poly[1] * poly[149];
    poly[407] = poly[2] * poly[149];
    poly[408] = poly[3] * poly[149];
    poly[409] = poly[4] * poly[149];
    poly[410] = poly[5] * poly[149];
    poly[411] = poly[6] * poly[149];
    poly[412] = poly[1] * poly[152];
    poly[413] = poly[1] * poly[154];
    poly[414] = poly[1] * poly[155];
    poly[415] = poly[2] * poly[155];
    poly[416] = poly[1] * poly[157];
    poly[417] = poly[1] * poly[158];
    poly[418] = poly[2] * poly[158];
    poly[419] = poly[1] * poly[159];
    poly[420] = poly[2] * poly[159];
    poly[421] = poly[3] * poly[159];
    poly[422] = poly[1] * poly[161];
    poly[423] = poly[1] * poly[162];
    poly[424] = poly[2] * poly[162];
    poly[425] = poly[1] * poly[163];
    poly[426] = poly[2] * poly[163];
    poly[427] = poly[3] * poly[163];
    poly[428] = poly[1] * poly[164];
    poly[429] = poly[2] * poly[164];
    poly[430] = poly[3] * poly[164];
    poly[431] = poly[4] * poly[164];
    poly[432] = poly[1] * poly[166];
    poly[433] = poly[1] * poly[167];
    poly[434] = poly[2] * poly[167];
    poly[435] = poly[1] * poly[168];
    poly[436] = poly[2] * poly[168];
    poly[437] = poly[3] * poly[168];
    poly[438] = poly[1] * poly[169];
    poly[439] = poly[2] * poly[169];
    poly[440] = poly[3] * poly[169];
    poly[441] = poly[4] * poly[169];
    poly[442] = poly[1] * poly[170];
    poly[443] = poly[2] * poly[170];
    poly[444] = poly[3] * poly[170];
    poly[445] = poly[4] * poly[170];
    poly[446] = poly[5] * poly[170];
    poly[447] = poly[1] * poly[172];
    poly[448] = poly[1] * poly[173];
    poly[449] = poly[2] * poly[173];
}

#[inline(never)]
fn f_polynomials9(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[450] = poly[1] * poly[174];
    poly[451] = poly[2] * poly[174];
    poly[452] = poly[3] * poly[174];
    poly[453] = poly[1] * poly[175];
    poly[454] = poly[2] * poly[175];
    poly[455] = poly[3] * poly[175];
    poly[456] = poly[4] * poly[175];
    poly[457] = poly[1] * poly[176];
    poly[458] = poly[2] * poly[176];
    poly[459] = poly[3] * poly[176];
    poly[460] = poly[4] * poly[176];
    poly[461] = poly[5] * poly[176];
    poly[462] = poly[1] * poly[177];
    poly[463] = poly[2] * poly[177];
    poly[464] = poly[3] * poly[177];
    poly[465] = poly[4] * poly[177];
    poly[466] = poly[5] * poly[177];
    poly[467] = poly[6] * poly[177];
    poly[468] = poly[1] * poly[179];
    poly[469] = poly[1] * poly[180];
    poly[470] = poly[2] * poly[180];
    poly[471] = poly[1] * poly[181];
    poly[472] = poly[2] * poly[181];
    poly[473] = poly[3] * poly[181];
    poly[474] = poly[1] * poly[182];
    poly[475] = poly[2] * poly[182];
    poly[476] = poly[3] * poly[182];
    poly[477] = poly[4] * poly[182];
    poly[478] = poly[1] * poly[183];
    poly[479] = poly[2] * poly[183];
    poly[480] = poly[3] * poly[183];
    poly[481] = poly[4] * poly[183];
    poly[482] = poly[5] * poly[183];
    poly[483] = poly[1] * poly[184];
    poly[484] = poly[2] * poly[184];
    poly[485] = poly[3] * poly[184];
    poly[486] = poly[4] * poly[184];
    poly[487] = poly[5] * poly[184];
    poly[488] = poly[6] * poly[184];
    poly[489] = poly[1] * poly[185];
    poly[490] = poly[2] * poly[185];
    poly[491] = poly[3] * poly[185];
    poly[492] = poly[4] * poly[185];
    poly[493] = poly[5] * poly[185];
    poly[494] = poly[6] * poly[185];
    poly[495] = poly[7] * poly[185];
    poly[496] = poly[1] * poly[66];
    poly[497] = poly[1] * poly[189];
    poly[498] = poly[1] * poly[191];
    poly[499] = poly[1] * poly[67];
}

#[inline(never)]
fn f_polynomials10(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[500] = poly[1] * poly[193];
    poly[501] = poly[1] * poly[68];
    poly[502] = poly[2] * poly[69];
    poly[503] = poly[1] * poly[194];
    poly[504] = poly[2] * poly[194];
    poly[505] = poly[1] * poly[196];
    poly[506] = poly[1] * poly[197];
    poly[507] = poly[2] * poly[197];
    poly[508] = poly[1] * poly[70];
    poly[509] = poly[1] * poly[199];
    poly[510] = poly[1] * poly[71];
    poly[511] = poly[2] * poly[72];
    poly[512] = poly[1] * poly[200];
    poly[513] = poly[2] * poly[200];
    poly[514] = poly[1] * poly[73];
    poly[515] = poly[2] * poly[74];
    poly[516] = poly[3] * poly[75];
    poly[517] = poly[1] * poly[201];
    poly[518] = poly[2] * poly[201];
    poly[519] = poly[3] * poly[201];
    poly[520] = poly[1] * poly[203];
    poly[521] = poly[1] * poly[204];
    poly[522] = poly[2] * poly[204];
    poly[523] = poly[1] * poly[205];
    poly[524] = poly[2] * poly[205];
    poly[525] = poly[3] * poly[205];
    poly[526] = poly[1] * poly[76];
    poly[527] = poly[1] * poly[207];
    poly[528] = poly[1] * poly[77];
    poly[529] = poly[2] * poly[78];
    poly[530] = poly[1] * poly[208];
    poly[531] = poly[2] * poly[208];
    poly[532] = poly[1] * poly[79];
    poly[533] = poly[2] * poly[80];
    poly[534] = poly[3] * poly[81];
    poly[535] = poly[1] * poly[209];
    poly[536] = poly[2] * poly[209];
    poly[537] = poly[3] * poly[209];
    poly[538] = poly[1] * poly[82];
    poly[539] = poly[2] * poly[83];
    poly[540] = poly[3] * poly[84];
    poly[541] = poly[4] * poly[85];
    poly[542] = poly[1] * poly[210];
    poly[543] = poly[2] * poly[210];
    poly[544] = poly[3] * poly[210];
    poly[545] = poly[4] * poly[210];
    poly[546] = poly[1] * poly[212];
    poly[547] = poly[1] * poly[213];
    poly[548] = poly[2] * poly[213];
    poly[549] = poly[1] * poly[214];
}

#[inline(never)]
fn f_polynomials11(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[550] = poly[2] * poly[214];
    poly[551] = poly[3] * poly[214];
    poly[552] = poly[1] * poly[215];
    poly[553] = poly[2] * poly[215];
    poly[554] = poly[3] * poly[215];
    poly[555] = poly[4] * poly[215];
    poly[556] = poly[1] * poly[86];
    poly[557] = poly[1] * poly[217];
    poly[558] = poly[1] * poly[87];
    poly[559] = poly[2] * poly[88];
    poly[560] = poly[1] * poly[218];
    poly[561] = poly[2] * poly[218];
    poly[562] = poly[1] * poly[89];
    poly[563] = poly[2] * poly[90];
    poly[564] = poly[3] * poly[91];
    poly[565] = poly[1] * poly[219];
    poly[566] = poly[2] * poly[219];
    poly[567] = poly[3] * poly[219];
    poly[568] = poly[1] * poly[92];
    poly[569] = poly[2] * poly[93];
    poly[570] = poly[3] * poly[94];
    poly[571] = poly[4] * poly[95];
    poly[572] = poly[1] * poly[220];
    poly[573] = poly[2] * poly[220];
    poly[574] = poly[3] * poly[220];
    poly[575] = poly[4] * poly[220];
    poly[576] = poly[1] * poly[96];
    poly[577] = poly[2] * poly[97];
    poly[578] = poly[3] * poly[98];
    poly[579] = poly[4] * poly[99];
    poly[580] = poly[5] * poly[100];
    poly[581] = poly[1] * poly[221];
    poly[582] = poly[2] * poly[221];
    poly[583] = poly[3] * poly[221];
    poly[584] = poly[4] * poly[221];
    poly[585] = poly[5] * poly[221];
    poly[586] = poly[1] * poly[223];
    poly[587] = poly[1] * poly[224];
    poly[588] = poly[2] * poly[224];
    poly[589] = poly[1] * poly[225];
    poly[590] = poly[2] * poly[225];
    poly[591] = poly[3] * poly[225];
    poly[592] = poly[1] * poly[226];
    poly[593] = poly[2] * poly[226];
    poly[594] = poly[3] * poly[226];
    poly[595] = poly[4] * poly[226];
    poly[596] = poly[1] * poly[227];
    poly[597] = poly[2] * poly[227];
    poly[598] = poly[3] * poly[227];
    poly[599] = poly[4] * poly[227];
}

#[inline(never)]
fn f_polynomials12(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[600] = poly[5] * poly[227];
    poly[601] = poly[1] * poly[101];
    poly[602] = poly[1] * poly[229];
    poly[603] = poly[1] * poly[102];
    poly[604] = poly[2] * poly[103];
    poly[605] = poly[1] * poly[230];
    poly[606] = poly[2] * poly[230];
    poly[607] = poly[1] * poly[104];
    poly[608] = poly[2] * poly[105];
    poly[609] = poly[3] * poly[106];
    poly[610] = poly[1] * poly[231];
    poly[611] = poly[2] * poly[231];
    poly[612] = poly[3] * poly[231];
    poly[613] = poly[1] * poly[107];
    poly[614] = poly[2] * poly[108];
    poly[615] = poly[3] * poly[109];
    poly[616] = poly[4] * poly[110];
    poly[617] = poly[1] * poly[232];
    poly[618] = poly[2] * poly[232];
    poly[619] = poly[3] * poly[232];
    poly[620] = poly[4] * poly[232];
    poly[621] = poly[1] * poly[111];
    poly[622] = poly[2] * poly[112];
    poly[623] = poly[3] * poly[113];
    poly[624] = poly[4] * poly[114];
    poly[625] = poly[5] * poly[115];
    poly[626] = poly[1] * poly[233];
    poly[627] = poly[2] * poly[233];
    poly[628] = poly[3] * poly[233];
    poly[629] = poly[4] * poly[233];
    poly[630] = poly[5] * poly[233];
    poly[631] = poly[1] * poly[116];
    poly[632] = poly[2] * poly[117];
    poly[633] = poly[3] * poly[118];
    poly[634] = poly[4] * poly[119];
    poly[635] = poly[5] * poly[120];
    poly[636] = poly[6] * poly[121];
    poly[637] = poly[1] * poly[234];
    poly[638] = poly[2] * poly[234];
    poly[639] = poly[3] * poly[234];
    poly[640] = poly[4] * poly[234];
    poly[641] = poly[5] * poly[234];
    poly[642] = poly[6] * poly[234];
    poly[643] = poly[1] * poly[236];
    poly[644] = poly[1] * poly[237];
    poly[645] = poly[2] * poly[237];
    poly[646] = poly[1] * poly[238];
    poly[647] = poly[2] * poly[238];
    poly[648] = poly[3] * poly[238];
    poly[649] = poly[1] * poly[239];
}

#[inline(never)]
fn f_polynomials13(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[650] = poly[2] * poly[239];
    poly[651] = poly[3] * poly[239];
    poly[652] = poly[4] * poly[239];
    poly[653] = poly[1] * poly[240];
    poly[654] = poly[2] * poly[240];
    poly[655] = poly[3] * poly[240];
    poly[656] = poly[4] * poly[240];
    poly[657] = poly[5] * poly[240];
    poly[658] = poly[1] * poly[241];
    poly[659] = poly[2] * poly[241];
    poly[660] = poly[3] * poly[241];
    poly[661] = poly[4] * poly[241];
    poly[662] = poly[5] * poly[241];
    poly[663] = poly[6] * poly[241];
    poly[664] = poly[1] * poly[122];
    poly[665] = poly[1] * poly[243];
    poly[666] = poly[1] * poly[123];
    poly[667] = poly[2] * poly[124];
    poly[668] = poly[1] * poly[244];
    poly[669] = poly[2] * poly[244];
    poly[670] = poly[1] * poly[125];
    poly[671] = poly[2] * poly[126];
    poly[672] = poly[3] * poly[127];
    poly[673] = poly[1] * poly[245];
    poly[674] = poly[2] * poly[245];
    poly[675] = poly[3] * poly[245];
    poly[676] = poly[1] * poly[128];
    poly[677] = poly[2] * poly[129];
    poly[678] = poly[3] * poly[130];
    poly[679] = poly[4] * poly[131];
    poly[680] = poly[1] * poly[246];
    poly[681] = poly[2] * poly[246];
    poly[682] = poly[3] * poly[246];
    poly[683] = poly[4] * poly[246];
    poly[684] = poly[1] * poly[132];
    poly[685] = poly[2] * poly[133];
    poly[686] = poly[3] * poly[134];
    poly[687] = poly[4] * poly[135];
    poly[688] = poly[5] * poly[136];
    poly[689] = poly[1] * poly[247];
    poly[690] = poly[2] * poly[247];
    poly[691] = poly[3] * poly[247];
    poly[692] = poly[4] * poly[247];
    poly[693] = poly[5] * poly[247];
    poly[694] = poly[1] * poly[137];
    poly[695] = poly[2] * poly[138];
    poly[696] = poly[3] * poly[139];
    poly[697] = poly[4] * poly[140];
    poly[698] = poly[5] * poly[141];
    poly[699] = poly[6] * poly[142];
}

#[inline(never)]
fn f_polynomials14(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[700] = poly[1] * poly[248];
    poly[701] = poly[2] * poly[248];
    poly[702] = poly[3] * poly[248];
    poly[703] = poly[4] * poly[248];
    poly[704] = poly[5] * poly[248];
    poly[705] = poly[6] * poly[248];
    poly[706] = poly[1] * poly[143];
    poly[707] = poly[2] * poly[144];
    poly[708] = poly[3] * poly[145];
    poly[709] = poly[4] * poly[146];
    poly[710] = poly[5] * poly[147];
    poly[711] = poly[6] * poly[148];
    poly[712] = poly[7] * poly[149];
    poly[713] = poly[1] * poly[249];
    poly[714] = poly[2] * poly[249];
    poly[715] = poly[3] * poly[249];
    poly[716] = poly[4] * poly[249];
    poly[717] = poly[5] * poly[249];
    poly[718] = poly[6] * poly[249];
    poly[719] = poly[7] * poly[249];
    poly[720] = poly[1] * poly[251];
    poly[721] = poly[1] * poly[252];
    poly[722] = poly[2] * poly[252];
    poly[723] = poly[1] * poly[253];
    poly[724] = poly[2] * poly[253];
    poly[725] = poly[3] * poly[253];
    poly[726] = poly[1] * poly[254];
    poly[727] = poly[2] * poly[254];
    poly[728] = poly[3] * poly[254];
    poly[729] = poly[4] * poly[254];
    poly[730] = poly[1] * poly[255];
    poly[731] = poly[2] * poly[255];
    poly[732] = poly[3] * poly[255];
    poly[733] = poly[4] * poly[255];
    poly[734] = poly[5] * poly[255];
    poly[735] = poly[1] * poly[256];
    poly[736] = poly[2] * poly[256];
    poly[737] = poly[3] * poly[256];
    poly[738] = poly[4] * poly[256];
    poly[739] = poly[5] * poly[256];
    poly[740] = poly[6] * poly[256];
    poly[741] = poly[1] * poly[257];
    poly[742] = poly[2] * poly[257];
    poly[743] = poly[3] * poly[257];
    poly[744] = poly[4] * poly[257];
    poly[745] = poly[5] * poly[257];
    poly[746] = poly[6] * poly[257];
    poly[747] = poly[7] * poly[257];
    poly[748] = poly[1] * poly[150];
    poly[749] = poly[1] * poly[259];
}

#[inline(never)]
fn f_polynomials15(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[750] = poly[1] * poly[151];
    poly[751] = poly[2] * poly[152];
    poly[752] = poly[1] * poly[260];
    poly[753] = poly[2] * poly[260];
    poly[754] = poly[1] * poly[153];
    poly[755] = poly[2] * poly[154];
    poly[756] = poly[3] * poly[155];
    poly[757] = poly[1] * poly[261];
    poly[758] = poly[2] * poly[261];
    poly[759] = poly[3] * poly[261];
    poly[760] = poly[1] * poly[156];
    poly[761] = poly[2] * poly[157];
    poly[762] = poly[3] * poly[158];
    poly[763] = poly[4] * poly[159];
    poly[764] = poly[1] * poly[262];
    poly[765] = poly[2] * poly[262];
    poly[766] = poly[3] * poly[262];
    poly[767] = poly[4] * poly[262];
    poly[768] = poly[1] * poly[160];
    poly[769] = poly[2] * poly[161];
    poly[770] = poly[3] * poly[162];
    poly[771] = poly[4] * poly[163];
    poly[772] = poly[5] * poly[164];
    poly[773] = poly[1] * poly[263];
    poly[774] = poly[2] * poly[263];
    poly[775] = poly[3] * poly[263];
    poly[776] = poly[4] * poly[263];
    poly[777] = poly[5] * poly[263];
    poly[778] = poly[1] * poly[165];
    poly[779] = poly[2] * poly[166];
    poly[780] = poly[3] * poly[167];
    poly[781] = poly[4] * poly[168];
    poly[782] = poly[5] * poly[169];
    poly[783] = poly[6] * poly[170];
    poly[784] = poly[1] * poly[264];
    poly[785] = poly[2] * poly[264];
    poly[786] = poly[3] * poly[264];
    poly[787] = poly[4] * poly[264];
    poly[788] = poly[5] * poly[264];
    poly[789] = poly[6] * poly[264];
    poly[790] = poly[1] * poly[171];
    poly[791] = poly[2] * poly[172];
    poly[792] = poly[3] * poly[173];
    poly[793] = poly[4] * poly[174];
    poly[794] = poly[5] * poly[175];
    poly[795] = poly[6] * poly[176];
    poly[796] = poly[7] * poly[177];
    poly[797] = poly[1] * poly[265];
    poly[798] = poly[2] * poly[265];
    poly[799] = poly[3] * poly[265];
}

#[inline(never)]
fn f_polynomials16(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[800] = poly[4] * poly[265];
    poly[801] = poly[5] * poly[265];
    poly[802] = poly[6] * poly[265];
    poly[803] = poly[7] * poly[265];
    poly[804] = poly[1] * poly[178];
    poly[805] = poly[2] * poly[179];
    poly[806] = poly[3] * poly[180];
    poly[807] = poly[4] * poly[181];
    poly[808] = poly[5] * poly[182];
    poly[809] = poly[6] * poly[183];
    poly[810] = poly[7] * poly[184];
    poly[811] = poly[8] * poly[185];
    poly[812] = poly[1] * poly[266];
    poly[813] = poly[2] * poly[266];
    poly[814] = poly[3] * poly[266];
    poly[815] = poly[4] * poly[266];
    poly[816] = poly[5] * poly[266];
    poly[817] = poly[6] * poly[266];
    poly[818] = poly[7] * poly[266];
    poly[819] = poly[8] * poly[266];
    poly[820] = poly[1] * poly[268];
    poly[821] = poly[1] * poly[269];
    poly[822] = poly[2] * poly[269];
    poly[823] = poly[1] * poly[270];
    poly[824] = poly[2] * poly[270];
    poly[825] = poly[3] * poly[270];
    poly[826] = poly[1] * poly[271];
    poly[827] = poly[2] * poly[271];
    poly[828] = poly[3] * poly[271];
    poly[829] = poly[4] * poly[271];
    poly[830] = poly[1] * poly[272];
    poly[831] = poly[2] * poly[272];
    poly[832] = poly[3] * poly[272];
    poly[833] = poly[4] * poly[272];
    poly[834] = poly[5] * poly[272];
    poly[835] = poly[1] * poly[273];
    poly[836] = poly[2] * poly[273];
    poly[837] = poly[3] * poly[273];
    poly[838] = poly[4] * poly[273];
    poly[839] = poly[5] * poly[273];
    poly[840] = poly[6] * poly[273];
    poly[841] = poly[1] * poly[274];
    poly[842] = poly[2] * poly[274];
    poly[843] = poly[3] * poly[274];
    poly[844] = poly[4] * poly[274];
    poly[845] = poly[5] * poly[274];
    poly[846] = poly[6] * poly[274];
    poly[847] = poly[7] * poly[274];
    poly[848] = poly[1] * poly[275];
    poly[849] = poly[2] * poly[275];
}

#[inline(never)]
fn f_polynomials17(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[850] = poly[3] * poly[275];
    poly[851] = poly[4] * poly[275];
    poly[852] = poly[5] * poly[275];
    poly[853] = poly[6] * poly[275];
    poly[854] = poly[7] * poly[275];
    poly[855] = poly[8] * poly[275];
    poly[856] = poly[1] * poly[186];
    poly[857] = poly[1] * poly[187];
    poly[858] = poly[1] * poly[277];
    poly[859] = poly[1] * poly[188];
    poly[860] = poly[2] * poly[189];
    poly[861] = poly[1] * poly[190];
    poly[862] = poly[2] * poly[191];
    poly[863] = poly[1] * poly[278];
    poly[864] = poly[2] * poly[278];
    poly[865] = poly[1] * poly[192];
    poly[866] = poly[2] * poly[193];
    poly[867] = poly[3] * poly[194];
    poly[868] = poly[1] * poly[195];
    poly[869] = poly[2] * poly[196];
    poly[870] = poly[3] * poly[197];
    poly[871] = poly[1] * poly[279];
    poly[872] = poly[2] * poly[279];
    poly[873] = poly[3] * poly[279];
    poly[874] = poly[1] * poly[198];
    poly[875] = poly[2] * poly[199];
    poly[876] = poly[3] * poly[200];
    poly[877] = poly[4] * poly[201];
    poly[878] = poly[1] * poly[202];
    poly[879] = poly[2] * poly[203];
    poly[880] = poly[3] * poly[204];
    poly[881] = poly[4] * poly[205];
    poly[882] = poly[1] * poly[280];
    poly[883] = poly[2] * poly[280];
    poly[884] = poly[3] * poly[280];
    poly[885] = poly[4] * poly[280];
    poly[886] = poly[1] * poly[206];
    poly[887] = poly[2] * poly[207];
    poly[888] = poly[3] * poly[208];
    poly[889] = poly[4] * poly[209];
    poly[890] = poly[5] * poly[210];
    poly[891] = poly[1] * poly[211];
    poly[892] = poly[2] * poly[212];
    poly[893] = poly[3] * poly[213];
    poly[894] = poly[4] * poly[214];
    poly[895] = poly[5] * poly[215];
    poly[896] = poly[1] * poly[281];
    poly[897] = poly[2] * poly[281];
    poly[898] = poly[3] * poly[281];
    poly[899] = poly[4] * poly[281];
}

#[inline(never)]
fn f_polynomials18(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[900] = poly[5] * poly[281];
    poly[901] = poly[1] * poly[216];
    poly[902] = poly[2] * poly[217];
    poly[903] = poly[3] * poly[218];
    poly[904] = poly[4] * poly[219];
    poly[905] = poly[5] * poly[220];
    poly[906] = poly[6] * poly[221];
    poly[907] = poly[1] * poly[222];
    poly[908] = poly[2] * poly[223];
    poly[909] = poly[3] * poly[224];
    poly[910] = poly[4] * poly[225];
    poly[911] = poly[5] * poly[226];
    poly[912] = poly[6] * poly[227];
    poly[913] = poly[1] * poly[282];
    poly[914] = poly[2] * poly[282];
    poly[915] = poly[3] * poly[282];
    poly[916] = poly[4] * poly[282];
    poly[917] = poly[5] * poly[282];
    poly[918] = poly[6] * poly[282];
    poly[919] = poly[1] * poly[228];
    poly[920] = poly[2] * poly[229];
    poly[921] = poly[3] * poly[230];
    poly[922] = poly[4] * poly[231];
    poly[923] = poly[5] * poly[232];
    poly[924] = poly[6] * poly[233];
    poly[925] = poly[7] * poly[234];
    poly[926] = poly[1] * poly[235];
    poly[927] = poly[2] * poly[236];
    poly[928] = poly[3] * poly[237];
    poly[929] = poly[4] * poly[238];
    poly[930] = poly[5] * poly[239];
    poly[931] = poly[6] * poly[240];
    poly[932] = poly[7] * poly[241];
    poly[933] = poly[1] * poly[283];
    poly[934] = poly[2] * poly[283];
    poly[935] = poly[3] * poly[283];
    poly[936] = poly[4] * poly[283];
    poly[937] = poly[5] * poly[283];
    poly[938] = poly[6] * poly[283];
    poly[939] = poly[7] * poly[283];
    poly[940] = poly[1] * poly[242];
    poly[941] = poly[2] * poly[243];
    poly[942] = poly[3] * poly[244];
    poly[943] = poly[4] * poly[245];
    poly[944] = poly[5] * poly[246];
    poly[945] = poly[6] * poly[247];
    poly[946] = poly[7] * poly[248];
    poly[947] = poly[8] * poly[249];
    poly[948] = poly[1] * poly[250];
    poly[949] = poly[2] * poly[251];
}

#[inline(never)]
fn f_polynomials19(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[950] = poly[3] * poly[252];
    poly[951] = poly[4] * poly[253];
    poly[952] = poly[5] * poly[254];
    poly[953] = poly[6] * poly[255];
    poly[954] = poly[7] * poly[256];
    poly[955] = poly[8] * poly[257];
    poly[956] = poly[1] * poly[284];
    poly[957] = poly[2] * poly[284];
    poly[958] = poly[3] * poly[284];
    poly[959] = poly[4] * poly[284];
    poly[960] = poly[5] * poly[284];
    poly[961] = poly[6] * poly[284];
    poly[962] = poly[7] * poly[284];
    poly[963] = poly[8] * poly[284];
    poly[964] = poly[1] * poly[258];
    poly[965] = poly[2] * poly[259];
    poly[966] = poly[3] * poly[260];
    poly[967] = poly[4] * poly[261];
    poly[968] = poly[5] * poly[262];
    poly[969] = poly[6] * poly[263];
    poly[970] = poly[7] * poly[264];
    poly[971] = poly[8] * poly[265];
    poly[972] = poly[9] * poly[266];
    poly[973] = poly[1] * poly[267];
    poly[974] = poly[2] * poly[268];
    poly[975] = poly[3] * poly[269];
    poly[976] = poly[4] * poly[270];
    poly[977] = poly[5] * poly[271];
    poly[978] = poly[6] * poly[272];
    poly[979] = poly[7] * poly[273];
    poly[980] = poly[8] * poly[274];
    poly[981] = poly[9] * poly[275];
    poly[982] = poly[1] * poly[285];
    poly[983] = poly[2] * poly[285];
    poly[984] = poly[3] * poly[285];
    poly[985] = poly[4] * poly[285];
    poly[986] = poly[5] * poly[285];
    poly[987] = poly[6] * poly[285];
    poly[988] = poly[7] * poly[285];
    poly[989] = poly[8] * poly[285];
    poly[990] = poly[9] * poly[285];
    poly[991] = poly[1] * poly[276];
    poly[992] = poly[2] * poly[277];
    poly[993] = poly[3] * poly[278];
    poly[994] = poly[4] * poly[279];
    poly[995] = poly[5] * poly[280];
    poly[996] = poly[6] * poly[281];
    poly[997] = poly[7] * poly[282];
    poly[998] = poly[8] * poly[283];
    poly[999] = poly[9] * poly[284];
}

#[inline(never)]
fn f_polynomials20(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[1000] = poly[10] * poly[285];
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
    f_polynomials12(&mut poly, &mono);
    f_polynomials13(&mut poly, &mono);
    f_polynomials14(&mut poly, &mono);
    f_polynomials15(&mut poly, &mono);
    f_polynomials16(&mut poly, &mono);
    f_polynomials17(&mut poly, &mono);
    f_polynomials18(&mut poly, &mono);
    f_polynomials19(&mut poly, &mono);
    f_polynomials20(&mut poly, &mono);

    return poly;
}
