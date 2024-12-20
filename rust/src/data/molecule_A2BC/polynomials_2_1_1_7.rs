#![allow(unused_variables)]
use super::monomials_2_1_1_7::*;


pub const N_POLYS: usize = 918;

// File created from data/molecule_A2BC/MOL_2_1_1_7.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3];
    poly[3] = mono[4] + mono[5];
    poly[4] = mono[6];
    poly[5] = poly[1] * poly[2];
    poly[6] = mono[7];
    poly[7] = poly[1] * poly[3];
    poly[8] = mono[8] + mono[9];
    poly[9] = mono[10];
    poly[10] = poly[2] * poly[3] - poly[8];
    poly[11] = poly[1] * poly[4];
    poly[12] = poly[4] * poly[2];
    poly[13] = poly[4] * poly[3];
    poly[14] = poly[1] * poly[1];
    poly[15] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[16] = poly[3] * poly[3] - poly[9] - poly[9];
    poly[17] = poly[4] * poly[4];
    poly[18] = poly[1] * poly[6];
    poly[19] = poly[1] * poly[8];
    poly[20] = poly[1] * poly[9];
    poly[21] = poly[1] * poly[10];
    poly[22] = poly[6] * poly[3];
    poly[23] = poly[9] * poly[2];
    poly[24] = poly[1] * poly[12];
    poly[25] = poly[4] * poly[6];
    poly[26] = poly[1] * poly[13];
    poly[27] = poly[4] * poly[8];
    poly[28] = poly[4] * poly[9];
    poly[29] = poly[4] * poly[10];
    poly[30] = poly[1] * poly[5];
    poly[31] = poly[1] * poly[15];
    poly[32] = poly[6] * poly[2];
    poly[33] = poly[1] * poly[7];
    poly[34] = poly[2] * poly[8] - poly[22];
    poly[35] = poly[2] * poly[10] - poly[22];
    poly[36] = poly[1] * poly[16];
    poly[37] = poly[3] * poly[8] - poly[23];
    poly[38] = poly[9] * poly[3];
    poly[39] = poly[2] * poly[16] - poly[37];
    poly[40] = poly[1] * poly[11];
    poly[41] = poly[4] * poly[15];
    poly[42] = poly[4] * poly[16];
    poly[43] = poly[1] * poly[17];
    poly[44] = poly[4] * poly[12];
    poly[45] = poly[4] * poly[13];
    poly[46] = poly[1] * poly[14];
    poly[47] = poly[2] * poly[15] - poly[32];
    poly[48] = poly[3] * poly[16] - poly[38];
    poly[49] = poly[4] * poly[17];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[22];
    poly[51] = poly[1] * poly[23];
    poly[52] = poly[6] * poly[9];
    poly[53] = poly[1] * poly[25];
    poly[54] = poly[1] * poly[27];
    poly[55] = poly[1] * poly[28];
    poly[56] = poly[1] * poly[29];
    poly[57] = poly[4] * poly[22];
    poly[58] = poly[4] * poly[23];
    poly[59] = poly[1] * poly[18];
    poly[60] = poly[1] * poly[32];
    poly[61] = poly[1] * poly[19];
    poly[62] = poly[1] * poly[34];
    poly[63] = poly[1] * poly[20];
    poly[64] = poly[1] * poly[21];
    poly[65] = poly[6] * poly[8];
    poly[66] = poly[1] * poly[35];
    poly[67] = poly[6] * poly[10];
    poly[68] = poly[9] * poly[15];
    poly[69] = poly[1] * poly[37];
    poly[70] = poly[1] * poly[38];
    poly[71] = poly[9] * poly[8];
    poly[72] = poly[1] * poly[39];
    poly[73] = poly[6] * poly[16];
    poly[74] = poly[9] * poly[10];
    poly[75] = poly[1] * poly[24];
    poly[76] = poly[1] * poly[41];
    poly[77] = poly[4] * poly[32];
    poly[78] = poly[1] * poly[26];
    poly[79] = poly[4] * poly[34];
    poly[80] = poly[4] * poly[35];
    poly[81] = poly[1] * poly[42];
    poly[82] = poly[4] * poly[37];
    poly[83] = poly[4] * poly[38];
    poly[84] = poly[4] * poly[39];
    poly[85] = poly[1] * poly[44];
    poly[86] = poly[4] * poly[25];
    poly[87] = poly[1] * poly[45];
    poly[88] = poly[4] * poly[27];
    poly[89] = poly[4] * poly[28];
    poly[90] = poly[4] * poly[29];
    poly[91] = poly[1] * poly[30];
    poly[92] = poly[1] * poly[31];
    poly[93] = poly[6] * poly[6];
    poly[94] = poly[1] * poly[47];
    poly[95] = poly[6] * poly[15];
    poly[96] = poly[1] * poly[33];
    poly[97] = poly[2] * poly[34] - poly[65];
    poly[98] = poly[2] * poly[35] - poly[67];
    poly[99] = poly[1] * poly[36];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[2] * poly[37] - poly[73];
    poly[101] = poly[9] * poly[9];
    poly[102] = poly[2] * poly[39] - poly[73];
    poly[103] = poly[1] * poly[48];
    poly[104] = poly[3] * poly[37] - poly[71];
    poly[105] = poly[9] * poly[16];
    poly[106] = poly[2] * poly[48] - poly[104];
    poly[107] = poly[1] * poly[40];
    poly[108] = poly[4] * poly[47];
    poly[109] = poly[4] * poly[48];
    poly[110] = poly[1] * poly[43];
    poly[111] = poly[4] * poly[41];
    poly[112] = poly[4] * poly[42];
    poly[113] = poly[1] * poly[49];
    poly[114] = poly[4] * poly[44];
    poly[115] = poly[4] * poly[45];
    poly[116] = poly[1] * poly[46];
    poly[117] = poly[2] * poly[47] - poly[95];
    poly[118] = poly[3] * poly[48] - poly[105];
    poly[119] = poly[4] * poly[49];
    poly[120] = poly[1] * poly[52];
    poly[121] = poly[1] * poly[57];
    poly[122] = poly[1] * poly[58];
    poly[123] = poly[4] * poly[52];
    poly[124] = poly[1] * poly[50];
    poly[125] = poly[1] * poly[65];
    poly[126] = poly[1] * poly[51];
    poly[127] = poly[1] * poly[67];
    poly[128] = poly[1] * poly[68];
    poly[129] = poly[6] * poly[23];
    poly[130] = poly[1] * poly[71];
    poly[131] = poly[1] * poly[73];
    poly[132] = poly[1] * poly[74];
    poly[133] = poly[6] * poly[38];
    poly[134] = poly[1] * poly[53];
    poly[135] = poly[1] * poly[77];
    poly[136] = poly[1] * poly[54];
    poly[137] = poly[1] * poly[79];
    poly[138] = poly[1] * poly[55];
    poly[139] = poly[1] * poly[56];
    poly[140] = poly[4] * poly[65];
    poly[141] = poly[1] * poly[80];
    poly[142] = poly[4] * poly[67];
    poly[143] = poly[4] * poly[68];
    poly[144] = poly[1] * poly[82];
    poly[145] = poly[1] * poly[83];
    poly[146] = poly[4] * poly[71];
    poly[147] = poly[1] * poly[84];
    poly[148] = poly[4] * poly[73];
    poly[149] = poly[4] * poly[74];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[86];
    poly[151] = poly[1] * poly[88];
    poly[152] = poly[1] * poly[89];
    poly[153] = poly[1] * poly[90];
    poly[154] = poly[4] * poly[57];
    poly[155] = poly[4] * poly[58];
    poly[156] = poly[1] * poly[59];
    poly[157] = poly[1] * poly[60];
    poly[158] = poly[1] * poly[93];
    poly[159] = poly[1] * poly[95];
    poly[160] = poly[1] * poly[61];
    poly[161] = poly[1] * poly[62];
    poly[162] = poly[1] * poly[97];
    poly[163] = poly[1] * poly[63];
    poly[164] = poly[1] * poly[64];
    poly[165] = poly[6] * poly[34];
    poly[166] = poly[1] * poly[66];
    poly[167] = poly[6] * poly[22];
    poly[168] = poly[1] * poly[98];
    poly[169] = poly[6] * poly[35];
    poly[170] = poly[9] * poly[47];
    poly[171] = poly[1] * poly[69];
    poly[172] = poly[1] * poly[100];
    poly[173] = poly[1] * poly[70];
    poly[174] = poly[9] * poly[34];
    poly[175] = poly[1] * poly[101];
    poly[176] = poly[1] * poly[72];
    poly[177] = poly[6] * poly[37];
    poly[178] = poly[9] * poly[23];
    poly[179] = poly[1] * poly[102];
    poly[180] = poly[6] * poly[39];
    poly[181] = poly[9] * poly[35];
    poly[182] = poly[1] * poly[104];
    poly[183] = poly[1] * poly[105];
    poly[184] = poly[9] * poly[37];
    poly[185] = poly[1] * poly[106];
    poly[186] = poly[6] * poly[48];
    poly[187] = poly[9] * poly[39];
    poly[188] = poly[1] * poly[75];
    poly[189] = poly[1] * poly[76];
    poly[190] = poly[4] * poly[93];
    poly[191] = poly[1] * poly[108];
    poly[192] = poly[4] * poly[95];
    poly[193] = poly[1] * poly[78];
    poly[194] = poly[4] * poly[97];
    poly[195] = poly[4] * poly[98];
    poly[196] = poly[1] * poly[81];
    poly[197] = poly[4] * poly[100];
    poly[198] = poly[4] * poly[101];
    poly[199] = poly[4] * poly[102];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[1] * poly[109];
    poly[201] = poly[4] * poly[104];
    poly[202] = poly[4] * poly[105];
    poly[203] = poly[4] * poly[106];
    poly[204] = poly[1] * poly[85];
    poly[205] = poly[1] * poly[111];
    poly[206] = poly[4] * poly[77];
    poly[207] = poly[1] * poly[87];
    poly[208] = poly[4] * poly[79];
    poly[209] = poly[4] * poly[80];
    poly[210] = poly[1] * poly[112];
    poly[211] = poly[4] * poly[82];
    poly[212] = poly[4] * poly[83];
    poly[213] = poly[4] * poly[84];
    poly[214] = poly[1] * poly[114];
    poly[215] = poly[4] * poly[86];
    poly[216] = poly[1] * poly[115];
    poly[217] = poly[4] * poly[88];
    poly[218] = poly[4] * poly[89];
    poly[219] = poly[4] * poly[90];
    poly[220] = poly[1] * poly[91];
    poly[221] = poly[1] * poly[92];
    poly[222] = poly[1] * poly[94];
    poly[223] = poly[6] * poly[32];
    poly[224] = poly[1] * poly[117];
    poly[225] = poly[6] * poly[47];
    poly[226] = poly[1] * poly[96];
    poly[227] = poly[2] * poly[97] - poly[165];
    poly[228] = poly[2] * poly[98] - poly[169];
    poly[229] = poly[1] * poly[99];
    poly[230] = poly[2] * poly[100] - poly[177];
    poly[231] = poly[2] * poly[102] - poly[180];
    poly[232] = poly[1] * poly[103];
    poly[233] = poly[2] * poly[104] - poly[186];
    poly[234] = poly[9] * poly[38];
    poly[235] = poly[2] * poly[106] - poly[186];
    poly[236] = poly[1] * poly[118];
    poly[237] = poly[3] * poly[104] - poly[184];
    poly[238] = poly[9] * poly[48];
    poly[239] = poly[2] * poly[118] - poly[237];
    poly[240] = poly[1] * poly[107];
    poly[241] = poly[4] * poly[117];
    poly[242] = poly[4] * poly[118];
    poly[243] = poly[1] * poly[110];
    poly[244] = poly[4] * poly[108];
    poly[245] = poly[4] * poly[109];
    poly[246] = poly[1] * poly[113];
    poly[247] = poly[4] * poly[111];
    poly[248] = poly[4] * poly[112];
    poly[249] = poly[1] * poly[119];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[4] * poly[114];
    poly[251] = poly[4] * poly[115];
    poly[252] = poly[1] * poly[116];
    poly[253] = poly[2] * poly[117] - poly[225];
    poly[254] = poly[3] * poly[118] - poly[238];
    poly[255] = poly[4] * poly[119];
    poly[256] = poly[1] * poly[123];
    poly[257] = poly[1] * poly[120];
    poly[258] = poly[1] * poly[129];
    poly[259] = poly[1] * poly[133];
    poly[260] = poly[1] * poly[121];
    poly[261] = poly[1] * poly[140];
    poly[262] = poly[1] * poly[122];
    poly[263] = poly[1] * poly[142];
    poly[264] = poly[1] * poly[143];
    poly[265] = poly[4] * poly[129];
    poly[266] = poly[1] * poly[146];
    poly[267] = poly[1] * poly[148];
    poly[268] = poly[1] * poly[149];
    poly[269] = poly[4] * poly[133];
    poly[270] = poly[1] * poly[154];
    poly[271] = poly[1] * poly[155];
    poly[272] = poly[4] * poly[123];
    poly[273] = poly[1] * poly[124];
    poly[274] = poly[1] * poly[125];
    poly[275] = poly[1] * poly[165];
    poly[276] = poly[1] * poly[126];
    poly[277] = poly[1] * poly[127];
    poly[278] = poly[1] * poly[167];
    poly[279] = poly[1] * poly[128];
    poly[280] = poly[6] * poly[52];
    poly[281] = poly[1] * poly[169];
    poly[282] = poly[1] * poly[170];
    poly[283] = poly[6] * poly[68];
    poly[284] = poly[1] * poly[130];
    poly[285] = poly[1] * poly[174];
    poly[286] = poly[1] * poly[131];
    poly[287] = poly[1] * poly[177];
    poly[288] = poly[1] * poly[132];
    poly[289] = poly[6] * poly[71];
    poly[290] = poly[1] * poly[178];
    poly[291] = poly[6] * poly[101];
    poly[292] = poly[1] * poly[180];
    poly[293] = poly[1] * poly[181];
    poly[294] = poly[6] * poly[74];
    poly[295] = poly[1] * poly[184];
    poly[296] = poly[1] * poly[186];
    poly[297] = poly[1] * poly[187];
    poly[298] = poly[6] * poly[105];
    poly[299] = poly[1] * poly[134];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[1] * poly[135];
    poly[301] = poly[1] * poly[190];
    poly[302] = poly[1] * poly[192];
    poly[303] = poly[1] * poly[136];
    poly[304] = poly[1] * poly[137];
    poly[305] = poly[1] * poly[194];
    poly[306] = poly[1] * poly[138];
    poly[307] = poly[1] * poly[139];
    poly[308] = poly[4] * poly[165];
    poly[309] = poly[1] * poly[141];
    poly[310] = poly[4] * poly[167];
    poly[311] = poly[1] * poly[195];
    poly[312] = poly[4] * poly[169];
    poly[313] = poly[4] * poly[170];
    poly[314] = poly[1] * poly[144];
    poly[315] = poly[1] * poly[197];
    poly[316] = poly[1] * poly[145];
    poly[317] = poly[4] * poly[174];
    poly[318] = poly[1] * poly[198];
    poly[319] = poly[1] * poly[147];
    poly[320] = poly[4] * poly[177];
    poly[321] = poly[4] * poly[178];
    poly[322] = poly[1] * poly[199];
    poly[323] = poly[4] * poly[180];
    poly[324] = poly[4] * poly[181];
    poly[325] = poly[1] * poly[201];
    poly[326] = poly[1] * poly[202];
    poly[327] = poly[4] * poly[184];
    poly[328] = poly[1] * poly[203];
    poly[329] = poly[4] * poly[186];
    poly[330] = poly[4] * poly[187];
    poly[331] = poly[1] * poly[150];
    poly[332] = poly[1] * poly[206];
    poly[333] = poly[1] * poly[151];
    poly[334] = poly[1] * poly[208];
    poly[335] = poly[1] * poly[152];
    poly[336] = poly[1] * poly[153];
    poly[337] = poly[4] * poly[140];
    poly[338] = poly[1] * poly[209];
    poly[339] = poly[4] * poly[142];
    poly[340] = poly[4] * poly[143];
    poly[341] = poly[1] * poly[211];
    poly[342] = poly[1] * poly[212];
    poly[343] = poly[4] * poly[146];
    poly[344] = poly[1] * poly[213];
    poly[345] = poly[4] * poly[148];
    poly[346] = poly[4] * poly[149];
    poly[347] = poly[1] * poly[215];
    poly[348] = poly[1] * poly[217];
    poly[349] = poly[1] * poly[218];
}

#[inline(never)]
fn f_polynomials7(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[350] = poly[1] * poly[219];
    poly[351] = poly[4] * poly[154];
    poly[352] = poly[4] * poly[155];
    poly[353] = poly[1] * poly[156];
    poly[354] = poly[1] * poly[157];
    poly[355] = poly[1] * poly[158];
    poly[356] = poly[1] * poly[159];
    poly[357] = poly[1] * poly[223];
    poly[358] = poly[1] * poly[225];
    poly[359] = poly[1] * poly[160];
    poly[360] = poly[1] * poly[161];
    poly[361] = poly[1] * poly[162];
    poly[362] = poly[1] * poly[227];
    poly[363] = poly[1] * poly[163];
    poly[364] = poly[1] * poly[164];
    poly[365] = poly[6] * poly[97];
    poly[366] = poly[1] * poly[166];
    poly[367] = poly[6] * poly[65];
    poly[368] = poly[1] * poly[168];
    poly[369] = poly[6] * poly[67];
    poly[370] = poly[1] * poly[228];
    poly[371] = poly[6] * poly[98];
    poly[372] = poly[9] * poly[117];
    poly[373] = poly[1] * poly[171];
    poly[374] = poly[1] * poly[172];
    poly[375] = poly[1] * poly[230];
    poly[376] = poly[1] * poly[173];
    poly[377] = poly[9] * poly[97];
    poly[378] = poly[1] * poly[175];
    poly[379] = poly[1] * poly[176];
    poly[380] = poly[6] * poly[100];
    poly[381] = poly[1] * poly[179];
    poly[382] = poly[6] * poly[73];
    poly[383] = poly[9] * poly[68];
    poly[384] = poly[1] * poly[231];
    poly[385] = poly[6] * poly[102];
    poly[386] = poly[9] * poly[98];
    poly[387] = poly[1] * poly[182];
    poly[388] = poly[1] * poly[233];
    poly[389] = poly[1] * poly[183];
    poly[390] = poly[9] * poly[100];
    poly[391] = poly[1] * poly[234];
    poly[392] = poly[9] * poly[71];
    poly[393] = poly[1] * poly[185];
    poly[394] = poly[6] * poly[104];
    poly[395] = poly[9] * poly[74];
    poly[396] = poly[1] * poly[235];
    poly[397] = poly[6] * poly[106];
    poly[398] = poly[9] * poly[102];
    poly[399] = poly[1] * poly[237];
}

#[inline(never)]
fn f_polynomials8(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[400] = poly[1] * poly[238];
    poly[401] = poly[9] * poly[104];
    poly[402] = poly[1] * poly[239];
    poly[403] = poly[6] * poly[118];
    poly[404] = poly[9] * poly[106];
    poly[405] = poly[1] * poly[188];
    poly[406] = poly[1] * poly[189];
    poly[407] = poly[1] * poly[191];
    poly[408] = poly[4] * poly[223];
    poly[409] = poly[1] * poly[241];
    poly[410] = poly[4] * poly[225];
    poly[411] = poly[1] * poly[193];
    poly[412] = poly[4] * poly[227];
    poly[413] = poly[4] * poly[228];
    poly[414] = poly[1] * poly[196];
    poly[415] = poly[4] * poly[230];
    poly[416] = poly[4] * poly[231];
    poly[417] = poly[1] * poly[200];
    poly[418] = poly[4] * poly[233];
    poly[419] = poly[4] * poly[234];
    poly[420] = poly[4] * poly[235];
    poly[421] = poly[1] * poly[242];
    poly[422] = poly[4] * poly[237];
    poly[423] = poly[4] * poly[238];
    poly[424] = poly[4] * poly[239];
    poly[425] = poly[1] * poly[204];
    poly[426] = poly[1] * poly[205];
    poly[427] = poly[4] * poly[190];
    poly[428] = poly[1] * poly[244];
    poly[429] = poly[4] * poly[192];
    poly[430] = poly[1] * poly[207];
    poly[431] = poly[4] * poly[194];
    poly[432] = poly[4] * poly[195];
    poly[433] = poly[1] * poly[210];
    poly[434] = poly[4] * poly[197];
    poly[435] = poly[4] * poly[198];
    poly[436] = poly[4] * poly[199];
    poly[437] = poly[1] * poly[245];
    poly[438] = poly[4] * poly[201];
    poly[439] = poly[4] * poly[202];
    poly[440] = poly[4] * poly[203];
    poly[441] = poly[1] * poly[214];
    poly[442] = poly[1] * poly[247];
    poly[443] = poly[4] * poly[206];
    poly[444] = poly[1] * poly[216];
    poly[445] = poly[4] * poly[208];
    poly[446] = poly[4] * poly[209];
    poly[447] = poly[1] * poly[248];
    poly[448] = poly[4] * poly[211];
    poly[449] = poly[4] * poly[212];
}

#[inline(never)]
fn f_polynomials9(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[450] = poly[4] * poly[213];
    poly[451] = poly[1] * poly[250];
    poly[452] = poly[4] * poly[215];
    poly[453] = poly[1] * poly[251];
    poly[454] = poly[4] * poly[217];
    poly[455] = poly[4] * poly[218];
    poly[456] = poly[4] * poly[219];
    poly[457] = poly[1] * poly[220];
    poly[458] = poly[1] * poly[221];
    poly[459] = poly[1] * poly[222];
    poly[460] = poly[6] * poly[93];
    poly[461] = poly[1] * poly[224];
    poly[462] = poly[6] * poly[95];
    poly[463] = poly[1] * poly[253];
    poly[464] = poly[6] * poly[117];
    poly[465] = poly[1] * poly[226];
    poly[466] = poly[2] * poly[227] - poly[365];
    poly[467] = poly[2] * poly[228] - poly[371];
    poly[468] = poly[1] * poly[229];
    poly[469] = poly[2] * poly[230] - poly[380];
    poly[470] = poly[2] * poly[231] - poly[385];
    poly[471] = poly[1] * poly[232];
    poly[472] = poly[2] * poly[233] - poly[394];
    poly[473] = poly[9] * poly[101];
    poly[474] = poly[2] * poly[235] - poly[397];
    poly[475] = poly[1] * poly[236];
    poly[476] = poly[2] * poly[237] - poly[403];
    poly[477] = poly[9] * poly[105];
    poly[478] = poly[2] * poly[239] - poly[403];
    poly[479] = poly[1] * poly[254];
    poly[480] = poly[3] * poly[237] - poly[401];
    poly[481] = poly[9] * poly[118];
    poly[482] = poly[2] * poly[254] - poly[480];
    poly[483] = poly[1] * poly[240];
    poly[484] = poly[4] * poly[253];
    poly[485] = poly[4] * poly[254];
    poly[486] = poly[1] * poly[243];
    poly[487] = poly[4] * poly[241];
    poly[488] = poly[4] * poly[242];
    poly[489] = poly[1] * poly[246];
    poly[490] = poly[4] * poly[244];
    poly[491] = poly[4] * poly[245];
    poly[492] = poly[1] * poly[249];
    poly[493] = poly[4] * poly[247];
    poly[494] = poly[4] * poly[248];
    poly[495] = poly[1] * poly[255];
    poly[496] = poly[4] * poly[250];
    poly[497] = poly[4] * poly[251];
    poly[498] = poly[1] * poly[252];
    poly[499] = poly[2] * poly[253] - poly[464];
}

#[inline(never)]
fn f_polynomials10(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[500] = poly[3] * poly[254] - poly[481];
    poly[501] = poly[4] * poly[255];
    poly[502] = poly[1] * poly[256];
    poly[503] = poly[1] * poly[265];
    poly[504] = poly[1] * poly[269];
    poly[505] = poly[1] * poly[272];
    poly[506] = poly[1] * poly[257];
    poly[507] = poly[1] * poly[258];
    poly[508] = poly[1] * poly[280];
    poly[509] = poly[1] * poly[283];
    poly[510] = poly[1] * poly[259];
    poly[511] = poly[1] * poly[289];
    poly[512] = poly[1] * poly[291];
    poly[513] = poly[1] * poly[294];
    poly[514] = poly[1] * poly[298];
    poly[515] = poly[1] * poly[260];
    poly[516] = poly[1] * poly[261];
    poly[517] = poly[1] * poly[308];
    poly[518] = poly[1] * poly[262];
    poly[519] = poly[1] * poly[263];
    poly[520] = poly[1] * poly[310];
    poly[521] = poly[1] * poly[264];
    poly[522] = poly[4] * poly[280];
    poly[523] = poly[1] * poly[312];
    poly[524] = poly[1] * poly[313];
    poly[525] = poly[4] * poly[283];
    poly[526] = poly[1] * poly[266];
    poly[527] = poly[1] * poly[317];
    poly[528] = poly[1] * poly[267];
    poly[529] = poly[1] * poly[320];
    poly[530] = poly[1] * poly[268];
    poly[531] = poly[4] * poly[289];
    poly[532] = poly[1] * poly[321];
    poly[533] = poly[4] * poly[291];
    poly[534] = poly[1] * poly[323];
    poly[535] = poly[1] * poly[324];
    poly[536] = poly[4] * poly[294];
    poly[537] = poly[1] * poly[327];
    poly[538] = poly[1] * poly[329];
    poly[539] = poly[1] * poly[330];
    poly[540] = poly[4] * poly[298];
    poly[541] = poly[1] * poly[270];
    poly[542] = poly[1] * poly[337];
    poly[543] = poly[1] * poly[271];
    poly[544] = poly[1] * poly[339];
    poly[545] = poly[1] * poly[340];
    poly[546] = poly[4] * poly[265];
    poly[547] = poly[1] * poly[343];
    poly[548] = poly[1] * poly[345];
    poly[549] = poly[1] * poly[346];
}

#[inline(never)]
fn f_polynomials11(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[550] = poly[4] * poly[269];
    poly[551] = poly[1] * poly[351];
    poly[552] = poly[1] * poly[352];
    poly[553] = poly[4] * poly[272];
    poly[554] = poly[1] * poly[273];
    poly[555] = poly[1] * poly[274];
    poly[556] = poly[1] * poly[275];
    poly[557] = poly[1] * poly[365];
    poly[558] = poly[1] * poly[276];
    poly[559] = poly[1] * poly[277];
    poly[560] = poly[1] * poly[278];
    poly[561] = poly[1] * poly[367];
    poly[562] = poly[1] * poly[279];
    poly[563] = poly[1] * poly[281];
    poly[564] = poly[1] * poly[369];
    poly[565] = poly[1] * poly[282];
    poly[566] = poly[6] * poly[129];
    poly[567] = poly[1] * poly[371];
    poly[568] = poly[1] * poly[372];
    poly[569] = poly[6] * poly[170];
    poly[570] = poly[1] * poly[284];
    poly[571] = poly[1] * poly[285];
    poly[572] = poly[1] * poly[377];
    poly[573] = poly[1] * poly[286];
    poly[574] = poly[1] * poly[287];
    poly[575] = poly[1] * poly[380];
    poly[576] = poly[1] * poly[288];
    poly[577] = poly[6] * poly[174];
    poly[578] = poly[1] * poly[290];
    poly[579] = poly[1] * poly[292];
    poly[580] = poly[1] * poly[382];
    poly[581] = poly[1] * poly[293];
    poly[582] = poly[6] * poly[133];
    poly[583] = poly[1] * poly[383];
    poly[584] = poly[6] * poly[178];
    poly[585] = poly[1] * poly[385];
    poly[586] = poly[1] * poly[386];
    poly[587] = poly[6] * poly[181];
    poly[588] = poly[1] * poly[295];
    poly[589] = poly[1] * poly[390];
    poly[590] = poly[1] * poly[392];
    poly[591] = poly[1] * poly[296];
    poly[592] = poly[1] * poly[394];
    poly[593] = poly[1] * poly[297];
    poly[594] = poly[6] * poly[184];
    poly[595] = poly[1] * poly[395];
    poly[596] = poly[6] * poly[234];
    poly[597] = poly[1] * poly[397];
    poly[598] = poly[1] * poly[398];
    poly[599] = poly[6] * poly[187];
}

#[inline(never)]
fn f_polynomials12(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[600] = poly[1] * poly[401];
    poly[601] = poly[1] * poly[403];
    poly[602] = poly[1] * poly[404];
    poly[603] = poly[6] * poly[238];
    poly[604] = poly[1] * poly[299];
    poly[605] = poly[1] * poly[300];
    poly[606] = poly[1] * poly[301];
    poly[607] = poly[1] * poly[302];
    poly[608] = poly[1] * poly[408];
    poly[609] = poly[1] * poly[410];
    poly[610] = poly[1] * poly[303];
    poly[611] = poly[1] * poly[304];
    poly[612] = poly[1] * poly[305];
    poly[613] = poly[1] * poly[412];
    poly[614] = poly[1] * poly[306];
    poly[615] = poly[1] * poly[307];
    poly[616] = poly[4] * poly[365];
    poly[617] = poly[1] * poly[309];
    poly[618] = poly[4] * poly[367];
    poly[619] = poly[1] * poly[311];
    poly[620] = poly[4] * poly[369];
    poly[621] = poly[1] * poly[413];
    poly[622] = poly[4] * poly[371];
    poly[623] = poly[4] * poly[372];
    poly[624] = poly[1] * poly[314];
    poly[625] = poly[1] * poly[315];
    poly[626] = poly[1] * poly[415];
    poly[627] = poly[1] * poly[316];
    poly[628] = poly[4] * poly[377];
    poly[629] = poly[1] * poly[318];
    poly[630] = poly[1] * poly[319];
    poly[631] = poly[4] * poly[380];
    poly[632] = poly[1] * poly[322];
    poly[633] = poly[4] * poly[382];
    poly[634] = poly[4] * poly[383];
    poly[635] = poly[1] * poly[416];
    poly[636] = poly[4] * poly[385];
    poly[637] = poly[4] * poly[386];
    poly[638] = poly[1] * poly[325];
    poly[639] = poly[1] * poly[418];
    poly[640] = poly[1] * poly[326];
    poly[641] = poly[4] * poly[390];
    poly[642] = poly[1] * poly[419];
    poly[643] = poly[4] * poly[392];
    poly[644] = poly[1] * poly[328];
    poly[645] = poly[4] * poly[394];
    poly[646] = poly[4] * poly[395];
    poly[647] = poly[1] * poly[420];
    poly[648] = poly[4] * poly[397];
    poly[649] = poly[4] * poly[398];
}

#[inline(never)]
fn f_polynomials13(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[650] = poly[1] * poly[422];
    poly[651] = poly[1] * poly[423];
    poly[652] = poly[4] * poly[401];
    poly[653] = poly[1] * poly[424];
    poly[654] = poly[4] * poly[403];
    poly[655] = poly[4] * poly[404];
    poly[656] = poly[1] * poly[331];
    poly[657] = poly[1] * poly[332];
    poly[658] = poly[1] * poly[427];
    poly[659] = poly[1] * poly[429];
    poly[660] = poly[1] * poly[333];
    poly[661] = poly[1] * poly[334];
    poly[662] = poly[1] * poly[431];
    poly[663] = poly[1] * poly[335];
    poly[664] = poly[1] * poly[336];
    poly[665] = poly[4] * poly[308];
    poly[666] = poly[1] * poly[338];
    poly[667] = poly[4] * poly[310];
    poly[668] = poly[1] * poly[432];
    poly[669] = poly[4] * poly[312];
    poly[670] = poly[4] * poly[313];
    poly[671] = poly[1] * poly[341];
    poly[672] = poly[1] * poly[434];
    poly[673] = poly[1] * poly[342];
    poly[674] = poly[4] * poly[317];
    poly[675] = poly[1] * poly[435];
    poly[676] = poly[1] * poly[344];
    poly[677] = poly[4] * poly[320];
    poly[678] = poly[4] * poly[321];
    poly[679] = poly[1] * poly[436];
    poly[680] = poly[4] * poly[323];
    poly[681] = poly[4] * poly[324];
    poly[682] = poly[1] * poly[438];
    poly[683] = poly[1] * poly[439];
    poly[684] = poly[4] * poly[327];
    poly[685] = poly[1] * poly[440];
    poly[686] = poly[4] * poly[329];
    poly[687] = poly[4] * poly[330];
    poly[688] = poly[1] * poly[347];
    poly[689] = poly[1] * poly[443];
    poly[690] = poly[1] * poly[348];
    poly[691] = poly[1] * poly[445];
    poly[692] = poly[1] * poly[349];
    poly[693] = poly[1] * poly[350];
    poly[694] = poly[4] * poly[337];
    poly[695] = poly[1] * poly[446];
    poly[696] = poly[4] * poly[339];
    poly[697] = poly[4] * poly[340];
    poly[698] = poly[1] * poly[448];
    poly[699] = poly[1] * poly[449];
}

#[inline(never)]
fn f_polynomials14(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[700] = poly[4] * poly[343];
    poly[701] = poly[1] * poly[450];
    poly[702] = poly[4] * poly[345];
    poly[703] = poly[4] * poly[346];
    poly[704] = poly[1] * poly[452];
    poly[705] = poly[1] * poly[454];
    poly[706] = poly[1] * poly[455];
    poly[707] = poly[1] * poly[456];
    poly[708] = poly[4] * poly[351];
    poly[709] = poly[4] * poly[352];
    poly[710] = poly[1] * poly[353];
    poly[711] = poly[1] * poly[354];
    poly[712] = poly[1] * poly[355];
    poly[713] = poly[1] * poly[356];
    poly[714] = poly[1] * poly[357];
    poly[715] = poly[1] * poly[460];
    poly[716] = poly[1] * poly[358];
    poly[717] = poly[1] * poly[462];
    poly[718] = poly[1] * poly[464];
    poly[719] = poly[1] * poly[359];
    poly[720] = poly[1] * poly[360];
    poly[721] = poly[1] * poly[361];
    poly[722] = poly[1] * poly[362];
    poly[723] = poly[1] * poly[466];
    poly[724] = poly[1] * poly[363];
    poly[725] = poly[1] * poly[364];
    poly[726] = poly[6] * poly[227];
    poly[727] = poly[1] * poly[366];
    poly[728] = poly[6] * poly[165];
    poly[729] = poly[1] * poly[368];
    poly[730] = poly[6] * poly[167];
    poly[731] = poly[1] * poly[370];
    poly[732] = poly[6] * poly[169];
    poly[733] = poly[1] * poly[467];
    poly[734] = poly[6] * poly[228];
    poly[735] = poly[9] * poly[253];
    poly[736] = poly[1] * poly[373];
    poly[737] = poly[1] * poly[374];
    poly[738] = poly[1] * poly[375];
    poly[739] = poly[1] * poly[469];
    poly[740] = poly[1] * poly[376];
    poly[741] = poly[9] * poly[227];
    poly[742] = poly[1] * poly[378];
    poly[743] = poly[1] * poly[379];
    poly[744] = poly[6] * poly[230];
    poly[745] = poly[1] * poly[381];
    poly[746] = poly[6] * poly[177];
    poly[747] = poly[1] * poly[384];
    poly[748] = poly[6] * poly[180];
    poly[749] = poly[9] * poly[170];
}

#[inline(never)]
fn f_polynomials15(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[750] = poly[1] * poly[470];
    poly[751] = poly[6] * poly[231];
    poly[752] = poly[9] * poly[228];
    poly[753] = poly[1] * poly[387];
    poly[754] = poly[1] * poly[388];
    poly[755] = poly[1] * poly[472];
    poly[756] = poly[1] * poly[389];
    poly[757] = poly[9] * poly[230];
    poly[758] = poly[1] * poly[391];
    poly[759] = poly[9] * poly[174];
    poly[760] = poly[1] * poly[473];
    poly[761] = poly[1] * poly[393];
    poly[762] = poly[6] * poly[233];
    poly[763] = poly[9] * poly[178];
    poly[764] = poly[1] * poly[396];
    poly[765] = poly[6] * poly[186];
    poly[766] = poly[9] * poly[181];
    poly[767] = poly[1] * poly[474];
    poly[768] = poly[6] * poly[235];
    poly[769] = poly[9] * poly[231];
    poly[770] = poly[1] * poly[399];
    poly[771] = poly[1] * poly[476];
    poly[772] = poly[1] * poly[400];
    poly[773] = poly[9] * poly[233];
    poly[774] = poly[1] * poly[477];
    poly[775] = poly[9] * poly[184];
    poly[776] = poly[1] * poly[402];
    poly[777] = poly[6] * poly[237];
    poly[778] = poly[9] * poly[187];
    poly[779] = poly[1] * poly[478];
    poly[780] = poly[6] * poly[239];
    poly[781] = poly[9] * poly[235];
    poly[782] = poly[1] * poly[480];
    poly[783] = poly[1] * poly[481];
    poly[784] = poly[9] * poly[237];
    poly[785] = poly[1] * poly[482];
    poly[786] = poly[6] * poly[254];
    poly[787] = poly[9] * poly[239];
    poly[788] = poly[1] * poly[405];
    poly[789] = poly[1] * poly[406];
    poly[790] = poly[1] * poly[407];
    poly[791] = poly[4] * poly[460];
    poly[792] = poly[1] * poly[409];
    poly[793] = poly[4] * poly[462];
    poly[794] = poly[1] * poly[484];
    poly[795] = poly[4] * poly[464];
    poly[796] = poly[1] * poly[411];
    poly[797] = poly[4] * poly[466];
    poly[798] = poly[4] * poly[467];
    poly[799] = poly[1] * poly[414];
}

#[inline(never)]
fn f_polynomials16(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[800] = poly[4] * poly[469];
    poly[801] = poly[4] * poly[470];
    poly[802] = poly[1] * poly[417];
    poly[803] = poly[4] * poly[472];
    poly[804] = poly[4] * poly[473];
    poly[805] = poly[4] * poly[474];
    poly[806] = poly[1] * poly[421];
    poly[807] = poly[4] * poly[476];
    poly[808] = poly[4] * poly[477];
    poly[809] = poly[4] * poly[478];
    poly[810] = poly[1] * poly[485];
    poly[811] = poly[4] * poly[480];
    poly[812] = poly[4] * poly[481];
    poly[813] = poly[4] * poly[482];
    poly[814] = poly[1] * poly[425];
    poly[815] = poly[1] * poly[426];
    poly[816] = poly[1] * poly[428];
    poly[817] = poly[4] * poly[408];
    poly[818] = poly[1] * poly[487];
    poly[819] = poly[4] * poly[410];
    poly[820] = poly[1] * poly[430];
    poly[821] = poly[4] * poly[412];
    poly[822] = poly[4] * poly[413];
    poly[823] = poly[1] * poly[433];
    poly[824] = poly[4] * poly[415];
    poly[825] = poly[4] * poly[416];
    poly[826] = poly[1] * poly[437];
    poly[827] = poly[4] * poly[418];
    poly[828] = poly[4] * poly[419];
    poly[829] = poly[4] * poly[420];
    poly[830] = poly[1] * poly[488];
    poly[831] = poly[4] * poly[422];
    poly[832] = poly[4] * poly[423];
    poly[833] = poly[4] * poly[424];
    poly[834] = poly[1] * poly[441];
    poly[835] = poly[1] * poly[442];
    poly[836] = poly[4] * poly[427];
    poly[837] = poly[1] * poly[490];
    poly[838] = poly[4] * poly[429];
    poly[839] = poly[1] * poly[444];
    poly[840] = poly[4] * poly[431];
    poly[841] = poly[4] * poly[432];
    poly[842] = poly[1] * poly[447];
    poly[843] = poly[4] * poly[434];
    poly[844] = poly[4] * poly[435];
    poly[845] = poly[4] * poly[436];
    poly[846] = poly[1] * poly[491];
    poly[847] = poly[4] * poly[438];
    poly[848] = poly[4] * poly[439];
    poly[849] = poly[4] * poly[440];
}

#[inline(never)]
fn f_polynomials17(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[850] = poly[1] * poly[451];
    poly[851] = poly[1] * poly[493];
    poly[852] = poly[4] * poly[443];
    poly[853] = poly[1] * poly[453];
    poly[854] = poly[4] * poly[445];
    poly[855] = poly[4] * poly[446];
    poly[856] = poly[1] * poly[494];
    poly[857] = poly[4] * poly[448];
    poly[858] = poly[4] * poly[449];
    poly[859] = poly[4] * poly[450];
    poly[860] = poly[1] * poly[496];
    poly[861] = poly[4] * poly[452];
    poly[862] = poly[1] * poly[497];
    poly[863] = poly[4] * poly[454];
    poly[864] = poly[4] * poly[455];
    poly[865] = poly[4] * poly[456];
    poly[866] = poly[1] * poly[457];
    poly[867] = poly[1] * poly[458];
    poly[868] = poly[1] * poly[459];
    poly[869] = poly[1] * poly[461];
    poly[870] = poly[6] * poly[223];
    poly[871] = poly[1] * poly[463];
    poly[872] = poly[6] * poly[225];
    poly[873] = poly[1] * poly[499];
    poly[874] = poly[6] * poly[253];
    poly[875] = poly[1] * poly[465];
    poly[876] = poly[2] * poly[466] - poly[726];
    poly[877] = poly[2] * poly[467] - poly[734];
    poly[878] = poly[1] * poly[468];
    poly[879] = poly[2] * poly[469] - poly[744];
    poly[880] = poly[2] * poly[470] - poly[751];
    poly[881] = poly[1] * poly[471];
    poly[882] = poly[2] * poly[472] - poly[762];
    poly[883] = poly[2] * poly[474] - poly[768];
    poly[884] = poly[1] * poly[475];
    poly[885] = poly[2] * poly[476] - poly[777];
    poly[886] = poly[9] * poly[234];
    poly[887] = poly[2] * poly[478] - poly[780];
    poly[888] = poly[1] * poly[479];
    poly[889] = poly[2] * poly[480] - poly[786];
    poly[890] = poly[9] * poly[238];
    poly[891] = poly[2] * poly[482] - poly[786];
    poly[892] = poly[1] * poly[500];
    poly[893] = poly[3] * poly[480] - poly[784];
    poly[894] = poly[9] * poly[254];
    poly[895] = poly[2] * poly[500] - poly[893];
    poly[896] = poly[1] * poly[483];
    poly[897] = poly[4] * poly[499];
    poly[898] = poly[4] * poly[500];
    poly[899] = poly[1] * poly[486];
}

#[inline(never)]
fn f_polynomials18(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[900] = poly[4] * poly[484];
    poly[901] = poly[4] * poly[485];
    poly[902] = poly[1] * poly[489];
    poly[903] = poly[4] * poly[487];
    poly[904] = poly[4] * poly[488];
    poly[905] = poly[1] * poly[492];
    poly[906] = poly[4] * poly[490];
    poly[907] = poly[4] * poly[491];
    poly[908] = poly[1] * poly[495];
    poly[909] = poly[4] * poly[493];
    poly[910] = poly[4] * poly[494];
    poly[911] = poly[1] * poly[501];
    poly[912] = poly[4] * poly[496];
    poly[913] = poly[4] * poly[497];
    poly[914] = poly[1] * poly[498];
    poly[915] = poly[2] * poly[499] - poly[874];
    poly[916] = poly[3] * poly[500] - poly[894];
    poly[917] = poly[4] * poly[501];
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

    return poly;
}
