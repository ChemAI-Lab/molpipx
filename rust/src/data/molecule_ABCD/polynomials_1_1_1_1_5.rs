#![allow(unused_variables)]
use super::monomials_1_1_1_1_5::*;


pub const N_POLYS: usize = 462;

// File created from data/molecule_ABCD/MOL_1_1_1_1_5.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4];
    poly[5] = mono[5];
    poly[6] = mono[6];
    poly[7] = poly[1] * poly[2];
    poly[8] = poly[1] * poly[3];
    poly[9] = poly[2] * poly[3];
    poly[10] = poly[1] * poly[4];
    poly[11] = poly[2] * poly[4];
    poly[12] = poly[3] * poly[4];
    poly[13] = poly[1] * poly[5];
    poly[14] = poly[2] * poly[5];
    poly[15] = poly[3] * poly[5];
    poly[16] = poly[4] * poly[5];
    poly[17] = poly[1] * poly[6];
    poly[18] = poly[2] * poly[6];
    poly[19] = poly[3] * poly[6];
    poly[20] = poly[4] * poly[6];
    poly[21] = poly[5] * poly[6];
    poly[22] = poly[1] * poly[1];
    poly[23] = poly[2] * poly[2];
    poly[24] = poly[3] * poly[3];
    poly[25] = poly[4] * poly[4];
    poly[26] = poly[5] * poly[5];
    poly[27] = poly[6] * poly[6];
    poly[28] = poly[1] * poly[9];
    poly[29] = poly[1] * poly[11];
    poly[30] = poly[1] * poly[12];
    poly[31] = poly[2] * poly[12];
    poly[32] = poly[1] * poly[14];
    poly[33] = poly[1] * poly[15];
    poly[34] = poly[2] * poly[15];
    poly[35] = poly[1] * poly[16];
    poly[36] = poly[2] * poly[16];
    poly[37] = poly[3] * poly[16];
    poly[38] = poly[1] * poly[18];
    poly[39] = poly[1] * poly[19];
    poly[40] = poly[2] * poly[19];
    poly[41] = poly[1] * poly[20];
    poly[42] = poly[2] * poly[20];
    poly[43] = poly[3] * poly[20];
    poly[44] = poly[1] * poly[21];
    poly[45] = poly[2] * poly[21];
    poly[46] = poly[3] * poly[21];
    poly[47] = poly[4] * poly[21];
    poly[48] = poly[1] * poly[7];
    poly[49] = poly[1] * poly[23];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[8];
    poly[51] = poly[2] * poly[9];
    poly[52] = poly[1] * poly[24];
    poly[53] = poly[2] * poly[24];
    poly[54] = poly[1] * poly[10];
    poly[55] = poly[2] * poly[11];
    poly[56] = poly[3] * poly[12];
    poly[57] = poly[1] * poly[25];
    poly[58] = poly[2] * poly[25];
    poly[59] = poly[3] * poly[25];
    poly[60] = poly[1] * poly[13];
    poly[61] = poly[2] * poly[14];
    poly[62] = poly[3] * poly[15];
    poly[63] = poly[4] * poly[16];
    poly[64] = poly[1] * poly[26];
    poly[65] = poly[2] * poly[26];
    poly[66] = poly[3] * poly[26];
    poly[67] = poly[4] * poly[26];
    poly[68] = poly[1] * poly[17];
    poly[69] = poly[2] * poly[18];
    poly[70] = poly[3] * poly[19];
    poly[71] = poly[4] * poly[20];
    poly[72] = poly[5] * poly[21];
    poly[73] = poly[1] * poly[27];
    poly[74] = poly[2] * poly[27];
    poly[75] = poly[3] * poly[27];
    poly[76] = poly[4] * poly[27];
    poly[77] = poly[5] * poly[27];
    poly[78] = poly[1] * poly[22];
    poly[79] = poly[2] * poly[23];
    poly[80] = poly[3] * poly[24];
    poly[81] = poly[4] * poly[25];
    poly[82] = poly[5] * poly[26];
    poly[83] = poly[6] * poly[27];
    poly[84] = poly[1] * poly[31];
    poly[85] = poly[1] * poly[34];
    poly[86] = poly[1] * poly[36];
    poly[87] = poly[1] * poly[37];
    poly[88] = poly[2] * poly[37];
    poly[89] = poly[1] * poly[40];
    poly[90] = poly[1] * poly[42];
    poly[91] = poly[1] * poly[43];
    poly[92] = poly[2] * poly[43];
    poly[93] = poly[1] * poly[45];
    poly[94] = poly[1] * poly[46];
    poly[95] = poly[2] * poly[46];
    poly[96] = poly[1] * poly[47];
    poly[97] = poly[2] * poly[47];
    poly[98] = poly[3] * poly[47];
    poly[99] = poly[1] * poly[28];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[51];
    poly[101] = poly[1] * poly[53];
    poly[102] = poly[1] * poly[29];
    poly[103] = poly[1] * poly[55];
    poly[104] = poly[1] * poly[30];
    poly[105] = poly[2] * poly[31];
    poly[106] = poly[1] * poly[56];
    poly[107] = poly[2] * poly[56];
    poly[108] = poly[1] * poly[58];
    poly[109] = poly[1] * poly[59];
    poly[110] = poly[2] * poly[59];
    poly[111] = poly[1] * poly[32];
    poly[112] = poly[1] * poly[61];
    poly[113] = poly[1] * poly[33];
    poly[114] = poly[2] * poly[34];
    poly[115] = poly[1] * poly[62];
    poly[116] = poly[2] * poly[62];
    poly[117] = poly[1] * poly[35];
    poly[118] = poly[2] * poly[36];
    poly[119] = poly[3] * poly[37];
    poly[120] = poly[1] * poly[63];
    poly[121] = poly[2] * poly[63];
    poly[122] = poly[3] * poly[63];
    poly[123] = poly[1] * poly[65];
    poly[124] = poly[1] * poly[66];
    poly[125] = poly[2] * poly[66];
    poly[126] = poly[1] * poly[67];
    poly[127] = poly[2] * poly[67];
    poly[128] = poly[3] * poly[67];
    poly[129] = poly[1] * poly[38];
    poly[130] = poly[1] * poly[69];
    poly[131] = poly[1] * poly[39];
    poly[132] = poly[2] * poly[40];
    poly[133] = poly[1] * poly[70];
    poly[134] = poly[2] * poly[70];
    poly[135] = poly[1] * poly[41];
    poly[136] = poly[2] * poly[42];
    poly[137] = poly[3] * poly[43];
    poly[138] = poly[1] * poly[71];
    poly[139] = poly[2] * poly[71];
    poly[140] = poly[3] * poly[71];
    poly[141] = poly[1] * poly[44];
    poly[142] = poly[2] * poly[45];
    poly[143] = poly[3] * poly[46];
    poly[144] = poly[4] * poly[47];
    poly[145] = poly[1] * poly[72];
    poly[146] = poly[2] * poly[72];
    poly[147] = poly[3] * poly[72];
    poly[148] = poly[4] * poly[72];
    poly[149] = poly[1] * poly[74];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[75];
    poly[151] = poly[2] * poly[75];
    poly[152] = poly[1] * poly[76];
    poly[153] = poly[2] * poly[76];
    poly[154] = poly[3] * poly[76];
    poly[155] = poly[1] * poly[77];
    poly[156] = poly[2] * poly[77];
    poly[157] = poly[3] * poly[77];
    poly[158] = poly[4] * poly[77];
    poly[159] = poly[1] * poly[48];
    poly[160] = poly[1] * poly[49];
    poly[161] = poly[1] * poly[79];
    poly[162] = poly[1] * poly[50];
    poly[163] = poly[2] * poly[51];
    poly[164] = poly[1] * poly[52];
    poly[165] = poly[2] * poly[53];
    poly[166] = poly[1] * poly[80];
    poly[167] = poly[2] * poly[80];
    poly[168] = poly[1] * poly[54];
    poly[169] = poly[2] * poly[55];
    poly[170] = poly[3] * poly[56];
    poly[171] = poly[1] * poly[57];
    poly[172] = poly[2] * poly[58];
    poly[173] = poly[3] * poly[59];
    poly[174] = poly[1] * poly[81];
    poly[175] = poly[2] * poly[81];
    poly[176] = poly[3] * poly[81];
    poly[177] = poly[1] * poly[60];
    poly[178] = poly[2] * poly[61];
    poly[179] = poly[3] * poly[62];
    poly[180] = poly[4] * poly[63];
    poly[181] = poly[1] * poly[64];
    poly[182] = poly[2] * poly[65];
    poly[183] = poly[3] * poly[66];
    poly[184] = poly[4] * poly[67];
    poly[185] = poly[1] * poly[82];
    poly[186] = poly[2] * poly[82];
    poly[187] = poly[3] * poly[82];
    poly[188] = poly[4] * poly[82];
    poly[189] = poly[1] * poly[68];
    poly[190] = poly[2] * poly[69];
    poly[191] = poly[3] * poly[70];
    poly[192] = poly[4] * poly[71];
    poly[193] = poly[5] * poly[72];
    poly[194] = poly[1] * poly[73];
    poly[195] = poly[2] * poly[74];
    poly[196] = poly[3] * poly[75];
    poly[197] = poly[4] * poly[76];
    poly[198] = poly[5] * poly[77];
    poly[199] = poly[1] * poly[83];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[2] * poly[83];
    poly[201] = poly[3] * poly[83];
    poly[202] = poly[4] * poly[83];
    poly[203] = poly[5] * poly[83];
    poly[204] = poly[1] * poly[78];
    poly[205] = poly[2] * poly[79];
    poly[206] = poly[3] * poly[80];
    poly[207] = poly[4] * poly[81];
    poly[208] = poly[5] * poly[82];
    poly[209] = poly[6] * poly[83];
    poly[210] = poly[1] * poly[88];
    poly[211] = poly[1] * poly[92];
    poly[212] = poly[1] * poly[95];
    poly[213] = poly[1] * poly[97];
    poly[214] = poly[1] * poly[98];
    poly[215] = poly[2] * poly[98];
    poly[216] = poly[1] * poly[84];
    poly[217] = poly[1] * poly[105];
    poly[218] = poly[1] * poly[107];
    poly[219] = poly[1] * poly[110];
    poly[220] = poly[1] * poly[85];
    poly[221] = poly[1] * poly[114];
    poly[222] = poly[1] * poly[116];
    poly[223] = poly[1] * poly[86];
    poly[224] = poly[1] * poly[118];
    poly[225] = poly[1] * poly[87];
    poly[226] = poly[2] * poly[88];
    poly[227] = poly[1] * poly[119];
    poly[228] = poly[2] * poly[119];
    poly[229] = poly[1] * poly[121];
    poly[230] = poly[1] * poly[122];
    poly[231] = poly[2] * poly[122];
    poly[232] = poly[1] * poly[125];
    poly[233] = poly[1] * poly[127];
    poly[234] = poly[1] * poly[128];
    poly[235] = poly[2] * poly[128];
    poly[236] = poly[1] * poly[89];
    poly[237] = poly[1] * poly[132];
    poly[238] = poly[1] * poly[134];
    poly[239] = poly[1] * poly[90];
    poly[240] = poly[1] * poly[136];
    poly[241] = poly[1] * poly[91];
    poly[242] = poly[2] * poly[92];
    poly[243] = poly[1] * poly[137];
    poly[244] = poly[2] * poly[137];
    poly[245] = poly[1] * poly[139];
    poly[246] = poly[1] * poly[140];
    poly[247] = poly[2] * poly[140];
    poly[248] = poly[1] * poly[93];
    poly[249] = poly[1] * poly[142];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[1] * poly[94];
    poly[251] = poly[2] * poly[95];
    poly[252] = poly[1] * poly[143];
    poly[253] = poly[2] * poly[143];
    poly[254] = poly[1] * poly[96];
    poly[255] = poly[2] * poly[97];
    poly[256] = poly[3] * poly[98];
    poly[257] = poly[1] * poly[144];
    poly[258] = poly[2] * poly[144];
    poly[259] = poly[3] * poly[144];
    poly[260] = poly[1] * poly[146];
    poly[261] = poly[1] * poly[147];
    poly[262] = poly[2] * poly[147];
    poly[263] = poly[1] * poly[148];
    poly[264] = poly[2] * poly[148];
    poly[265] = poly[3] * poly[148];
    poly[266] = poly[1] * poly[151];
    poly[267] = poly[1] * poly[153];
    poly[268] = poly[1] * poly[154];
    poly[269] = poly[2] * poly[154];
    poly[270] = poly[1] * poly[156];
    poly[271] = poly[1] * poly[157];
    poly[272] = poly[2] * poly[157];
    poly[273] = poly[1] * poly[158];
    poly[274] = poly[2] * poly[158];
    poly[275] = poly[3] * poly[158];
    poly[276] = poly[1] * poly[99];
    poly[277] = poly[1] * poly[100];
    poly[278] = poly[1] * poly[163];
    poly[279] = poly[1] * poly[101];
    poly[280] = poly[1] * poly[165];
    poly[281] = poly[1] * poly[167];
    poly[282] = poly[1] * poly[102];
    poly[283] = poly[1] * poly[103];
    poly[284] = poly[1] * poly[169];
    poly[285] = poly[1] * poly[104];
    poly[286] = poly[2] * poly[105];
    poly[287] = poly[1] * poly[106];
    poly[288] = poly[2] * poly[107];
    poly[289] = poly[1] * poly[170];
    poly[290] = poly[2] * poly[170];
    poly[291] = poly[1] * poly[108];
    poly[292] = poly[1] * poly[172];
    poly[293] = poly[1] * poly[109];
    poly[294] = poly[2] * poly[110];
    poly[295] = poly[1] * poly[173];
    poly[296] = poly[2] * poly[173];
    poly[297] = poly[1] * poly[175];
    poly[298] = poly[1] * poly[176];
    poly[299] = poly[2] * poly[176];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[1] * poly[111];
    poly[301] = poly[1] * poly[112];
    poly[302] = poly[1] * poly[178];
    poly[303] = poly[1] * poly[113];
    poly[304] = poly[2] * poly[114];
    poly[305] = poly[1] * poly[115];
    poly[306] = poly[2] * poly[116];
    poly[307] = poly[1] * poly[179];
    poly[308] = poly[2] * poly[179];
    poly[309] = poly[1] * poly[117];
    poly[310] = poly[2] * poly[118];
    poly[311] = poly[3] * poly[119];
    poly[312] = poly[1] * poly[120];
    poly[313] = poly[2] * poly[121];
    poly[314] = poly[3] * poly[122];
    poly[315] = poly[1] * poly[180];
    poly[316] = poly[2] * poly[180];
    poly[317] = poly[3] * poly[180];
    poly[318] = poly[1] * poly[123];
    poly[319] = poly[1] * poly[182];
    poly[320] = poly[1] * poly[124];
    poly[321] = poly[2] * poly[125];
    poly[322] = poly[1] * poly[183];
    poly[323] = poly[2] * poly[183];
    poly[324] = poly[1] * poly[126];
    poly[325] = poly[2] * poly[127];
    poly[326] = poly[3] * poly[128];
    poly[327] = poly[1] * poly[184];
    poly[328] = poly[2] * poly[184];
    poly[329] = poly[3] * poly[184];
    poly[330] = poly[1] * poly[186];
    poly[331] = poly[1] * poly[187];
    poly[332] = poly[2] * poly[187];
    poly[333] = poly[1] * poly[188];
    poly[334] = poly[2] * poly[188];
    poly[335] = poly[3] * poly[188];
    poly[336] = poly[1] * poly[129];
    poly[337] = poly[1] * poly[130];
    poly[338] = poly[1] * poly[190];
    poly[339] = poly[1] * poly[131];
    poly[340] = poly[2] * poly[132];
    poly[341] = poly[1] * poly[133];
    poly[342] = poly[2] * poly[134];
    poly[343] = poly[1] * poly[191];
    poly[344] = poly[2] * poly[191];
    poly[345] = poly[1] * poly[135];
    poly[346] = poly[2] * poly[136];
    poly[347] = poly[3] * poly[137];
    poly[348] = poly[1] * poly[138];
    poly[349] = poly[2] * poly[139];
}

#[inline(never)]
fn f_polynomials7(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[350] = poly[3] * poly[140];
    poly[351] = poly[1] * poly[192];
    poly[352] = poly[2] * poly[192];
    poly[353] = poly[3] * poly[192];
    poly[354] = poly[1] * poly[141];
    poly[355] = poly[2] * poly[142];
    poly[356] = poly[3] * poly[143];
    poly[357] = poly[4] * poly[144];
    poly[358] = poly[1] * poly[145];
    poly[359] = poly[2] * poly[146];
    poly[360] = poly[3] * poly[147];
    poly[361] = poly[4] * poly[148];
    poly[362] = poly[1] * poly[193];
    poly[363] = poly[2] * poly[193];
    poly[364] = poly[3] * poly[193];
    poly[365] = poly[4] * poly[193];
    poly[366] = poly[1] * poly[149];
    poly[367] = poly[1] * poly[195];
    poly[368] = poly[1] * poly[150];
    poly[369] = poly[2] * poly[151];
    poly[370] = poly[1] * poly[196];
    poly[371] = poly[2] * poly[196];
    poly[372] = poly[1] * poly[152];
    poly[373] = poly[2] * poly[153];
    poly[374] = poly[3] * poly[154];
    poly[375] = poly[1] * poly[197];
    poly[376] = poly[2] * poly[197];
    poly[377] = poly[3] * poly[197];
    poly[378] = poly[1] * poly[155];
    poly[379] = poly[2] * poly[156];
    poly[380] = poly[3] * poly[157];
    poly[381] = poly[4] * poly[158];
    poly[382] = poly[1] * poly[198];
    poly[383] = poly[2] * poly[198];
    poly[384] = poly[3] * poly[198];
    poly[385] = poly[4] * poly[198];
    poly[386] = poly[1] * poly[200];
    poly[387] = poly[1] * poly[201];
    poly[388] = poly[2] * poly[201];
    poly[389] = poly[1] * poly[202];
    poly[390] = poly[2] * poly[202];
    poly[391] = poly[3] * poly[202];
    poly[392] = poly[1] * poly[203];
    poly[393] = poly[2] * poly[203];
    poly[394] = poly[3] * poly[203];
    poly[395] = poly[4] * poly[203];
    poly[396] = poly[1] * poly[159];
    poly[397] = poly[1] * poly[160];
    poly[398] = poly[1] * poly[161];
    poly[399] = poly[1] * poly[205];
}

#[inline(never)]
fn f_polynomials8(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[400] = poly[1] * poly[162];
    poly[401] = poly[2] * poly[163];
    poly[402] = poly[1] * poly[164];
    poly[403] = poly[2] * poly[165];
    poly[404] = poly[1] * poly[166];
    poly[405] = poly[2] * poly[167];
    poly[406] = poly[1] * poly[206];
    poly[407] = poly[2] * poly[206];
    poly[408] = poly[1] * poly[168];
    poly[409] = poly[2] * poly[169];
    poly[410] = poly[3] * poly[170];
    poly[411] = poly[1] * poly[171];
    poly[412] = poly[2] * poly[172];
    poly[413] = poly[3] * poly[173];
    poly[414] = poly[1] * poly[174];
    poly[415] = poly[2] * poly[175];
    poly[416] = poly[3] * poly[176];
    poly[417] = poly[1] * poly[207];
    poly[418] = poly[2] * poly[207];
    poly[419] = poly[3] * poly[207];
    poly[420] = poly[1] * poly[177];
    poly[421] = poly[2] * poly[178];
    poly[422] = poly[3] * poly[179];
    poly[423] = poly[4] * poly[180];
    poly[424] = poly[1] * poly[181];
    poly[425] = poly[2] * poly[182];
    poly[426] = poly[3] * poly[183];
    poly[427] = poly[4] * poly[184];
    poly[428] = poly[1] * poly[185];
    poly[429] = poly[2] * poly[186];
    poly[430] = poly[3] * poly[187];
    poly[431] = poly[4] * poly[188];
    poly[432] = poly[1] * poly[208];
    poly[433] = poly[2] * poly[208];
    poly[434] = poly[3] * poly[208];
    poly[435] = poly[4] * poly[208];
    poly[436] = poly[1] * poly[189];
    poly[437] = poly[2] * poly[190];
    poly[438] = poly[3] * poly[191];
    poly[439] = poly[4] * poly[192];
    poly[440] = poly[5] * poly[193];
    poly[441] = poly[1] * poly[194];
    poly[442] = poly[2] * poly[195];
    poly[443] = poly[3] * poly[196];
    poly[444] = poly[4] * poly[197];
    poly[445] = poly[5] * poly[198];
    poly[446] = poly[1] * poly[199];
    poly[447] = poly[2] * poly[200];
    poly[448] = poly[3] * poly[201];
    poly[449] = poly[4] * poly[202];
}

#[inline(never)]
fn f_polynomials9(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[450] = poly[5] * poly[203];
    poly[451] = poly[1] * poly[209];
    poly[452] = poly[2] * poly[209];
    poly[453] = poly[3] * poly[209];
    poly[454] = poly[4] * poly[209];
    poly[455] = poly[5] * poly[209];
    poly[456] = poly[1] * poly[204];
    poly[457] = poly[2] * poly[205];
    poly[458] = poly[3] * poly[206];
    poly[459] = poly[4] * poly[207];
    poly[460] = poly[5] * poly[208];
    poly[461] = poly[6] * poly[209];
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

    return poly;
}
