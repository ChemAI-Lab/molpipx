// File created from ../pipx/examples/Data/DDEthanol/PIP_deg_3/MOL_1_1_1_2_3_1_3.MONO 
// Total number of monomials = 1067 

pub const N_MONOS: usize = 1067;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 36;
pub const N_ATOMS: usize = 9;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
  mono[37] = mono[1] * mono[2];
  mono[38] = mono[1] * mono[3];
  mono[39] = mono[2] * mono[3];
  mono[40] = mono[3] * mono[4];
  mono[41] = mono[2] * mono[5];
  mono[42] = mono[1] * mono[6];
  mono[43] = mono[4] * mono[5];
  mono[44] = mono[4] * mono[6];
  mono[45] = mono[5] * mono[6];
  mono[46] = mono[7] * mono[8];
  mono[47] = mono[2] * mono[9];
  mono[48] = mono[3] * mono[9];
  mono[49] = mono[1] * mono[10];
  mono[50] = mono[3] * mono[10];
  mono[51] = mono[1] * mono[11];
  mono[52] = mono[2] * mono[11];
  mono[53] = mono[2] * mono[12];
  mono[54] = mono[3] * mono[12];
  mono[55] = mono[1] * mono[13];
  mono[56] = mono[3] * mono[13];
  mono[57] = mono[1] * mono[14];
  mono[58] = mono[2] * mono[14];
  mono[59] = mono[6] * mono[9];
  mono[60] = mono[5] * mono[10];
  mono[61] = mono[4] * mono[11];
  mono[62] = mono[6] * mono[12];
  mono[63] = mono[5] * mono[13];
  mono[64] = mono[4] * mono[14];
  mono[65] = mono[8] * mono[9];
  mono[66] = mono[8] * mono[10];
  mono[67] = mono[8] * mono[11];
  mono[68] = mono[7] * mono[12];
  mono[69] = mono[7] * mono[13];
  mono[70] = mono[7] * mono[14];
  mono[71] = mono[10] * mono[12];
  mono[72] = mono[11] * mono[12];
  mono[73] = mono[9] * mono[13];
  mono[74] = mono[11] * mono[13];
  mono[75] = mono[9] * mono[14];
  mono[76] = mono[10] * mono[14];
  mono[77] = mono[9] * mono[12];
  mono[78] = mono[10] * mono[13];
  mono[79] = mono[11] * mono[14];
  mono[80] = mono[9] * mono[10];
  mono[81] = mono[9] * mono[11];
  mono[82] = mono[10] * mono[11];
  mono[83] = mono[12] * mono[13];
  mono[84] = mono[12] * mono[14];
  mono[85] = mono[13] * mono[14];
  mono[86] = mono[2] * mono[17];
}

fn f_monomials1(mono: &mut [f64; N_MONOS]) { 
  mono[87] = mono[3] * mono[17];
  mono[88] = mono[1] * mono[18];
  mono[89] = mono[3] * mono[18];
  mono[90] = mono[1] * mono[19];
  mono[91] = mono[2] * mono[19];
  mono[92] = mono[6] * mono[17];
  mono[93] = mono[5] * mono[18];
  mono[94] = mono[4] * mono[19];
  mono[95] = mono[10] * mono[17];
  mono[96] = mono[11] * mono[17];
  mono[97] = mono[13] * mono[17];
  mono[98] = mono[14] * mono[17];
  mono[99] = mono[9] * mono[18];
  mono[100] = mono[11] * mono[18];
  mono[101] = mono[12] * mono[18];
  mono[102] = mono[14] * mono[18];
  mono[103] = mono[9] * mono[19];
  mono[104] = mono[10] * mono[19];
  mono[105] = mono[12] * mono[19];
  mono[106] = mono[13] * mono[19];
  mono[107] = mono[17] * mono[18];
  mono[108] = mono[17] * mono[19];
  mono[109] = mono[18] * mono[19];
  mono[110] = mono[8] * mono[20];
  mono[111] = mono[7] * mono[21];
  mono[112] = mono[12] * mono[20];
  mono[113] = mono[13] * mono[20];
  mono[114] = mono[14] * mono[20];
  mono[115] = mono[9] * mono[21];
  mono[116] = mono[10] * mono[21];
  mono[117] = mono[11] * mono[21];
  mono[118] = mono[20] * mono[21];
  mono[119] = mono[2] * mono[23];
  mono[120] = mono[3] * mono[23];
  mono[121] = mono[1] * mono[24];
  mono[122] = mono[3] * mono[24];
  mono[123] = mono[1] * mono[25];
  mono[124] = mono[2] * mono[25];
  mono[125] = mono[6] * mono[23];
  mono[126] = mono[5] * mono[24];
  mono[127] = mono[4] * mono[25];
  mono[128] = mono[10] * mono[23];
  mono[129] = mono[11] * mono[23];
  mono[130] = mono[13] * mono[23];
  mono[131] = mono[14] * mono[23];
  mono[132] = mono[9] * mono[24];
  mono[133] = mono[11] * mono[24];
  mono[134] = mono[12] * mono[24];
  mono[135] = mono[14] * mono[24];
  mono[136] = mono[9] * mono[25];
}

fn f_monomials2(mono: &mut [f64; N_MONOS]) { 
  mono[137] = mono[10] * mono[25];
  mono[138] = mono[12] * mono[25];
  mono[139] = mono[13] * mono[25];
  mono[140] = mono[18] * mono[23];
  mono[141] = mono[19] * mono[23];
  mono[142] = mono[17] * mono[24];
  mono[143] = mono[19] * mono[24];
  mono[144] = mono[17] * mono[25];
  mono[145] = mono[18] * mono[25];
  mono[146] = mono[23] * mono[24];
  mono[147] = mono[23] * mono[25];
  mono[148] = mono[24] * mono[25];
  mono[149] = mono[8] * mono[26];
  mono[150] = mono[7] * mono[27];
  mono[151] = mono[12] * mono[26];
  mono[152] = mono[13] * mono[26];
  mono[153] = mono[14] * mono[26];
  mono[154] = mono[9] * mono[27];
  mono[155] = mono[10] * mono[27];
  mono[156] = mono[11] * mono[27];
  mono[157] = mono[21] * mono[26];
  mono[158] = mono[20] * mono[27];
  mono[159] = mono[26] * mono[27];
  mono[160] = mono[2] * mono[30];
  mono[161] = mono[3] * mono[30];
  mono[162] = mono[1] * mono[31];
  mono[163] = mono[3] * mono[31];
  mono[164] = mono[1] * mono[32];
  mono[165] = mono[2] * mono[32];
  mono[166] = mono[6] * mono[30];
  mono[167] = mono[5] * mono[31];
  mono[168] = mono[4] * mono[32];
  mono[169] = mono[10] * mono[30];
  mono[170] = mono[11] * mono[30];
  mono[171] = mono[13] * mono[30];
  mono[172] = mono[14] * mono[30];
  mono[173] = mono[9] * mono[31];
  mono[174] = mono[11] * mono[31];
  mono[175] = mono[12] * mono[31];
  mono[176] = mono[14] * mono[31];
  mono[177] = mono[9] * mono[32];
  mono[178] = mono[10] * mono[32];
  mono[179] = mono[12] * mono[32];
  mono[180] = mono[13] * mono[32];
  mono[181] = mono[18] * mono[30];
  mono[182] = mono[19] * mono[30];
  mono[183] = mono[17] * mono[31];
  mono[184] = mono[19] * mono[31];
  mono[185] = mono[17] * mono[32];
  mono[186] = mono[18] * mono[32];
}

fn f_monomials3(mono: &mut [f64; N_MONOS]) { 
  mono[187] = mono[24] * mono[30];
  mono[188] = mono[25] * mono[30];
  mono[189] = mono[23] * mono[31];
  mono[190] = mono[25] * mono[31];
  mono[191] = mono[23] * mono[32];
  mono[192] = mono[24] * mono[32];
  mono[193] = mono[30] * mono[31];
  mono[194] = mono[30] * mono[32];
  mono[195] = mono[31] * mono[32];
  mono[196] = mono[8] * mono[33];
  mono[197] = mono[7] * mono[34];
  mono[198] = mono[12] * mono[33];
  mono[199] = mono[13] * mono[33];
  mono[200] = mono[14] * mono[33];
  mono[201] = mono[9] * mono[34];
  mono[202] = mono[10] * mono[34];
  mono[203] = mono[11] * mono[34];
  mono[204] = mono[21] * mono[33];
  mono[205] = mono[20] * mono[34];
  mono[206] = mono[27] * mono[33];
  mono[207] = mono[26] * mono[34];
  mono[208] = mono[33] * mono[34];
  mono[209] = mono[1] * mono[39];
  mono[210] = mono[1] * mono[40];
  mono[211] = mono[2] * mono[40];
  mono[212] = mono[1] * mono[41];
  mono[213] = mono[3] * mono[41];
  mono[214] = mono[2] * mono[42];
  mono[215] = mono[3] * mono[42];
  mono[216] = mono[2] * mono[43];
  mono[217] = mono[3] * mono[43];
  mono[218] = mono[1] * mono[44];
  mono[219] = mono[3] * mono[44];
  mono[220] = mono[1] * mono[45];
  mono[221] = mono[2] * mono[45];
  mono[222] = mono[4] * mono[45];
  mono[223] = mono[2] * mono[48];
  mono[224] = mono[1] * mono[50];
  mono[225] = mono[1] * mono[52];
  mono[226] = mono[2] * mono[54];
  mono[227] = mono[1] * mono[56];
  mono[228] = mono[1] * mono[58];
  mono[229] = mono[2] * mono[59];
  mono[230] = mono[3] * mono[59];
  mono[231] = mono[1] * mono[60];
  mono[232] = mono[3] * mono[60];
  mono[233] = mono[1] * mono[61];
  mono[234] = mono[2] * mono[61];
  mono[235] = mono[2] * mono[62];
  mono[236] = mono[3] * mono[62];
}

fn f_monomials4(mono: &mut [f64; N_MONOS]) { 
  mono[237] = mono[1] * mono[63];
  mono[238] = mono[3] * mono[63];
  mono[239] = mono[1] * mono[64];
  mono[240] = mono[2] * mono[64];
  mono[241] = mono[4] * mono[59];
  mono[242] = mono[5] * mono[59];
  mono[243] = mono[4] * mono[60];
  mono[244] = mono[6] * mono[60];
  mono[245] = mono[5] * mono[61];
  mono[246] = mono[6] * mono[61];
  mono[247] = mono[4] * mono[62];
  mono[248] = mono[5] * mono[62];
  mono[249] = mono[4] * mono[63];
  mono[250] = mono[6] * mono[63];
  mono[251] = mono[5] * mono[64];
  mono[252] = mono[6] * mono[64];
  mono[253] = mono[2] * mono[65];
  mono[254] = mono[3] * mono[65];
  mono[255] = mono[1] * mono[66];
  mono[256] = mono[3] * mono[66];
  mono[257] = mono[1] * mono[67];
  mono[258] = mono[2] * mono[67];
  mono[259] = mono[2] * mono[68];
  mono[260] = mono[3] * mono[68];
  mono[261] = mono[1] * mono[69];
  mono[262] = mono[3] * mono[69];
  mono[263] = mono[1] * mono[70];
  mono[264] = mono[2] * mono[70];
  mono[265] = mono[6] * mono[65];
  mono[266] = mono[5] * mono[66];
  mono[267] = mono[4] * mono[67];
  mono[268] = mono[6] * mono[68];
  mono[269] = mono[5] * mono[69];
  mono[270] = mono[4] * mono[70];
  mono[271] = mono[3] * mono[71];
  mono[272] = mono[2] * mono[72];
  mono[273] = mono[3] * mono[73];
  mono[274] = mono[1] * mono[74];
  mono[275] = mono[2] * mono[75];
  mono[276] = mono[1] * mono[76];
  mono[277] = mono[5] * mono[71];
  mono[278] = mono[6] * mono[71];
  mono[279] = mono[4] * mono[72];
  mono[280] = mono[6] * mono[72];
  mono[281] = mono[5] * mono[73];
  mono[282] = mono[6] * mono[73];
  mono[283] = mono[4] * mono[74];
  mono[284] = mono[5] * mono[74];
  mono[285] = mono[4] * mono[75];
  mono[286] = mono[6] * mono[75];
}

fn f_monomials5(mono: &mut [f64; N_MONOS]) { 
  mono[287] = mono[4] * mono[76];
  mono[288] = mono[5] * mono[76];
  mono[289] = mono[2] * mono[77];
  mono[290] = mono[3] * mono[77];
  mono[291] = mono[1] * mono[78];
  mono[292] = mono[3] * mono[78];
  mono[293] = mono[1] * mono[79];
  mono[294] = mono[2] * mono[79];
  mono[295] = mono[6] * mono[77];
  mono[296] = mono[5] * mono[78];
  mono[297] = mono[4] * mono[79];
  mono[298] = mono[3] * mono[80];
  mono[299] = mono[2] * mono[81];
  mono[300] = mono[1] * mono[82];
  mono[301] = mono[3] * mono[83];
  mono[302] = mono[2] * mono[84];
  mono[303] = mono[1] * mono[85];
  mono[304] = mono[5] * mono[80];
  mono[305] = mono[6] * mono[80];
  mono[306] = mono[4] * mono[81];
  mono[307] = mono[6] * mono[81];
  mono[308] = mono[4] * mono[82];
  mono[309] = mono[5] * mono[82];
  mono[310] = mono[5] * mono[83];
  mono[311] = mono[6] * mono[83];
  mono[312] = mono[4] * mono[84];
  mono[313] = mono[6] * mono[84];
  mono[314] = mono[4] * mono[85];
  mono[315] = mono[5] * mono[85];
  mono[316] = mono[8] * mono[80];
  mono[317] = mono[8] * mono[81];
  mono[318] = mono[8] * mono[82];
  mono[319] = mono[7] * mono[83];
  mono[320] = mono[7] * mono[84];
  mono[321] = mono[7] * mono[85];
  mono[322] = mono[10] * mono[72];
  mono[323] = mono[9] * mono[74];
  mono[324] = mono[11] * mono[83];
  mono[325] = mono[9] * mono[76];
  mono[326] = mono[10] * mono[84];
  mono[327] = mono[9] * mono[85];
  mono[328] = mono[9] * mono[71];
  mono[329] = mono[9] * mono[72];
  mono[330] = mono[9] * mono[78];
  mono[331] = mono[10] * mono[74];
  mono[332] = mono[9] * mono[83];
  mono[333] = mono[10] * mono[83];
  mono[334] = mono[9] * mono[79];
  mono[335] = mono[10] * mono[79];
  mono[336] = mono[9] * mono[84];
}

fn f_monomials6(mono: &mut [f64; N_MONOS]) { 
  mono[337] = mono[11] * mono[84];
  mono[338] = mono[10] * mono[85];
  mono[339] = mono[11] * mono[85];
  mono[340] = mono[9] * mono[82];
  mono[341] = mono[12] * mono[85];
  mono[342] = mono[2] * mono[87];
  mono[343] = mono[1] * mono[89];
  mono[344] = mono[1] * mono[91];
  mono[345] = mono[2] * mono[92];
  mono[346] = mono[3] * mono[92];
  mono[347] = mono[1] * mono[93];
  mono[348] = mono[3] * mono[93];
  mono[349] = mono[1] * mono[94];
  mono[350] = mono[2] * mono[94];
  mono[351] = mono[4] * mono[92];
  mono[352] = mono[5] * mono[92];
  mono[353] = mono[4] * mono[93];
  mono[354] = mono[6] * mono[93];
  mono[355] = mono[5] * mono[94];
  mono[356] = mono[6] * mono[94];
  mono[357] = mono[3] * mono[95];
  mono[358] = mono[2] * mono[96];
  mono[359] = mono[3] * mono[97];
  mono[360] = mono[2] * mono[98];
  mono[361] = mono[3] * mono[99];
  mono[362] = mono[1] * mono[100];
  mono[363] = mono[3] * mono[101];
  mono[364] = mono[1] * mono[102];
  mono[365] = mono[2] * mono[103];
  mono[366] = mono[1] * mono[104];
  mono[367] = mono[2] * mono[105];
  mono[368] = mono[1] * mono[106];
  mono[369] = mono[2] * mono[95];
  mono[370] = mono[3] * mono[96];
  mono[371] = mono[2] * mono[97];
  mono[372] = mono[3] * mono[98];
  mono[373] = mono[1] * mono[99];
  mono[374] = mono[3] * mono[100];
  mono[375] = mono[1] * mono[101];
  mono[376] = mono[3] * mono[102];
  mono[377] = mono[1] * mono[103];
  mono[378] = mono[2] * mono[104];
  mono[379] = mono[1] * mono[105];
  mono[380] = mono[2] * mono[106];
  mono[381] = mono[6] * mono[95];
  mono[382] = mono[6] * mono[96];
  mono[383] = mono[6] * mono[97];
  mono[384] = mono[6] * mono[98];
  mono[385] = mono[5] * mono[99];
  mono[386] = mono[5] * mono[100];
}

fn f_monomials7(mono: &mut [f64; N_MONOS]) { 
  mono[387] = mono[5] * mono[101];
  mono[388] = mono[5] * mono[102];
  mono[389] = mono[4] * mono[103];
  mono[390] = mono[4] * mono[104];
  mono[391] = mono[4] * mono[105];
  mono[392] = mono[4] * mono[106];
  mono[393] = mono[5] * mono[95];
  mono[394] = mono[4] * mono[96];
  mono[395] = mono[5] * mono[97];
  mono[396] = mono[4] * mono[98];
  mono[397] = mono[6] * mono[99];
  mono[398] = mono[4] * mono[100];
  mono[399] = mono[6] * mono[101];
  mono[400] = mono[4] * mono[102];
  mono[401] = mono[6] * mono[103];
  mono[402] = mono[5] * mono[104];
  mono[403] = mono[6] * mono[105];
  mono[404] = mono[5] * mono[106];
  mono[405] = mono[8] * mono[95];
  mono[406] = mono[8] * mono[96];
  mono[407] = mono[7] * mono[97];
  mono[408] = mono[7] * mono[98];
  mono[409] = mono[8] * mono[99];
  mono[410] = mono[8] * mono[100];
  mono[411] = mono[7] * mono[101];
  mono[412] = mono[7] * mono[102];
  mono[413] = mono[8] * mono[103];
  mono[414] = mono[8] * mono[104];
  mono[415] = mono[7] * mono[105];
  mono[416] = mono[7] * mono[106];
  mono[417] = mono[11] * mono[97];
  mono[418] = mono[10] * mono[98];
  mono[419] = mono[11] * mono[101];
  mono[420] = mono[9] * mono[102];
  mono[421] = mono[10] * mono[105];
  mono[422] = mono[9] * mono[106];
  mono[423] = mono[10] * mono[97];
  mono[424] = mono[11] * mono[98];
  mono[425] = mono[9] * mono[101];
  mono[426] = mono[11] * mono[102];
  mono[427] = mono[9] * mono[105];
  mono[428] = mono[10] * mono[106];
  mono[429] = mono[10] * mono[96];
  mono[430] = mono[13] * mono[98];
  mono[431] = mono[9] * mono[100];
  mono[432] = mono[12] * mono[102];
  mono[433] = mono[9] * mono[104];
  mono[434] = mono[12] * mono[106];
  mono[435] = mono[3] * mono[107];
  mono[436] = mono[2] * mono[108];
}

fn f_monomials8(mono: &mut [f64; N_MONOS]) { 
  mono[437] = mono[1] * mono[109];
  mono[438] = mono[5] * mono[107];
  mono[439] = mono[6] * mono[107];
  mono[440] = mono[4] * mono[108];
  mono[441] = mono[6] * mono[108];
  mono[442] = mono[4] * mono[109];
  mono[443] = mono[5] * mono[109];
  mono[444] = mono[11] * mono[107];
  mono[445] = mono[14] * mono[107];
  mono[446] = mono[10] * mono[108];
  mono[447] = mono[13] * mono[108];
  mono[448] = mono[9] * mono[109];
  mono[449] = mono[12] * mono[109];
  mono[450] = mono[17] * mono[109];
  mono[451] = mono[2] * mono[112];
  mono[452] = mono[3] * mono[112];
  mono[453] = mono[1] * mono[113];
  mono[454] = mono[3] * mono[113];
  mono[455] = mono[1] * mono[114];
  mono[456] = mono[2] * mono[114];
  mono[457] = mono[2] * mono[115];
  mono[458] = mono[3] * mono[115];
  mono[459] = mono[1] * mono[116];
  mono[460] = mono[3] * mono[116];
  mono[461] = mono[1] * mono[117];
  mono[462] = mono[2] * mono[117];
  mono[463] = mono[6] * mono[112];
  mono[464] = mono[5] * mono[113];
  mono[465] = mono[4] * mono[114];
  mono[466] = mono[6] * mono[115];
  mono[467] = mono[5] * mono[116];
  mono[468] = mono[4] * mono[117];
  mono[469] = mono[8] * mono[112];
  mono[470] = mono[8] * mono[113];
  mono[471] = mono[8] * mono[114];
  mono[472] = mono[7] * mono[115];
  mono[473] = mono[7] * mono[116];
  mono[474] = mono[7] * mono[117];
  mono[475] = mono[12] * mono[113];
  mono[476] = mono[12] * mono[114];
  mono[477] = mono[13] * mono[114];
  mono[478] = mono[9] * mono[116];
  mono[479] = mono[9] * mono[117];
  mono[480] = mono[10] * mono[117];
  mono[481] = mono[17] * mono[113];
  mono[482] = mono[17] * mono[114];
  mono[483] = mono[18] * mono[112];
  mono[484] = mono[18] * mono[114];
  mono[485] = mono[19] * mono[112];
  mono[486] = mono[19] * mono[113];
}

fn f_monomials9(mono: &mut [f64; N_MONOS]) { 
  mono[487] = mono[17] * mono[116];
  mono[488] = mono[17] * mono[117];
  mono[489] = mono[18] * mono[115];
  mono[490] = mono[18] * mono[117];
  mono[491] = mono[19] * mono[115];
  mono[492] = mono[19] * mono[116];
  mono[493] = mono[2] * mono[120];
  mono[494] = mono[1] * mono[122];
  mono[495] = mono[1] * mono[124];
  mono[496] = mono[2] * mono[125];
  mono[497] = mono[3] * mono[125];
  mono[498] = mono[1] * mono[126];
  mono[499] = mono[3] * mono[126];
  mono[500] = mono[1] * mono[127];
  mono[501] = mono[2] * mono[127];
  mono[502] = mono[4] * mono[125];
  mono[503] = mono[5] * mono[125];
  mono[504] = mono[4] * mono[126];
  mono[505] = mono[6] * mono[126];
  mono[506] = mono[5] * mono[127];
  mono[507] = mono[6] * mono[127];
  mono[508] = mono[3] * mono[128];
  mono[509] = mono[2] * mono[129];
  mono[510] = mono[3] * mono[130];
  mono[511] = mono[2] * mono[131];
  mono[512] = mono[3] * mono[132];
  mono[513] = mono[1] * mono[133];
  mono[514] = mono[3] * mono[134];
  mono[515] = mono[1] * mono[135];
  mono[516] = mono[2] * mono[136];
  mono[517] = mono[1] * mono[137];
  mono[518] = mono[2] * mono[138];
  mono[519] = mono[1] * mono[139];
  mono[520] = mono[2] * mono[128];
  mono[521] = mono[3] * mono[129];
  mono[522] = mono[2] * mono[130];
  mono[523] = mono[3] * mono[131];
  mono[524] = mono[1] * mono[132];
  mono[525] = mono[3] * mono[133];
  mono[526] = mono[1] * mono[134];
  mono[527] = mono[3] * mono[135];
  mono[528] = mono[1] * mono[136];
  mono[529] = mono[2] * mono[137];
  mono[530] = mono[1] * mono[138];
  mono[531] = mono[2] * mono[139];
  mono[532] = mono[6] * mono[128];
  mono[533] = mono[6] * mono[129];
  mono[534] = mono[6] * mono[130];
  mono[535] = mono[6] * mono[131];
  mono[536] = mono[5] * mono[132];
}

fn f_monomials10(mono: &mut [f64; N_MONOS]) { 
  mono[537] = mono[5] * mono[133];
  mono[538] = mono[5] * mono[134];
  mono[539] = mono[5] * mono[135];
  mono[540] = mono[4] * mono[136];
  mono[541] = mono[4] * mono[137];
  mono[542] = mono[4] * mono[138];
  mono[543] = mono[4] * mono[139];
  mono[544] = mono[5] * mono[128];
  mono[545] = mono[4] * mono[129];
  mono[546] = mono[5] * mono[130];
  mono[547] = mono[4] * mono[131];
  mono[548] = mono[6] * mono[132];
  mono[549] = mono[4] * mono[133];
  mono[550] = mono[6] * mono[134];
  mono[551] = mono[4] * mono[135];
  mono[552] = mono[6] * mono[136];
  mono[553] = mono[5] * mono[137];
  mono[554] = mono[6] * mono[138];
  mono[555] = mono[5] * mono[139];
  mono[556] = mono[8] * mono[128];
  mono[557] = mono[8] * mono[129];
  mono[558] = mono[7] * mono[130];
  mono[559] = mono[7] * mono[131];
  mono[560] = mono[8] * mono[132];
  mono[561] = mono[8] * mono[133];
  mono[562] = mono[7] * mono[134];
  mono[563] = mono[7] * mono[135];
  mono[564] = mono[8] * mono[136];
  mono[565] = mono[8] * mono[137];
  mono[566] = mono[7] * mono[138];
  mono[567] = mono[7] * mono[139];
  mono[568] = mono[11] * mono[130];
  mono[569] = mono[10] * mono[131];
  mono[570] = mono[11] * mono[134];
  mono[571] = mono[9] * mono[135];
  mono[572] = mono[10] * mono[138];
  mono[573] = mono[9] * mono[139];
  mono[574] = mono[10] * mono[130];
  mono[575] = mono[11] * mono[131];
  mono[576] = mono[9] * mono[134];
  mono[577] = mono[11] * mono[135];
  mono[578] = mono[9] * mono[138];
  mono[579] = mono[10] * mono[139];
  mono[580] = mono[10] * mono[129];
  mono[581] = mono[13] * mono[131];
  mono[582] = mono[9] * mono[133];
  mono[583] = mono[12] * mono[135];
  mono[584] = mono[9] * mono[137];
  mono[585] = mono[12] * mono[139];
  mono[586] = mono[3] * mono[140];
}

fn f_monomials11(mono: &mut [f64; N_MONOS]) { 
  mono[587] = mono[2] * mono[141];
  mono[588] = mono[3] * mono[142];
  mono[589] = mono[1] * mono[143];
  mono[590] = mono[2] * mono[144];
  mono[591] = mono[1] * mono[145];
  mono[592] = mono[2] * mono[140];
  mono[593] = mono[3] * mono[141];
  mono[594] = mono[1] * mono[142];
  mono[595] = mono[3] * mono[143];
  mono[596] = mono[1] * mono[144];
  mono[597] = mono[2] * mono[145];
  mono[598] = mono[6] * mono[140];
  mono[599] = mono[6] * mono[141];
  mono[600] = mono[5] * mono[142];
  mono[601] = mono[5] * mono[143];
  mono[602] = mono[4] * mono[144];
  mono[603] = mono[4] * mono[145];
  mono[604] = mono[5] * mono[140];
  mono[605] = mono[4] * mono[141];
  mono[606] = mono[6] * mono[142];
  mono[607] = mono[4] * mono[143];
  mono[608] = mono[6] * mono[144];
  mono[609] = mono[5] * mono[145];
  mono[610] = mono[11] * mono[140];
  mono[611] = mono[14] * mono[140];
  mono[612] = mono[10] * mono[141];
  mono[613] = mono[13] * mono[141];
  mono[614] = mono[11] * mono[142];
  mono[615] = mono[14] * mono[142];
  mono[616] = mono[9] * mono[143];
  mono[617] = mono[12] * mono[143];
  mono[618] = mono[10] * mono[144];
  mono[619] = mono[13] * mono[144];
  mono[620] = mono[9] * mono[145];
  mono[621] = mono[12] * mono[145];
  mono[622] = mono[10] * mono[140];
  mono[623] = mono[13] * mono[140];
  mono[624] = mono[11] * mono[141];
  mono[625] = mono[14] * mono[141];
  mono[626] = mono[9] * mono[142];
  mono[627] = mono[12] * mono[142];
  mono[628] = mono[11] * mono[143];
  mono[629] = mono[14] * mono[143];
  mono[630] = mono[9] * mono[144];
  mono[631] = mono[12] * mono[144];
  mono[632] = mono[10] * mono[145];
  mono[633] = mono[13] * mono[145];
  mono[634] = mono[18] * mono[141];
  mono[635] = mono[17] * mono[143];
  mono[636] = mono[17] * mono[145];
}

fn f_monomials12(mono: &mut [f64; N_MONOS]) { 
  mono[637] = mono[20] * mono[130];
  mono[638] = mono[20] * mono[131];
  mono[639] = mono[21] * mono[128];
  mono[640] = mono[21] * mono[129];
  mono[641] = mono[20] * mono[134];
  mono[642] = mono[20] * mono[135];
  mono[643] = mono[21] * mono[132];
  mono[644] = mono[21] * mono[133];
  mono[645] = mono[20] * mono[138];
  mono[646] = mono[20] * mono[139];
  mono[647] = mono[21] * mono[136];
  mono[648] = mono[21] * mono[137];
  mono[649] = mono[3] * mono[146];
  mono[650] = mono[2] * mono[147];
  mono[651] = mono[1] * mono[148];
  mono[652] = mono[5] * mono[146];
  mono[653] = mono[6] * mono[146];
  mono[654] = mono[4] * mono[147];
  mono[655] = mono[6] * mono[147];
  mono[656] = mono[4] * mono[148];
  mono[657] = mono[5] * mono[148];
  mono[658] = mono[11] * mono[146];
  mono[659] = mono[14] * mono[146];
  mono[660] = mono[10] * mono[147];
  mono[661] = mono[13] * mono[147];
  mono[662] = mono[9] * mono[148];
  mono[663] = mono[12] * mono[148];
  mono[664] = mono[19] * mono[146];
  mono[665] = mono[18] * mono[147];
  mono[666] = mono[17] * mono[148];
  mono[667] = mono[23] * mono[148];
  mono[668] = mono[2] * mono[151];
  mono[669] = mono[3] * mono[151];
  mono[670] = mono[1] * mono[152];
  mono[671] = mono[3] * mono[152];
  mono[672] = mono[1] * mono[153];
  mono[673] = mono[2] * mono[153];
  mono[674] = mono[2] * mono[154];
  mono[675] = mono[3] * mono[154];
  mono[676] = mono[1] * mono[155];
  mono[677] = mono[3] * mono[155];
  mono[678] = mono[1] * mono[156];
  mono[679] = mono[2] * mono[156];
  mono[680] = mono[6] * mono[151];
  mono[681] = mono[5] * mono[152];
  mono[682] = mono[4] * mono[153];
  mono[683] = mono[6] * mono[154];
  mono[684] = mono[5] * mono[155];
  mono[685] = mono[4] * mono[156];
  mono[686] = mono[8] * mono[151];
}

fn f_monomials13(mono: &mut [f64; N_MONOS]) { 
  mono[687] = mono[8] * mono[152];
  mono[688] = mono[8] * mono[153];
  mono[689] = mono[7] * mono[154];
  mono[690] = mono[7] * mono[155];
  mono[691] = mono[7] * mono[156];
  mono[692] = mono[12] * mono[152];
  mono[693] = mono[12] * mono[153];
  mono[694] = mono[13] * mono[153];
  mono[695] = mono[9] * mono[155];
  mono[696] = mono[9] * mono[156];
  mono[697] = mono[10] * mono[156];
  mono[698] = mono[17] * mono[152];
  mono[699] = mono[17] * mono[153];
  mono[700] = mono[18] * mono[151];
  mono[701] = mono[18] * mono[153];
  mono[702] = mono[19] * mono[151];
  mono[703] = mono[19] * mono[152];
  mono[704] = mono[17] * mono[155];
  mono[705] = mono[17] * mono[156];
  mono[706] = mono[18] * mono[154];
  mono[707] = mono[18] * mono[156];
  mono[708] = mono[19] * mono[154];
  mono[709] = mono[19] * mono[155];
  mono[710] = mono[8] * mono[157];
  mono[711] = mono[7] * mono[158];
  mono[712] = mono[12] * mono[157];
  mono[713] = mono[13] * mono[157];
  mono[714] = mono[14] * mono[157];
  mono[715] = mono[9] * mono[158];
  mono[716] = mono[10] * mono[158];
  mono[717] = mono[11] * mono[158];
  mono[718] = mono[23] * mono[152];
  mono[719] = mono[23] * mono[153];
  mono[720] = mono[24] * mono[151];
  mono[721] = mono[24] * mono[153];
  mono[722] = mono[25] * mono[151];
  mono[723] = mono[25] * mono[152];
  mono[724] = mono[23] * mono[155];
  mono[725] = mono[23] * mono[156];
  mono[726] = mono[24] * mono[154];
  mono[727] = mono[24] * mono[156];
  mono[728] = mono[25] * mono[154];
  mono[729] = mono[25] * mono[155];
  mono[730] = mono[2] * mono[161];
  mono[731] = mono[1] * mono[163];
  mono[732] = mono[1] * mono[165];
  mono[733] = mono[2] * mono[166];
  mono[734] = mono[3] * mono[166];
  mono[735] = mono[1] * mono[167];
  mono[736] = mono[3] * mono[167];
}

fn f_monomials14(mono: &mut [f64; N_MONOS]) { 
  mono[737] = mono[1] * mono[168];
  mono[738] = mono[2] * mono[168];
  mono[739] = mono[4] * mono[166];
  mono[740] = mono[5] * mono[166];
  mono[741] = mono[4] * mono[167];
  mono[742] = mono[6] * mono[167];
  mono[743] = mono[5] * mono[168];
  mono[744] = mono[6] * mono[168];
  mono[745] = mono[3] * mono[169];
  mono[746] = mono[2] * mono[170];
  mono[747] = mono[3] * mono[171];
  mono[748] = mono[2] * mono[172];
  mono[749] = mono[3] * mono[173];
  mono[750] = mono[1] * mono[174];
  mono[751] = mono[3] * mono[175];
  mono[752] = mono[1] * mono[176];
  mono[753] = mono[2] * mono[177];
  mono[754] = mono[1] * mono[178];
  mono[755] = mono[2] * mono[179];
  mono[756] = mono[1] * mono[180];
  mono[757] = mono[2] * mono[169];
  mono[758] = mono[3] * mono[170];
  mono[759] = mono[2] * mono[171];
  mono[760] = mono[3] * mono[172];
  mono[761] = mono[1] * mono[173];
  mono[762] = mono[3] * mono[174];
  mono[763] = mono[1] * mono[175];
  mono[764] = mono[3] * mono[176];
  mono[765] = mono[1] * mono[177];
  mono[766] = mono[2] * mono[178];
  mono[767] = mono[1] * mono[179];
  mono[768] = mono[2] * mono[180];
  mono[769] = mono[6] * mono[169];
  mono[770] = mono[6] * mono[170];
  mono[771] = mono[6] * mono[171];
  mono[772] = mono[6] * mono[172];
  mono[773] = mono[5] * mono[173];
  mono[774] = mono[5] * mono[174];
  mono[775] = mono[5] * mono[175];
  mono[776] = mono[5] * mono[176];
  mono[777] = mono[4] * mono[177];
  mono[778] = mono[4] * mono[178];
  mono[779] = mono[4] * mono[179];
  mono[780] = mono[4] * mono[180];
  mono[781] = mono[5] * mono[169];
  mono[782] = mono[4] * mono[170];
  mono[783] = mono[5] * mono[171];
  mono[784] = mono[4] * mono[172];
  mono[785] = mono[6] * mono[173];
  mono[786] = mono[4] * mono[174];
}

fn f_monomials15(mono: &mut [f64; N_MONOS]) { 
  mono[787] = mono[6] * mono[175];
  mono[788] = mono[4] * mono[176];
  mono[789] = mono[6] * mono[177];
  mono[790] = mono[5] * mono[178];
  mono[791] = mono[6] * mono[179];
  mono[792] = mono[5] * mono[180];
  mono[793] = mono[8] * mono[169];
  mono[794] = mono[8] * mono[170];
  mono[795] = mono[7] * mono[171];
  mono[796] = mono[7] * mono[172];
  mono[797] = mono[8] * mono[173];
  mono[798] = mono[8] * mono[174];
  mono[799] = mono[7] * mono[175];
  mono[800] = mono[7] * mono[176];
  mono[801] = mono[8] * mono[177];
  mono[802] = mono[8] * mono[178];
  mono[803] = mono[7] * mono[179];
  mono[804] = mono[7] * mono[180];
  mono[805] = mono[11] * mono[171];
  mono[806] = mono[10] * mono[172];
  mono[807] = mono[11] * mono[175];
  mono[808] = mono[9] * mono[176];
  mono[809] = mono[10] * mono[179];
  mono[810] = mono[9] * mono[180];
  mono[811] = mono[10] * mono[171];
  mono[812] = mono[11] * mono[172];
  mono[813] = mono[9] * mono[175];
  mono[814] = mono[11] * mono[176];
  mono[815] = mono[9] * mono[179];
  mono[816] = mono[10] * mono[180];
  mono[817] = mono[10] * mono[170];
  mono[818] = mono[13] * mono[172];
  mono[819] = mono[9] * mono[174];
  mono[820] = mono[12] * mono[176];
  mono[821] = mono[9] * mono[178];
  mono[822] = mono[12] * mono[180];
  mono[823] = mono[3] * mono[181];
  mono[824] = mono[2] * mono[182];
  mono[825] = mono[3] * mono[183];
  mono[826] = mono[1] * mono[184];
  mono[827] = mono[2] * mono[185];
  mono[828] = mono[1] * mono[186];
  mono[829] = mono[2] * mono[181];
  mono[830] = mono[3] * mono[182];
  mono[831] = mono[1] * mono[183];
  mono[832] = mono[3] * mono[184];
  mono[833] = mono[1] * mono[185];
  mono[834] = mono[2] * mono[186];
  mono[835] = mono[6] * mono[181];
  mono[836] = mono[6] * mono[182];
}

fn f_monomials16(mono: &mut [f64; N_MONOS]) { 
  mono[837] = mono[5] * mono[183];
  mono[838] = mono[5] * mono[184];
  mono[839] = mono[4] * mono[185];
  mono[840] = mono[4] * mono[186];
  mono[841] = mono[5] * mono[181];
  mono[842] = mono[4] * mono[182];
  mono[843] = mono[6] * mono[183];
  mono[844] = mono[4] * mono[184];
  mono[845] = mono[6] * mono[185];
  mono[846] = mono[5] * mono[186];
  mono[847] = mono[11] * mono[181];
  mono[848] = mono[14] * mono[181];
  mono[849] = mono[10] * mono[182];
  mono[850] = mono[13] * mono[182];
  mono[851] = mono[11] * mono[183];
  mono[852] = mono[14] * mono[183];
  mono[853] = mono[9] * mono[184];
  mono[854] = mono[12] * mono[184];
  mono[855] = mono[10] * mono[185];
  mono[856] = mono[13] * mono[185];
  mono[857] = mono[9] * mono[186];
  mono[858] = mono[12] * mono[186];
  mono[859] = mono[10] * mono[181];
  mono[860] = mono[13] * mono[181];
  mono[861] = mono[11] * mono[182];
  mono[862] = mono[14] * mono[182];
  mono[863] = mono[9] * mono[183];
  mono[864] = mono[12] * mono[183];
  mono[865] = mono[11] * mono[184];
  mono[866] = mono[14] * mono[184];
  mono[867] = mono[9] * mono[185];
  mono[868] = mono[12] * mono[185];
  mono[869] = mono[10] * mono[186];
  mono[870] = mono[13] * mono[186];
  mono[871] = mono[18] * mono[182];
  mono[872] = mono[17] * mono[184];
  mono[873] = mono[17] * mono[186];
  mono[874] = mono[20] * mono[171];
  mono[875] = mono[20] * mono[172];
  mono[876] = mono[21] * mono[169];
  mono[877] = mono[21] * mono[170];
  mono[878] = mono[20] * mono[175];
  mono[879] = mono[20] * mono[176];
  mono[880] = mono[21] * mono[173];
  mono[881] = mono[21] * mono[174];
  mono[882] = mono[20] * mono[179];
  mono[883] = mono[20] * mono[180];
  mono[884] = mono[21] * mono[177];
  mono[885] = mono[21] * mono[178];
  mono[886] = mono[3] * mono[187];
}

fn f_monomials17(mono: &mut [f64; N_MONOS]) { 
  mono[887] = mono[2] * mono[188];
  mono[888] = mono[3] * mono[189];
  mono[889] = mono[1] * mono[190];
  mono[890] = mono[2] * mono[191];
  mono[891] = mono[1] * mono[192];
  mono[892] = mono[2] * mono[187];
  mono[893] = mono[3] * mono[188];
  mono[894] = mono[1] * mono[189];
  mono[895] = mono[3] * mono[190];
  mono[896] = mono[1] * mono[191];
  mono[897] = mono[2] * mono[192];
  mono[898] = mono[6] * mono[187];
  mono[899] = mono[6] * mono[188];
  mono[900] = mono[5] * mono[189];
  mono[901] = mono[5] * mono[190];
  mono[902] = mono[4] * mono[191];
  mono[903] = mono[4] * mono[192];
  mono[904] = mono[5] * mono[187];
  mono[905] = mono[4] * mono[188];
  mono[906] = mono[6] * mono[189];
  mono[907] = mono[4] * mono[190];
  mono[908] = mono[6] * mono[191];
  mono[909] = mono[5] * mono[192];
  mono[910] = mono[11] * mono[187];
  mono[911] = mono[14] * mono[187];
  mono[912] = mono[10] * mono[188];
  mono[913] = mono[13] * mono[188];
  mono[914] = mono[11] * mono[189];
  mono[915] = mono[14] * mono[189];
  mono[916] = mono[9] * mono[190];
  mono[917] = mono[12] * mono[190];
  mono[918] = mono[10] * mono[191];
  mono[919] = mono[13] * mono[191];
  mono[920] = mono[9] * mono[192];
  mono[921] = mono[12] * mono[192];
  mono[922] = mono[10] * mono[187];
  mono[923] = mono[13] * mono[187];
  mono[924] = mono[11] * mono[188];
  mono[925] = mono[14] * mono[188];
  mono[926] = mono[9] * mono[189];
  mono[927] = mono[12] * mono[189];
  mono[928] = mono[11] * mono[190];
  mono[929] = mono[14] * mono[190];
  mono[930] = mono[9] * mono[191];
  mono[931] = mono[12] * mono[191];
  mono[932] = mono[10] * mono[192];
  mono[933] = mono[13] * mono[192];
  mono[934] = mono[19] * mono[187];
  mono[935] = mono[18] * mono[188];
  mono[936] = mono[19] * mono[189];
}

fn f_monomials18(mono: &mut [f64; N_MONOS]) { 
  mono[937] = mono[17] * mono[190];
  mono[938] = mono[18] * mono[191];
  mono[939] = mono[17] * mono[192];
  mono[940] = mono[18] * mono[187];
  mono[941] = mono[19] * mono[188];
  mono[942] = mono[17] * mono[189];
  mono[943] = mono[19] * mono[190];
  mono[944] = mono[17] * mono[191];
  mono[945] = mono[18] * mono[192];
  mono[946] = mono[24] * mono[188];
  mono[947] = mono[23] * mono[190];
  mono[948] = mono[23] * mono[192];
  mono[949] = mono[26] * mono[171];
  mono[950] = mono[26] * mono[172];
  mono[951] = mono[27] * mono[169];
  mono[952] = mono[27] * mono[170];
  mono[953] = mono[26] * mono[175];
  mono[954] = mono[26] * mono[176];
  mono[955] = mono[27] * mono[173];
  mono[956] = mono[27] * mono[174];
  mono[957] = mono[26] * mono[179];
  mono[958] = mono[26] * mono[180];
  mono[959] = mono[27] * mono[177];
  mono[960] = mono[27] * mono[178];
  mono[961] = mono[3] * mono[193];
  mono[962] = mono[2] * mono[194];
  mono[963] = mono[1] * mono[195];
  mono[964] = mono[5] * mono[193];
  mono[965] = mono[6] * mono[193];
  mono[966] = mono[4] * mono[194];
  mono[967] = mono[6] * mono[194];
  mono[968] = mono[4] * mono[195];
  mono[969] = mono[5] * mono[195];
  mono[970] = mono[11] * mono[193];
  mono[971] = mono[14] * mono[193];
  mono[972] = mono[10] * mono[194];
  mono[973] = mono[13] * mono[194];
  mono[974] = mono[9] * mono[195];
  mono[975] = mono[12] * mono[195];
  mono[976] = mono[19] * mono[193];
  mono[977] = mono[18] * mono[194];
  mono[978] = mono[17] * mono[195];
  mono[979] = mono[25] * mono[193];
  mono[980] = mono[24] * mono[194];
  mono[981] = mono[23] * mono[195];
  mono[982] = mono[30] * mono[195];
  mono[983] = mono[2] * mono[198];
  mono[984] = mono[3] * mono[198];
  mono[985] = mono[1] * mono[199];
  mono[986] = mono[3] * mono[199];
}

fn f_monomials19(mono: &mut [f64; N_MONOS]) { 
  mono[987] = mono[1] * mono[200];
  mono[988] = mono[2] * mono[200];
  mono[989] = mono[2] * mono[201];
  mono[990] = mono[3] * mono[201];
  mono[991] = mono[1] * mono[202];
  mono[992] = mono[3] * mono[202];
  mono[993] = mono[1] * mono[203];
  mono[994] = mono[2] * mono[203];
  mono[995] = mono[6] * mono[198];
  mono[996] = mono[5] * mono[199];
  mono[997] = mono[4] * mono[200];
  mono[998] = mono[6] * mono[201];
  mono[999] = mono[5] * mono[202];
  mono[1000] = mono[4] * mono[203];
  mono[1001] = mono[8] * mono[198];
  mono[1002] = mono[8] * mono[199];
  mono[1003] = mono[8] * mono[200];
  mono[1004] = mono[7] * mono[201];
  mono[1005] = mono[7] * mono[202];
  mono[1006] = mono[7] * mono[203];
  mono[1007] = mono[12] * mono[199];
  mono[1008] = mono[12] * mono[200];
  mono[1009] = mono[13] * mono[200];
  mono[1010] = mono[9] * mono[202];
  mono[1011] = mono[9] * mono[203];
  mono[1012] = mono[10] * mono[203];
  mono[1013] = mono[17] * mono[199];
  mono[1014] = mono[17] * mono[200];
  mono[1015] = mono[18] * mono[198];
  mono[1016] = mono[18] * mono[200];
  mono[1017] = mono[19] * mono[198];
  mono[1018] = mono[19] * mono[199];
  mono[1019] = mono[17] * mono[202];
  mono[1020] = mono[17] * mono[203];
  mono[1021] = mono[18] * mono[201];
  mono[1022] = mono[18] * mono[203];
  mono[1023] = mono[19] * mono[201];
  mono[1024] = mono[19] * mono[202];
  mono[1025] = mono[8] * mono[204];
  mono[1026] = mono[7] * mono[205];
  mono[1027] = mono[12] * mono[204];
  mono[1028] = mono[13] * mono[204];
  mono[1029] = mono[14] * mono[204];
  mono[1030] = mono[9] * mono[205];
  mono[1031] = mono[10] * mono[205];
  mono[1032] = mono[11] * mono[205];
  mono[1033] = mono[23] * mono[199];
  mono[1034] = mono[23] * mono[200];
  mono[1035] = mono[24] * mono[198];
  mono[1036] = mono[24] * mono[200];
}

fn f_monomials20(mono: &mut [f64; N_MONOS]) { 
  mono[1037] = mono[25] * mono[198];
  mono[1038] = mono[25] * mono[199];
  mono[1039] = mono[23] * mono[202];
  mono[1040] = mono[23] * mono[203];
  mono[1041] = mono[24] * mono[201];
  mono[1042] = mono[24] * mono[203];
  mono[1043] = mono[25] * mono[201];
  mono[1044] = mono[25] * mono[202];
  mono[1045] = mono[8] * mono[206];
  mono[1046] = mono[7] * mono[207];
  mono[1047] = mono[12] * mono[206];
  mono[1048] = mono[13] * mono[206];
  mono[1049] = mono[14] * mono[206];
  mono[1050] = mono[9] * mono[207];
  mono[1051] = mono[10] * mono[207];
  mono[1052] = mono[11] * mono[207];
  mono[1053] = mono[21] * mono[206];
  mono[1054] = mono[20] * mono[207];
  mono[1055] = mono[30] * mono[199];
  mono[1056] = mono[30] * mono[200];
  mono[1057] = mono[31] * mono[198];
  mono[1058] = mono[31] * mono[200];
  mono[1059] = mono[32] * mono[198];
  mono[1060] = mono[32] * mono[199];
  mono[1061] = mono[30] * mono[202];
  mono[1062] = mono[30] * mono[203];
  mono[1063] = mono[31] * mono[201];
  mono[1064] = mono[31] * mono[203];
  mono[1065] = mono[32] * mono[201];
  mono[1066] = mono[32] * mono[202];
}

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[35];
  mono[2] = r[34];
  mono[3] = r[32];
  mono[4] = r[33];
  mono[5] = r[31];
  mono[6] = r[30];
  mono[7] = r[29];
  mono[8] = r[25];
  mono[9] = r[28];
  mono[10] = r[27];
  mono[11] = r[26];
  mono[12] = r[24];
  mono[13] = r[23];
  mono[14] = r[22];
  mono[15] = r[21];
  mono[16] = r[20];
  mono[17] = r[19];
  mono[18] = r[18];
  mono[19] = r[17];
  mono[20] = r[16];
  mono[21] = r[15];
  mono[22] = r[14];
  mono[23] = r[13];
  mono[24] = r[12];
  mono[25] = r[11];
  mono[26] = r[10];
  mono[27] = r[9];
  mono[28] = r[8];
  mono[29] = r[7];
  mono[30] = r[6];
  mono[31] = r[5];
  mono[32] = r[4];
  mono[33] = r[3];
  mono[34] = r[2];
  mono[35] = r[1];
  mono[36] = r[0];
}

pub fn f_monomials(r: &[f64; N_DISTANCES]) -> [f64; N_MONOS] {

  let mut mono = [0.0; N_MONOS];

  f_init_monomials(&mut mono, r);

  f_monomials0(&mut mono);
  f_monomials1(&mut mono);
  f_monomials2(&mut mono);
  f_monomials3(&mut mono);
  f_monomials4(&mut mono);
  f_monomials5(&mut mono);
  f_monomials6(&mut mono);
  f_monomials7(&mut mono);
  f_monomials8(&mut mono);
  f_monomials9(&mut mono);
  f_monomials10(&mut mono);
  f_monomials11(&mut mono);
  f_monomials12(&mut mono);
  f_monomials13(&mut mono);
  f_monomials14(&mut mono);
  f_monomials15(&mut mono);
  f_monomials16(&mut mono);
  f_monomials17(&mut mono);
  f_monomials18(&mut mono);
  f_monomials19(&mut mono);
  f_monomials20(&mut mono);

  return mono; 
}

