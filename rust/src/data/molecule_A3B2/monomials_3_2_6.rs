#![allow(unused_variables)]



// File created from data/molecule_A3B2/MOL_3_2_6.MONO 
pub const N_MONOS: usize = 390;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 390 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

  mono[0] = 1.0;
  mono[1] = r[9];
  mono[2] = r[8];
  mono[3] = r[7];
  mono[4] = r[6];
  mono[5] = r[5];
  mono[6] = r[3];
  mono[7] = r[2];
  mono[8] = r[4];
  mono[9] = r[1];
  mono[10] = r[0];
  mono[11] = mono[3] * mono[4];
  mono[12] = mono[2] * mono[5];
  mono[13] = mono[3] * mono[6];
  mono[14] = mono[5] * mono[6];
  mono[15] = mono[2] * mono[7];
  mono[16] = mono[4] * mono[7];
  mono[17] = mono[2] * mono[4];
  mono[18] = mono[3] * mono[5];
  mono[19] = mono[2] * mono[6];
  mono[20] = mono[4] * mono[6];
  mono[21] = mono[3] * mono[7];
  mono[22] = mono[5] * mono[7];
  mono[23] = mono[2] * mono[3];
  mono[24] = mono[4] * mono[5];
  mono[25] = mono[6] * mono[7];
  mono[26] = mono[6] * mono[8];
  mono[27] = mono[7] * mono[8];
  mono[28] = mono[4] * mono[9];
  mono[29] = mono[5] * mono[9];
  mono[30] = mono[2] * mono[10];
  mono[31] = mono[3] * mono[10];
  mono[32] = mono[8] * mono[9];
  mono[33] = mono[8] * mono[10];
  mono[34] = mono[9] * mono[10];
  mono[35] = mono[3] * mono[20];
  mono[36] = mono[2] * mono[14];
  mono[37] = mono[3] * mono[14];
  mono[38] = mono[2] * mono[16];
  mono[39] = mono[3] * mono[16];
  mono[40] = mono[2] * mono[22];
  mono[41] = mono[2] * mono[20];
  mono[42] = mono[3] * mono[22];
  mono[43] = mono[2] * mono[11];
  mono[44] = mono[2] * mono[18];
  mono[45] = mono[2] * mono[24];
  mono[46] = mono[3] * mono[24];
  mono[47] = mono[2] * mono[13];
  mono[48] = mono[4] * mono[14];
  mono[49] = mono[2] * mono[21];
  mono[50] = mono[4] * mono[22];
  mono[51] = mono[2] * mono[25];
  mono[52] = mono[3] * mono[25];
  mono[53] = mono[4] * mono[25];
  mono[54] = mono[5] * mono[25];
  mono[55] = mono[6] * mono[27];
  mono[56] = mono[4] * mono[29];
  mono[57] = mono[2] * mono[31];
  mono[58] = mono[3] * mono[26];
  mono[59] = mono[5] * mono[26];
  mono[60] = mono[2] * mono[27];
  mono[61] = mono[4] * mono[27];
  mono[62] = mono[3] * mono[28];
  mono[63] = mono[2] * mono[29];
  mono[64] = mono[6] * mono[29];
  mono[65] = mono[7] * mono[28];
  mono[66] = mono[4] * mono[31];
  mono[67] = mono[5] * mono[30];
  mono[68] = mono[6] * mono[31];
  mono[69] = mono[7] * mono[30];
  mono[70] = mono[2] * mono[26];
  mono[71] = mono[4] * mono[26];
  mono[72] = mono[3] * mono[27];
  mono[73] = mono[5] * mono[27];
  mono[74] = mono[2] * mono[28];
  mono[75] = mono[3] * mono[29];
  mono[76] = mono[6] * mono[28];
  mono[77] = mono[7] * mono[29];
  mono[78] = mono[4] * mono[30];
  mono[79] = mono[5] * mono[31];
  mono[80] = mono[6] * mono[30];
  mono[81] = mono[7] * mono[31];
  mono[82] = mono[4] * mono[32];
  mono[83] = mono[5] * mono[32];
  mono[84] = mono[6] * mono[32];
  mono[85] = mono[7] * mono[32];
  mono[86] = mono[2] * mono[33];
  mono[87] = mono[3] * mono[33];
  mono[88] = mono[6] * mono[33];
  mono[89] = mono[7] * mono[33];
  mono[90] = mono[2] * mono[34];
  mono[91] = mono[3] * mono[34];
  mono[92] = mono[4] * mono[34];
  mono[93] = mono[5] * mono[34];
  mono[94] = mono[8] * mono[34];
  mono[95] = mono[2] * mono[37];
  mono[96] = mono[3] * mono[48];
  mono[97] = mono[2] * mono[39];
  mono[98] = mono[2] * mono[50];
  mono[99] = mono[3] * mono[53];
  mono[100] = mono[2] * mono[54];
  mono[101] = mono[2] * mono[35];
  mono[102] = mono[2] * mono[48];
  mono[103] = mono[2] * mono[42];
  mono[104] = mono[3] * mono[50];
  mono[105] = mono[2] * mono[53];
  mono[106] = mono[3] * mono[54];
  mono[107] = mono[2] * mono[46];
  mono[108] = mono[2] * mono[52];
  mono[109] = mono[4] * mono[54];
  mono[110] = mono[2] * mono[55];
  mono[111] = mono[3] * mono[55];
  mono[112] = mono[4] * mono[55];
  mono[113] = mono[5] * mono[55];
  mono[114] = mono[2] * mono[56];
  mono[115] = mono[3] * mono[56];
  mono[116] = mono[4] * mono[64];
  mono[117] = mono[4] * mono[77];
  mono[118] = mono[2] * mono[66];
  mono[119] = mono[2] * mono[79];
  mono[120] = mono[2] * mono[68];
  mono[121] = mono[2] * mono[81];
  mono[122] = mono[3] * mono[71];
  mono[123] = mono[2] * mono[59];
  mono[124] = mono[3] * mono[61];
  mono[125] = mono[2] * mono[73];
  mono[126] = mono[3] * mono[76];
  mono[127] = mono[3] * mono[64];
  mono[128] = mono[2] * mono[65];
  mono[129] = mono[2] * mono[77];
  mono[130] = mono[5] * mono[80];
  mono[131] = mono[5] * mono[68];
  mono[132] = mono[4] * mono[69];
  mono[133] = mono[4] * mono[81];
  mono[134] = mono[2] * mono[58];
  mono[135] = mono[4] * mono[59];
  mono[136] = mono[2] * mono[72];
  mono[137] = mono[4] * mono[73];
  mono[138] = mono[2] * mono[62];
  mono[139] = mono[2] * mono[75];
  mono[140] = mono[6] * mono[65];
  mono[141] = mono[6] * mono[77];
  mono[142] = mono[4] * mono[67];
  mono[143] = mono[4] * mono[79];
  mono[144] = mono[6] * mono[69];
  mono[145] = mono[6] * mono[81];
  mono[146] = mono[5] * mono[84];
  mono[147] = mono[4] * mono[85];
  mono[148] = mono[3] * mono[88];
  mono[149] = mono[2] * mono[89];
  mono[150] = mono[3] * mono[92];
  mono[151] = mono[2] * mono[93];
  mono[152] = mono[4] * mono[84];
  mono[153] = mono[5] * mono[85];
  mono[154] = mono[2] * mono[88];
  mono[155] = mono[3] * mono[89];
  mono[156] = mono[2] * mono[92];
  mono[157] = mono[3] * mono[93];
  mono[158] = mono[4] * mono[83];
  mono[159] = mono[6] * mono[85];
  mono[160] = mono[2] * mono[87];
  mono[161] = mono[6] * mono[89];
  mono[162] = mono[2] * mono[91];
  mono[163] = mono[4] * mono[93];
  mono[164] = mono[3] * mono[35];
  mono[165] = mono[5] * mono[36];
  mono[166] = mono[6] * mono[37];
  mono[167] = mono[4] * mono[39];
  mono[168] = mono[2] * mono[40];
  mono[169] = mono[7] * mono[38];
  mono[170] = mono[2] * mono[96];
  mono[171] = mono[2] * mono[104];
  mono[172] = mono[2] * mono[99];
  mono[173] = mono[2] * mono[106];
  mono[174] = mono[2] * mono[109];
  mono[175] = mono[3] * mono[109];
  mono[176] = mono[3] * mono[112];
  mono[177] = mono[2] * mono[113];
  mono[178] = mono[3] * mono[116];
  mono[179] = mono[2] * mono[117];
  mono[180] = mono[2] * mono[131];
  mono[181] = mono[2] * mono[133];
  mono[182] = mono[2] * mono[112];
  mono[183] = mono[3] * mono[113];
  mono[184] = mono[2] * mono[116];
  mono[185] = mono[3] * mono[117];
  mono[186] = mono[4] * mono[120];
  mono[187] = mono[5] * mono[121];
  mono[188] = mono[2] * mono[111];
  mono[189] = mono[4] * mono[113];
  mono[190] = mono[2] * mono[115];
  mono[191] = mono[4] * mono[141];
  mono[192] = mono[2] * mono[143];
  mono[193] = mono[2] * mono[145];
  mono[194] = mono[4] * mono[146];
  mono[195] = mono[4] * mono[153];
  mono[196] = mono[4] * mono[159];
  mono[197] = mono[5] * mono[159];
  mono[198] = mono[2] * mono[148];
  mono[199] = mono[2] * mono[155];
  mono[200] = mono[2] * mono[161];
  mono[201] = mono[3] * mono[161];
  mono[202] = mono[2] * mono[150];
  mono[203] = mono[2] * mono[157];
  mono[204] = mono[2] * mono[163];
  mono[205] = mono[3] * mono[163];
  mono[206] = mono[3] * mono[152];
  mono[207] = mono[2] * mono[153];
  mono[208] = mono[5] * mono[154];
  mono[209] = mono[4] * mono[155];
  mono[210] = mono[6] * mono[157];
  mono[211] = mono[7] * mono[156];
  mono[212] = mono[2] * mono[158];
  mono[213] = mono[3] * mono[158];
  mono[214] = mono[2] * mono[159];
  mono[215] = mono[3] * mono[159];
  mono[216] = mono[4] * mono[160];
  mono[217] = mono[5] * mono[160];
  mono[218] = mono[4] * mono[161];
  mono[219] = mono[5] * mono[161];
  mono[220] = mono[6] * mono[162];
  mono[221] = mono[6] * mono[163];
  mono[222] = mono[7] * mono[162];
  mono[223] = mono[7] * mono[163];
  mono[224] = mono[3] * mono[96];
  mono[225] = mono[3] * mono[165];
  mono[226] = mono[2] * mono[166];
  mono[227] = mono[4] * mono[166];
  mono[228] = mono[2] * mono[167];
  mono[229] = mono[2] * mono[98];
  mono[230] = mono[3] * mono[99];
  mono[231] = mono[4] * mono[99];
  mono[232] = mono[2] * mono[100];
  mono[233] = mono[5] * mono[100];
  mono[234] = mono[3] * mono[169];
  mono[235] = mono[5] * mono[169];
  mono[236] = mono[2] * mono[164];
  mono[237] = mono[4] * mono[165];
  mono[238] = mono[2] * mono[103];
  mono[239] = mono[4] * mono[104];
  mono[240] = mono[6] * mono[106];
  mono[241] = mono[6] * mono[169];
  mono[242] = mono[6] * mono[122];
  mono[243] = mono[6] * mono[123];
  mono[244] = mono[7] * mono[124];
  mono[245] = mono[7] * mono[125];
  mono[246] = mono[4] * mono[126];
  mono[247] = mono[5] * mono[127];
  mono[248] = mono[4] * mono[128];
  mono[249] = mono[5] * mono[129];
  mono[250] = mono[2] * mono[130];
  mono[251] = mono[3] * mono[131];
  mono[252] = mono[2] * mono[132];
  mono[253] = mono[3] * mono[133];
  mono[254] = mono[2] * mono[175];
  mono[255] = mono[2] * mono[176];
  mono[256] = mono[2] * mono[183];
  mono[257] = mono[2] * mono[189];
  mono[258] = mono[3] * mono[189];
  mono[259] = mono[2] * mono[178];
  mono[260] = mono[2] * mono[185];
  mono[261] = mono[2] * mono[191];
  mono[262] = mono[3] * mono[191];
  mono[263] = mono[4] * mono[180];
  mono[264] = mono[4] * mono[187];
  mono[265] = mono[4] * mono[193];
  mono[266] = mono[5] * mono[193];
  mono[267] = mono[4] * mono[197];
  mono[268] = mono[2] * mono[201];
  mono[269] = mono[2] * mono[205];
  mono[270] = mono[3] * mono[194];
  mono[271] = mono[2] * mono[195];
  mono[272] = mono[3] * mono[196];
  mono[273] = mono[2] * mono[197];
  mono[274] = mono[3] * mono[208];
  mono[275] = mono[2] * mono[209];
  mono[276] = mono[3] * mono[218];
  mono[277] = mono[2] * mono[219];
  mono[278] = mono[2] * mono[210];
  mono[279] = mono[3] * mono[221];
  mono[280] = mono[3] * mono[211];
  mono[281] = mono[2] * mono[223];
  mono[282] = mono[2] * mono[194];
  mono[283] = mono[3] * mono[195];
  mono[284] = mono[2] * mono[196];
  mono[285] = mono[3] * mono[197];
  mono[286] = mono[4] * mono[198];
  mono[287] = mono[5] * mono[199];
  mono[288] = mono[2] * mono[218];
  mono[289] = mono[3] * mono[219];
  mono[290] = mono[4] * mono[220];
  mono[291] = mono[2] * mono[221];
  mono[292] = mono[5] * mono[222];
  mono[293] = mono[3] * mono[223];
  mono[294] = mono[2] * mono[227];
  mono[295] = mono[2] * mono[231];
  mono[296] = mono[2] * mono[174];
  mono[297] = mono[3] * mono[175];
  mono[298] = mono[3] * mono[233];
  mono[299] = mono[3] * mono[235];
  mono[300] = mono[2] * mono[224];
  mono[301] = mono[3] * mono[237];
  mono[302] = mono[2] * mono[171];
  mono[303] = mono[2] * mono[239];
  mono[304] = mono[2] * mono[230];
  mono[305] = mono[2] * mono[173];
  mono[306] = mono[4] * mono[175];
  mono[307] = mono[4] * mono[233];
  mono[308] = mono[2] * mono[240];
  mono[309] = mono[4] * mono[240];
  mono[310] = mono[3] * mono[241];
  mono[311] = mono[5] * mono[241];
  mono[312] = mono[6] * mono[176];
  mono[313] = mono[6] * mono[177];
  mono[314] = mono[6] * mono[244];
  mono[315] = mono[6] * mono[245];
  mono[316] = mono[4] * mono[178];
  mono[317] = mono[4] * mono[247];
  mono[318] = mono[4] * mono[179];
  mono[319] = mono[4] * mono[249];
  mono[320] = mono[2] * mono[180];
  mono[321] = mono[2] * mono[251];
  mono[322] = mono[2] * mono[181];
  mono[323] = mono[2] * mono[253];
  mono[324] = mono[6] * mono[194];
  mono[325] = mono[4] * mono[196];
  mono[326] = mono[5] * mono[197];
  mono[327] = mono[7] * mono[195];
  mono[328] = mono[6] * mono[198];
  mono[329] = mono[2] * mono[200];
  mono[330] = mono[3] * mono[201];
  mono[331] = mono[7] * mono[199];
  mono[332] = mono[4] * mono[202];
  mono[333] = mono[2] * mono[204];
  mono[334] = mono[3] * mono[205];
  mono[335] = mono[5] * mono[203];
  mono[336] = mono[12] * mono[146];
  mono[337] = mono[13] * mono[146];
  mono[338] = mono[11] * mono[147];
  mono[339] = mono[15] * mono[147];
  mono[340] = mono[11] * mono[148];
  mono[341] = mono[14] * mono[148];
  mono[342] = mono[12] * mono[149];
  mono[343] = mono[16] * mono[149];
  mono[344] = mono[13] * mono[150];
  mono[345] = mono[14] * mono[151];
  mono[346] = mono[16] * mono[150];
  mono[347] = mono[15] * mono[151];
  mono[348] = mono[8] * mono[196];
  mono[349] = mono[8] * mono[197];
  mono[350] = mono[9] * mono[194];
  mono[351] = mono[9] * mono[195];
  mono[352] = mono[8] * mono[200];
  mono[353] = mono[8] * mono[201];
  mono[354] = mono[9] * mono[204];
  mono[355] = mono[9] * mono[205];
  mono[356] = mono[10] * mono[198];
  mono[357] = mono[10] * mono[199];
  mono[358] = mono[10] * mono[202];
  mono[359] = mono[10] * mono[203];
  mono[360] = mono[3] * mono[227];
  mono[361] = mono[5] * mono[226];
  mono[362] = mono[3] * mono[231];
  mono[363] = mono[2] * mono[233];
  mono[364] = mono[4] * mono[234];
  mono[365] = mono[2] * mono[235];
  mono[366] = r.iter().zip(vec![0, 0, 1, 3, 1, 0, 0, 1, 0, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[367] = r.iter().zip(vec![0, 0, 1, 3, 1, 1, 0, 0, 0, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[368] = r.iter().zip(vec![0, 0, 3, 1, 1, 0, 0, 0, 1, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[369] = r.iter().zip(vec![0, 0, 3, 1, 1, 0, 1, 0, 0, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[370] = r.iter().zip(vec![0, 1, 0, 0, 0, 1, 3, 1, 0, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[371] = r.iter().zip(vec![0, 1, 0, 0, 0, 3, 1, 0, 1, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[372] = r.iter().zip(vec![0, 1, 0, 1, 0, 3, 1, 0, 0, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[373] = r.iter().zip(vec![0, 1, 1, 0, 0, 1, 3, 0, 0, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[374] = r.iter().zip(vec![1, 0, 0, 0, 0, 0, 1, 3, 1, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[375] = r.iter().zip(vec![1, 0, 0, 0, 0, 1, 0, 1, 3, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[376] = r.iter().zip(vec![1, 0, 0, 1, 0, 0, 0, 3, 1, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[377] = r.iter().zip(vec![1, 0, 1, 0, 0, 0, 0, 1, 3, 0]).map(|(x,y)| x.powf(y as f64)).product();
  mono[378] = mono[6] * mono[242];
  mono[379] = mono[6] * mono[243];
  mono[380] = mono[7] * mono[244];
  mono[381] = mono[7] * mono[245];
  mono[382] = mono[4] * mono[246];
  mono[383] = mono[5] * mono[247];
  mono[384] = mono[4] * mono[248];
  mono[385] = mono[5] * mono[249];
  mono[386] = mono[2] * mono[250];
  mono[387] = mono[3] * mono[251];
  mono[388] = mono[2] * mono[252];
  mono[389] = mono[3] * mono[253];
  return mono;
}

