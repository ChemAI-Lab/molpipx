#![allow(unused_variables)]
use super::monomials_3_2_5::*;


pub const N_POLYS: usize = 364;

// File created from data/molecule_A3B2/MOL_3_2_5.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3] + mono[4] + mono[5] + mono[6] + mono[7];
    poly[3] = mono[8] + mono[9] + mono[10];
    poly[4] = poly[1] * poly[2];
    poly[5] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16];
    poly[6] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
    poly[7] = mono[23] + mono[24] + mono[25];
    poly[8] = poly[1] * poly[3];
    poly[9] = mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31];
    poly[10] = poly[2] * poly[3] - poly[9];
    poly[11] = mono[32] + mono[33] + mono[34];
    poly[12] = poly[1] * poly[1];
    poly[13] = poly[2] * poly[2] - poly[7] - poly[6] - poly[5] - poly[7] - poly[6] - poly[5];
    poly[14] = poly[3] * poly[3] - poly[11] - poly[11];
    poly[15] = poly[1] * poly[5];
    poly[16] = poly[1] * poly[6];
    poly[17] = mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40];
    poly[18] = mono[41] + mono[42];
    poly[19] = poly[1] * poly[7];
    poly[20] = mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54];
    poly[21] = poly[1] * poly[9];
    poly[22] = mono[55] + mono[56] + mono[57];
    poly[23] = poly[1] * poly[10];
    poly[24] = mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69];
    poly[25] = mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81];
    poly[26] = poly[3] * poly[5] - poly[24];
    poly[27] = poly[3] * poly[6] - poly[25];
    poly[28] = poly[3] * poly[7] - poly[22];
    poly[29] = poly[1] * poly[11];
    poly[30] = mono[82] + mono[83] + mono[84] + mono[85] + mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93];
    poly[31] = mono[94];
    poly[32] = poly[2] * poly[11] - poly[30];
    poly[33] = poly[1] * poly[4];
    poly[34] = poly[1] * poly[13];
    poly[35] = poly[2] * poly[5] - poly[20] - poly[17] - poly[17];
    poly[36] = poly[2] * poly[6] - poly[20] - poly[18] - poly[17] - poly[18] - poly[18];
    poly[37] = poly[2] * poly[7] - poly[20];
    poly[38] = poly[1] * poly[8];
    poly[39] = poly[2] * poly[9] - poly[25] - poly[24] - poly[22] - poly[22];
    poly[40] = poly[3] * poly[13] - poly[39];
    poly[41] = poly[1] * poly[14];
    poly[42] = poly[3] * poly[9] - poly[30];
    poly[43] = poly[2] * poly[14] - poly[42];
    poly[44] = poly[3] * poly[11] - poly[31] - poly[31] - poly[31];
    poly[45] = poly[1] * poly[12];
    poly[46] = poly[2] * poly[13] - poly[37] - poly[36] - poly[35];
    poly[47] = poly[3] * poly[14] - poly[44];
    poly[48] = poly[1] * poly[17];
    poly[49] = poly[1] * poly[18];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[20];
    poly[51] = mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100];
    poly[52] = mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106];
    poly[53] = mono[107] + mono[108] + mono[109];
    poly[54] = poly[1] * poly[22];
    poly[55] = poly[1] * poly[24];
    poly[56] = poly[1] * poly[25];
    poly[57] = mono[110] + mono[111] + mono[112] + mono[113] + mono[114] + mono[115] + mono[116] + mono[117] + mono[118] + mono[119] + mono[120] + mono[121];
    poly[58] = poly[1] * poly[26];
    poly[59] = mono[122] + mono[123] + mono[124] + mono[125] + mono[126] + mono[127] + mono[128] + mono[129] + mono[130] + mono[131] + mono[132] + mono[133];
    poly[60] = poly[1] * poly[27];
    poly[61] = poly[3] * poly[17] - poly[59];
    poly[62] = poly[3] * poly[18];
    poly[63] = poly[1] * poly[28];
    poly[64] = mono[134] + mono[135] + mono[136] + mono[137] + mono[138] + mono[139] + mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145];
    poly[65] = poly[3] * poly[20] - poly[64] - poly[57];
    poly[66] = poly[1] * poly[30];
    poly[67] = mono[146] + mono[147] + mono[148] + mono[149] + mono[150] + mono[151];
    poly[68] = mono[152] + mono[153] + mono[154] + mono[155] + mono[156] + mono[157];
    poly[69] = mono[158] + mono[159] + mono[160] + mono[161] + mono[162] + mono[163];
    poly[70] = poly[1] * poly[31];
    poly[71] = poly[1] * poly[32];
    poly[72] = poly[5] * poly[11] - poly[67];
    poly[73] = poly[6] * poly[11] - poly[68];
    poly[74] = poly[31] * poly[2];
    poly[75] = poly[7] * poly[11] - poly[69];
    poly[76] = poly[1] * poly[15];
    poly[77] = poly[1] * poly[16];
    poly[78] = poly[1] * poly[19];
    poly[79] = poly[1] * poly[35];
    poly[80] = mono[164] + mono[165] + mono[166] + mono[167] + mono[168] + mono[169];
    poly[81] = poly[1] * poly[36];
    poly[82] = poly[17] * poly[2] - poly[52] - poly[51] - poly[80] - poly[51];
    poly[83] = poly[2] * poly[18] - poly[52];
    poly[84] = poly[5] * poly[6] - poly[52] - poly[82] - poly[52];
    poly[85] = poly[1] * poly[37];
    poly[86] = poly[5] * poly[7] - poly[51];
    poly[87] = poly[6] * poly[7] - poly[52];
    poly[88] = poly[1] * poly[21];
    poly[89] = poly[1] * poly[39];
    poly[90] = poly[2] * poly[22] - poly[57];
    poly[91] = poly[1] * poly[23];
    poly[92] = poly[5] * poly[9] - poly[59] - poly[57];
    poly[93] = poly[6] * poly[9] - poly[62] - poly[61] - poly[57];
    poly[94] = poly[1] * poly[40];
    poly[95] = poly[2] * poly[24] - poly[64] - poly[59] - poly[61] - poly[57] - poly[92] - poly[61];
    poly[96] = poly[2] * poly[25] - poly[64] - poly[62] - poly[59] - poly[57] - poly[93] - poly[62];
    poly[97] = poly[2] * poly[26] - poly[65] - poly[59];
    poly[98] = poly[3] * poly[36] - poly[96] - poly[93];
    poly[99] = poly[3] * poly[37] - poly[90];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[29];
    poly[101] = poly[2] * poly[30] - poly[73] - poly[72] - poly[69] - poly[68] - poly[67] - poly[69] - poly[68] - poly[67];
    poly[102] = poly[11] * poly[13] - poly[101];
    poly[103] = poly[1] * poly[42];
    poly[104] = poly[3] * poly[22] - poly[69];
    poly[105] = poly[1] * poly[43];
    poly[106] = poly[3] * poly[24] - poly[72] - poly[67] - poly[67];
    poly[107] = poly[3] * poly[25] - poly[73] - poly[68] - poly[68];
    poly[108] = poly[3] * poly[26] - poly[72];
    poly[109] = poly[3] * poly[27] - poly[73];
    poly[110] = poly[7] * poly[14] - poly[104];
    poly[111] = poly[1] * poly[44];
    poly[112] = poly[9] * poly[11] - poly[74];
    poly[113] = poly[3] * poly[30] - poly[74] - poly[112] - poly[74];
    poly[114] = poly[31] * poly[3];
    poly[115] = poly[3] * poly[32] - poly[74];
    poly[116] = poly[1] * poly[33];
    poly[117] = poly[1] * poly[34];
    poly[118] = poly[5] * poly[5] - poly[53] - poly[51] - poly[80] - poly[53] - poly[51] - poly[80];
    poly[119] = poly[6] * poly[6] - poly[53] - poly[51] - poly[83] - poly[53] - poly[51] - poly[83];
    poly[120] = poly[7] * poly[7] - poly[53] - poly[53];
    poly[121] = poly[1] * poly[46];
    poly[122] = poly[5] * poly[13] - poly[87] - poly[82];
    poly[123] = poly[6] * poly[13] - poly[86] - poly[83] - poly[80];
    poly[124] = poly[7] * poly[13] - poly[84];
    poly[125] = poly[1] * poly[38];
    poly[126] = poly[2] * poly[39] - poly[93] - poly[92] - poly[90];
    poly[127] = poly[3] * poly[46] - poly[126];
    poly[128] = poly[1] * poly[41];
    poly[129] = poly[3] * poly[39] - poly[101];
    poly[130] = poly[13] * poly[14] - poly[129];
    poly[131] = poly[11] * poly[11] - poly[114] - poly[114];
    poly[132] = poly[1] * poly[47];
    poly[133] = poly[3] * poly[42] - poly[112];
    poly[134] = poly[2] * poly[47] - poly[133];
    poly[135] = poly[11] * poly[14] - poly[114];
    poly[136] = poly[1] * poly[45];
    poly[137] = poly[2] * poly[46] - poly[124] - poly[123] - poly[122];
    poly[138] = poly[3] * poly[47] - poly[135];
    poly[139] = poly[1] * poly[51];
    poly[140] = poly[1] * poly[52];
    poly[141] = poly[1] * poly[53];
    poly[142] = mono[170] + mono[171] + mono[172] + mono[173] + mono[174] + mono[175];
    poly[143] = poly[1] * poly[57];
    poly[144] = poly[1] * poly[59];
    poly[145] = mono[176] + mono[177] + mono[178] + mono[179] + mono[180] + mono[181];
    poly[146] = poly[1] * poly[61];
    poly[147] = poly[1] * poly[62];
    poly[148] = mono[182] + mono[183] + mono[184] + mono[185] + mono[186] + mono[187];
    poly[149] = poly[1] * poly[64];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = mono[188] + mono[189] + mono[190] + mono[191] + mono[192] + mono[193];
    poly[151] = poly[1] * poly[65];
    poly[152] = poly[3] * poly[51] - poly[145];
    poly[153] = poly[3] * poly[52] - poly[148];
    poly[154] = poly[3] * poly[53] - poly[150];
    poly[155] = poly[1] * poly[67];
    poly[156] = poly[1] * poly[68];
    poly[157] = poly[1] * poly[69];
    poly[158] = mono[194] + mono[195] + mono[196] + mono[197] + mono[198] + mono[199] + mono[200] + mono[201] + mono[202] + mono[203] + mono[204] + mono[205];
    poly[159] = poly[1] * poly[72];
    poly[160] = mono[206] + mono[207] + mono[208] + mono[209] + mono[210] + mono[211];
    poly[161] = poly[1] * poly[73];
    poly[162] = poly[11] * poly[17] - poly[160];
    poly[163] = poly[11] * poly[18];
    poly[164] = mono[212] + mono[213] + mono[214] + mono[215] + mono[216] + mono[217] + mono[218] + mono[219] + mono[220] + mono[221] + mono[222] + mono[223];
    poly[165] = poly[1] * poly[74];
    poly[166] = poly[31] * poly[5];
    poly[167] = poly[31] * poly[6];
    poly[168] = poly[1] * poly[75];
    poly[169] = poly[11] * poly[20] - poly[164] - poly[158];
    poly[170] = poly[31] * poly[7];
    poly[171] = poly[1] * poly[48];
    poly[172] = poly[1] * poly[49];
    poly[173] = poly[1] * poly[50];
    poly[174] = poly[1] * poly[80];
    poly[175] = poly[1] * poly[82];
    poly[176] = poly[1] * poly[83];
    poly[177] = poly[1] * poly[84];
    poly[178] = mono[224] + mono[225] + mono[226] + mono[227] + mono[228] + mono[229] + mono[230] + mono[231] + mono[232] + mono[233] + mono[234] + mono[235];
    poly[179] = poly[5] * poly[18];
    poly[180] = poly[1] * poly[86];
    poly[181] = mono[236] + mono[237] + mono[238] + mono[239] + mono[240] + mono[241];
    poly[182] = poly[1] * poly[87];
    poly[183] = poly[7] * poly[17] - poly[181];
    poly[184] = poly[7] * poly[18];
    poly[185] = poly[2] * poly[53] - poly[142];
    poly[186] = poly[1] * poly[54];
    poly[187] = poly[1] * poly[90];
    poly[188] = poly[1] * poly[55];
    poly[189] = poly[1] * poly[92];
    poly[190] = poly[1] * poly[56];
    poly[191] = poly[5] * poly[22] - poly[145];
    poly[192] = poly[1] * poly[93];
    poly[193] = poly[6] * poly[22] - poly[148];
    poly[194] = poly[1] * poly[58];
    poly[195] = mono[242] + mono[243] + mono[244] + mono[245] + mono[246] + mono[247] + mono[248] + mono[249] + mono[250] + mono[251] + mono[252] + mono[253];
    poly[196] = poly[1] * poly[60];
    poly[197] = poly[9] * poly[17] - poly[148] - poly[145] - poly[195] - poly[145];
    poly[198] = poly[9] * poly[18] - poly[148];
    poly[199] = poly[1] * poly[63];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[9] * poly[20] - poly[153] - poly[152] - poly[150] - poly[193] - poly[191] - poly[150];
    poly[201] = poly[1] * poly[95];
    poly[202] = poly[1] * poly[96];
    poly[203] = poly[2] * poly[57] - poly[150] - poly[148] - poly[145] - poly[193] - poly[191] - poly[150] - poly[148] - poly[145];
    poly[204] = poly[1] * poly[97];
    poly[205] = poly[3] * poly[80] - poly[197];
    poly[206] = poly[5] * poly[25] - poly[153] - poly[148] - poly[200] - poly[195] - poly[203] - poly[148];
    poly[207] = poly[1] * poly[98];
    poly[208] = poly[3] * poly[82] - poly[206] - poly[195];
    poly[209] = poly[3] * poly[83] - poly[198];
    poly[210] = poly[3] * poly[84] - poly[200] - poly[203];
    poly[211] = poly[1] * poly[99];
    poly[212] = poly[7] * poly[24] - poly[152] - poly[191];
    poly[213] = poly[7] * poly[25] - poly[153] - poly[193];
    poly[214] = poly[7] * poly[26] - poly[145];
    poly[215] = poly[7] * poly[27] - poly[148];
    poly[216] = poly[1] * poly[66];
    poly[217] = poly[1] * poly[101];
    poly[218] = poly[2] * poly[67] - poly[162] - poly[158];
    poly[219] = poly[2] * poly[68] - poly[163] - poly[160] - poly[158];
    poly[220] = poly[2] * poly[69] - poly[164] - poly[158];
    poly[221] = poly[1] * poly[70];
    poly[222] = poly[1] * poly[71];
    poly[223] = poly[5] * poly[30] - poly[164] - poly[160] - poly[162] - poly[158] - poly[218] - poly[160];
    poly[224] = poly[6] * poly[30] - poly[164] - poly[163] - poly[162] - poly[158] - poly[219] - poly[163];
    poly[225] = poly[1] * poly[102];
    poly[226] = poly[5] * poly[32] - poly[169] - poly[162];
    poly[227] = poly[11] * poly[36] - poly[224] - poly[219];
    poly[228] = poly[31] * poly[13];
    poly[229] = poly[2] * poly[75] - poly[169];
    poly[230] = poly[1] * poly[104];
    poly[231] = poly[1] * poly[106];
    poly[232] = poly[1] * poly[107];
    poly[233] = poly[3] * poly[57] - poly[164] - poly[158];
    poly[234] = poly[1] * poly[108];
    poly[235] = poly[9] * poly[26] - poly[164] - poly[223];
    poly[236] = poly[1] * poly[109];
    poly[237] = poly[3] * poly[61] - poly[162];
    poly[238] = poly[14] * poly[18];
    poly[239] = poly[1] * poly[110];
    poly[240] = poly[3] * poly[64] - poly[169] - poly[158];
    poly[241] = poly[3] * poly[65] - poly[169] - poly[164];
    poly[242] = poly[1] * poly[112];
    poly[243] = poly[11] * poly[22] - poly[170];
    poly[244] = poly[1] * poly[113];
    poly[245] = poly[3] * poly[67] - poly[166];
    poly[246] = poly[3] * poly[68] - poly[167];
    poly[247] = poly[3] * poly[69] - poly[170] - poly[243] - poly[170];
    poly[248] = poly[1] * poly[114];
    poly[249] = poly[31] * poly[9];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[1] * poly[115];
    poly[251] = poly[11] * poly[24] - poly[166] - poly[245] - poly[166];
    poly[252] = poly[11] * poly[25] - poly[167] - poly[246] - poly[167];
    poly[253] = poly[11] * poly[26] - poly[166];
    poly[254] = poly[11] * poly[27] - poly[167];
    poly[255] = poly[31] * poly[10];
    poly[256] = poly[3] * poly[75] - poly[170];
    poly[257] = poly[1] * poly[76];
    poly[258] = poly[1] * poly[77];
    poly[259] = poly[1] * poly[78];
    poly[260] = poly[1] * poly[79];
    poly[261] = poly[1] * poly[118];
    poly[262] = poly[1] * poly[81];
    poly[263] = poly[5] * poly[17] - poly[142] - poly[178] - poly[142];
    poly[264] = poly[1] * poly[119];
    poly[265] = poly[6] * poly[17] - poly[142] - poly[179] - poly[178];
    poly[266] = poly[6] * poly[18] - poly[142];
    poly[267] = poly[1] * poly[85];
    poly[268] = poly[5] * poly[20] - poly[142] - poly[185] - poly[181] - poly[183] - poly[178] - poly[142] - poly[181];
    poly[269] = poly[6] * poly[20] - poly[142] - poly[185] - poly[184] - poly[179] - poly[183] - poly[142] - poly[184];
    poly[270] = poly[1] * poly[120];
    poly[271] = poly[7] * poly[20] - poly[142] - poly[185] - poly[142];
    poly[272] = poly[1] * poly[122];
    poly[273] = poly[2] * poly[80] - poly[181] - poly[178] - poly[263];
    poly[274] = poly[1] * poly[123];
    poly[275] = poly[13] * poly[17] - poly[184] - poly[183] - poly[273];
    poly[276] = poly[13] * poly[18] - poly[181];
    poly[277] = poly[6] * poly[35] - poly[181] - poly[179] - poly[268] - poly[263] - poly[275] - poly[181];
    poly[278] = poly[1] * poly[124];
    poly[279] = poly[5] * poly[37] - poly[183] - poly[271];
    poly[280] = poly[7] * poly[36] - poly[179] - poly[269];
    poly[281] = poly[1] * poly[88];
    poly[282] = poly[1] * poly[89];
    poly[283] = poly[7] * poly[22] - poly[150];
    poly[284] = poly[1] * poly[126];
    poly[285] = poly[7] * poly[39] - poly[200];
    poly[286] = poly[1] * poly[91];
    poly[287] = poly[5] * poly[39] - poly[195] - poly[193];
    poly[288] = poly[6] * poly[39] - poly[198] - poly[197] - poly[191];
    poly[289] = poly[1] * poly[94];
    poly[290] = poly[9] * poly[35] - poly[205] - poly[206] - poly[203] - poly[191] - poly[287];
    poly[291] = poly[9] * poly[36] - poly[209] - poly[208] - poly[203] - poly[193] - poly[288];
    poly[292] = poly[3] * poly[118] - poly[290];
    poly[293] = poly[3] * poly[119] - poly[291];
    poly[294] = poly[3] * poly[120] - poly[283];
    poly[295] = poly[1] * poly[127];
    poly[296] = poly[2] * poly[95] - poly[212] - poly[205] - poly[208] - poly[203] - poly[290];
    poly[297] = poly[2] * poly[96] - poly[213] - poly[209] - poly[206] - poly[203] - poly[291];
    poly[298] = poly[3] * poly[122] - poly[296] - poly[287];
    poly[299] = poly[3] * poly[123] - poly[297] - poly[288];
}

#[inline(never)]
fn f_polynomials6(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[300] = poly[3] * poly[124] - poly[285];
    poly[301] = poly[1] * poly[100];
    poly[302] = poly[2] * poly[101] - poly[224] - poly[223] - poly[220] - poly[219] - poly[218];
    poly[303] = poly[11] * poly[46] - poly[302];
    poly[304] = poly[1] * poly[103];
    poly[305] = poly[1] * poly[129];
    poly[306] = poly[2] * poly[104] - poly[233];
    poly[307] = poly[1] * poly[105];
    poly[308] = poly[3] * poly[92] - poly[223] - poly[218];
    poly[309] = poly[3] * poly[93] - poly[224] - poly[219];
    poly[310] = poly[1] * poly[130];
    poly[311] = poly[3] * poly[95] - poly[226] - poly[218];
    poly[312] = poly[3] * poly[96] - poly[227] - poly[219];
    poly[313] = poly[2] * poly[108] - poly[241] - poly[235];
    poly[314] = poly[3] * poly[98] - poly[227] - poly[224];
    poly[315] = poly[14] * poly[37] - poly[306];
    poly[316] = poly[1] * poly[111];
    poly[317] = poly[11] * poly[39] - poly[228];
    poly[318] = poly[3] * poly[101] - poly[228] - poly[317] - poly[228];
    poly[319] = poly[3] * poly[102] - poly[228];
    poly[320] = poly[1] * poly[131];
    poly[321] = poly[11] * poly[30] - poly[255] - poly[249] - poly[249];
    poly[322] = poly[31] * poly[11];
    poly[323] = poly[2] * poly[131] - poly[321];
    poly[324] = poly[1] * poly[133];
    poly[325] = poly[3] * poly[104] - poly[243];
    poly[326] = poly[1] * poly[134];
    poly[327] = poly[3] * poly[106] - poly[251] - poly[245];
    poly[328] = poly[3] * poly[107] - poly[252] - poly[246];
    poly[329] = poly[3] * poly[108] - poly[253];
    poly[330] = poly[3] * poly[109] - poly[254];
    poly[331] = poly[7] * poly[47] - poly[325];
    poly[332] = poly[1] * poly[135];
    poly[333] = poly[11] * poly[42] - poly[249];
    poly[334] = poly[3] * poly[113] - poly[255] - poly[321];
    poly[335] = poly[31] * poly[14];
    poly[336] = poly[14] * poly[32] - poly[249];
    poly[337] = poly[1] * poly[116];
    poly[338] = poly[1] * poly[117];
    poly[339] = poly[1] * poly[121];
    poly[340] = poly[2] * poly[118] - poly[268] - poly[263];
    poly[341] = poly[2] * poly[119] - poly[269] - poly[266] - poly[265];
    poly[342] = poly[2] * poly[120] - poly[271];
    poly[343] = poly[1] * poly[137];
    poly[344] = poly[5] * poly[46] - poly[280] - poly[275];
    poly[345] = poly[6] * poly[46] - poly[279] - poly[276] - poly[273];
    poly[346] = poly[7] * poly[46] - poly[277];
    poly[347] = poly[1] * poly[125];
    poly[348] = poly[2] * poly[126] - poly[288] - poly[287] - poly[285];
    poly[349] = poly[3] * poly[137] - poly[348];
}

#[inline(never)]
fn f_polynomials7(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[350] = poly[1] * poly[128];
    poly[351] = poly[3] * poly[126] - poly[302];
    poly[352] = poly[14] * poly[46] - poly[351];
    poly[353] = poly[1] * poly[132];
    poly[354] = poly[3] * poly[129] - poly[317];
    poly[355] = poly[13] * poly[47] - poly[354];
    poly[356] = poly[3] * poly[131] - poly[322];
    poly[357] = poly[1] * poly[138];
    poly[358] = poly[3] * poly[133] - poly[333];
    poly[359] = poly[2] * poly[138] - poly[358];
    poly[360] = poly[11] * poly[47] - poly[335];
    poly[361] = poly[1] * poly[136];
    poly[362] = poly[2] * poly[137] - poly[346] - poly[345] - poly[344];
    poly[363] = poly[3] * poly[138] - poly[360];
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

    return poly;
}
