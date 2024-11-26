#![allow(unused_variables)]
use super::monomials_5_4::*;


pub const N_POLYS: usize = 29;

// File created from data/molecule_A5/MOL_5_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3] + mono[4] + mono[5] + mono[6] + mono[7] + mono[8] + mono[9] + mono[10];
    poly[2] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16] + mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22] + mono[23] + mono[24] + mono[25];
    poly[3] = mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31] + mono[32] + mono[33] + mono[34] + mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40] + mono[41] + mono[42] + mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54] + mono[55];
    poly[4] = poly[1] * poly[1] - poly[3] - poly[2] - poly[3] - poly[2];
    poly[5] = mono[56] + mono[57] + mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69] + mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85];
    poly[6] = mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93] + mono[94] + mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100] + mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106] + mono[107] + mono[108] + mono[109] + mono[110] + mono[111] + mono[112] + mono[113] + mono[114] + mono[115] + mono[116] + mono[117] + mono[118] + mono[119] + mono[120] + mono[121] + mono[122] + mono[123] + mono[124] + mono[125] + mono[126] + mono[127] + mono[128] + mono[129] + mono[130] + mono[131] + mono[132] + mono[133] + mono[134] + mono[135] + mono[136] + mono[137] + mono[138] + mono[139] + mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145];
    poly[7] = mono[146] + mono[147] + mono[148] + mono[149] + mono[150] + mono[151] + mono[152] + mono[153] + mono[154] + mono[155];
    poly[8] = mono[156] + mono[157] + mono[158] + mono[159] + mono[160] + mono[161] + mono[162] + mono[163] + mono[164] + mono[165] + mono[166] + mono[167] + mono[168] + mono[169] + mono[170] + mono[171] + mono[172] + mono[173] + mono[174] + mono[175];
    poly[9] = poly[1] * poly[2] - poly[6] - poly[5] - poly[5];
    poly[10] = poly[1] * poly[3] - poly[7] - poly[8] - poly[6] - poly[5] - poly[7] - poly[8] - poly[6] - poly[7] - poly[8];
    poly[11] = poly[1] * poly[4] - poly[10] - poly[9];
    poly[12] = mono[176] + mono[177] + mono[178] + mono[179] + mono[180] + mono[181] + mono[182] + mono[183] + mono[184] + mono[185] + mono[186] + mono[187] + mono[188] + mono[189] + mono[190] + mono[191] + mono[192] + mono[193] + mono[194] + mono[195] + mono[196] + mono[197] + mono[198] + mono[199] + mono[200] + mono[201] + mono[202] + mono[203] + mono[204] + mono[205] + mono[206] + mono[207] + mono[208] + mono[209] + mono[210] + mono[211] + mono[212] + mono[213] + mono[214] + mono[215] + mono[216] + mono[217] + mono[218] + mono[219] + mono[220] + mono[221] + mono[222] + mono[223] + mono[224] + mono[225] + mono[226] + mono[227] + mono[228] + mono[229] + mono[230] + mono[231] + mono[232] + mono[233] + mono[234] + mono[235];
    poly[13] = mono[236] + mono[237] + mono[238] + mono[239] + mono[240] + mono[241] + mono[242] + mono[243] + mono[244] + mono[245] + mono[246] + mono[247] + mono[248] + mono[249] + mono[250];
    poly[14] = mono[251] + mono[252] + mono[253] + mono[254] + mono[255] + mono[256] + mono[257] + mono[258] + mono[259] + mono[260];
    poly[15] = mono[261] + mono[262] + mono[263] + mono[264] + mono[265] + mono[266] + mono[267] + mono[268] + mono[269] + mono[270] + mono[271] + mono[272] + mono[273] + mono[274] + mono[275] + mono[276] + mono[277] + mono[278] + mono[279] + mono[280] + mono[281] + mono[282] + mono[283] + mono[284] + mono[285] + mono[286] + mono[287] + mono[288] + mono[289] + mono[290] + mono[291] + mono[292] + mono[293] + mono[294] + mono[295] + mono[296] + mono[297] + mono[298] + mono[299] + mono[300] + mono[301] + mono[302] + mono[303] + mono[304] + mono[305] + mono[306] + mono[307] + mono[308] + mono[309] + mono[310] + mono[311] + mono[312] + mono[313] + mono[314] + mono[315] + mono[316] + mono[317] + mono[318] + mono[319] + mono[320];
    poly[16] = mono[321] + mono[322] + mono[323] + mono[324] + mono[325] + mono[326] + mono[327] + mono[328] + mono[329] + mono[330] + mono[331] + mono[332] + mono[333] + mono[334] + mono[335] + mono[336] + mono[337] + mono[338] + mono[339] + mono[340] + mono[341] + mono[342] + mono[343] + mono[344] + mono[345] + mono[346] + mono[347] + mono[348] + mono[349] + mono[350] + mono[351] + mono[352] + mono[353] + mono[354] + mono[355] + mono[356] + mono[357] + mono[358] + mono[359] + mono[360] + mono[361] + mono[362] + mono[363] + mono[364] + mono[365] + mono[366] + mono[367] + mono[368] + mono[369] + mono[370] + mono[371] + mono[372] + mono[373] + mono[374] + mono[375] + mono[376] + mono[377] + mono[378] + mono[379] + mono[380];
    poly[17] = mono[381] + mono[382] + mono[383] + mono[384] + mono[385];
    poly[18] = mono[386] + mono[387] + mono[388] + mono[389] + mono[390] + mono[391] + mono[392] + mono[393] + mono[394] + mono[395] + mono[396] + mono[397] + mono[398] + mono[399] + mono[400] + mono[401] + mono[402] + mono[403] + mono[404] + mono[405] + mono[406] + mono[407] + mono[408] + mono[409] + mono[410] + mono[411] + mono[412] + mono[413] + mono[414] + mono[415];
    poly[19] = poly[5] * poly[1] - poly[15] - poly[12] - poly[14] - poly[18] - poly[12] - poly[14] - poly[14];
    poly[20] = poly[2] * poly[3] - poly[16] - poly[15] - poly[12] - poly[14] - poly[19] - poly[15] - poly[14] - poly[14];
    poly[21] = poly[1] * poly[6] - poly[16] - poly[13] - poly[15] - poly[12] - poly[20] - poly[16] - poly[13] - poly[15] - poly[12] - poly[13] - poly[13];
    poly[22] = poly[1] * poly[7] - poly[16] - poly[14];
    poly[23] = poly[1] * poly[8] - poly[16] - poly[17] - poly[15] - poly[17] - poly[17] - poly[17];
    poly[24] = poly[2] * poly[2] - poly[13] - poly[12] - poly[18] - poly[13] - poly[12] - poly[18];
    poly[25] = poly[3] * poly[3] - poly[16] - poly[13] - poly[17] - poly[15] - poly[12] - poly[22] - poly[23] - poly[21] - poly[16] - poly[13] - poly[17] - poly[15] - poly[12] - poly[22] - poly[23] - poly[21] - poly[16] - poly[13] - poly[17] - poly[16] - poly[13] - poly[17] - poly[17] - poly[17];
    poly[26] = poly[2] * poly[4] - poly[21] - poly[19];
    poly[27] = poly[3] * poly[4] - poly[22] - poly[23] - poly[20] - poly[18];
    poly[28] = poly[1] * poly[11] - poly[27] - poly[26];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
