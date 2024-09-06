#![allow(unused_variables)]
// File created from data/molecule_A4B/MOL_4_1_4.MONO 
// Total number of monomials = 262 

pub const N_MONOS: usize = 262;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
  mono[11] = mono[1] * mono[2];
  mono[12] = mono[1] * mono[3];
  mono[13] = mono[2] * mono[3];
  mono[14] = mono[1] * mono[4];
  mono[15] = mono[2] * mono[4];
  mono[16] = mono[3] * mono[4];
  mono[17] = mono[3] * mono[5];
  mono[18] = mono[2] * mono[6];
  mono[19] = mono[1] * mono[7];
  mono[20] = mono[4] * mono[5];
  mono[21] = mono[4] * mono[6];
  mono[22] = mono[4] * mono[7];
  mono[23] = mono[2] * mono[8];
  mono[24] = mono[3] * mono[8];
  mono[25] = mono[1] * mono[9];
  mono[26] = mono[3] * mono[9];
  mono[27] = mono[1] * mono[10];
  mono[28] = mono[2] * mono[10];
  mono[29] = mono[7] * mono[8];
  mono[30] = mono[6] * mono[9];
  mono[31] = mono[5] * mono[10];
  mono[32] = mono[5] * mono[6];
  mono[33] = mono[5] * mono[7];
  mono[34] = mono[6] * mono[7];
  mono[35] = mono[5] * mono[8];
  mono[36] = mono[6] * mono[8];
  mono[37] = mono[5] * mono[9];
  mono[38] = mono[7] * mono[9];
  mono[39] = mono[8] * mono[9];
  mono[40] = mono[6] * mono[10];
  mono[41] = mono[7] * mono[10];
  mono[42] = mono[8] * mono[10];
  mono[43] = mono[9] * mono[10];
  mono[44] = mono[1] * mono[13];
  mono[45] = mono[1] * mono[15];
  mono[46] = mono[1] * mono[16];
  mono[47] = mono[2] * mono[16];
  mono[48] = mono[3] * mono[20];
  mono[49] = mono[2] * mono[21];
  mono[50] = mono[1] * mono[22];
  mono[51] = mono[2] * mono[24];
  mono[52] = mono[1] * mono[26];
  mono[53] = mono[1] * mono[28];
  mono[54] = mono[1] * mono[17];
  mono[55] = mono[2] * mono[17];
  mono[56] = mono[1] * mono[18];
  mono[57] = mono[3] * mono[18];
  mono[58] = mono[2] * mono[19];
  mono[59] = mono[3] * mono[19];
  mono[60] = mono[1] * mono[20];
}

fn f_monomials1(mono: &mut [f64; N_MONOS]) { 
  mono[61] = mono[2] * mono[20];
  mono[62] = mono[1] * mono[21];
  mono[63] = mono[3] * mono[21];
  mono[64] = mono[2] * mono[22];
  mono[65] = mono[3] * mono[22];
  mono[66] = mono[1] * mono[23];
  mono[67] = mono[1] * mono[24];
  mono[68] = mono[4] * mono[23];
  mono[69] = mono[4] * mono[24];
  mono[70] = mono[2] * mono[25];
  mono[71] = mono[2] * mono[26];
  mono[72] = mono[4] * mono[25];
  mono[73] = mono[4] * mono[26];
  mono[74] = mono[3] * mono[27];
  mono[75] = mono[3] * mono[28];
  mono[76] = mono[4] * mono[27];
  mono[77] = mono[4] * mono[28];
  mono[78] = mono[4] * mono[32];
  mono[79] = mono[4] * mono[33];
  mono[80] = mono[4] * mono[34];
  mono[81] = mono[3] * mono[35];
  mono[82] = mono[2] * mono[36];
  mono[83] = mono[3] * mono[37];
  mono[84] = mono[1] * mono[38];
  mono[85] = mono[3] * mono[39];
  mono[86] = mono[2] * mono[40];
  mono[87] = mono[1] * mono[41];
  mono[88] = mono[2] * mono[42];
  mono[89] = mono[1] * mono[43];
  mono[90] = mono[2] * mono[32];
  mono[91] = mono[3] * mono[32];
  mono[92] = mono[1] * mono[33];
  mono[93] = mono[3] * mono[33];
  mono[94] = mono[1] * mono[34];
  mono[95] = mono[2] * mono[34];
  mono[96] = mono[2] * mono[35];
  mono[97] = mono[3] * mono[36];
  mono[98] = mono[4] * mono[35];
  mono[99] = mono[4] * mono[36];
  mono[100] = mono[1] * mono[37];
  mono[101] = mono[3] * mono[38];
  mono[102] = mono[4] * mono[37];
  mono[103] = mono[4] * mono[38];
  mono[104] = mono[1] * mono[39];
  mono[105] = mono[2] * mono[39];
  mono[106] = mono[1] * mono[40];
  mono[107] = mono[2] * mono[41];
  mono[108] = mono[4] * mono[40];
  mono[109] = mono[4] * mono[41];
  mono[110] = mono[1] * mono[42];
}

fn f_monomials2(mono: &mut [f64; N_MONOS]) { 
  mono[111] = mono[3] * mono[42];
  mono[112] = mono[2] * mono[43];
  mono[113] = mono[3] * mono[43];
  mono[114] = mono[5] * mono[29];
  mono[115] = mono[6] * mono[29];
  mono[116] = mono[5] * mono[30];
  mono[117] = mono[6] * mono[38];
  mono[118] = mono[6] * mono[39];
  mono[119] = mono[7] * mono[39];
  mono[120] = mono[5] * mono[40];
  mono[121] = mono[5] * mono[41];
  mono[122] = mono[5] * mono[42];
  mono[123] = mono[7] * mono[42];
  mono[124] = mono[5] * mono[43];
  mono[125] = mono[6] * mono[43];
  mono[126] = mono[5] * mono[34];
  mono[127] = mono[5] * mono[39];
  mono[128] = mono[6] * mono[42];
  mono[129] = mono[7] * mono[43];
  mono[130] = mono[5] * mono[36];
  mono[131] = mono[5] * mono[38];
  mono[132] = mono[6] * mono[41];
  mono[133] = mono[8] * mono[43];
  mono[134] = mono[1] * mono[47];
  mono[135] = mono[1] * mono[48];
  mono[136] = mono[2] * mono[48];
  mono[137] = mono[1] * mono[49];
  mono[138] = mono[2] * mono[63];
  mono[139] = mono[1] * mono[64];
  mono[140] = mono[1] * mono[65];
  mono[141] = mono[1] * mono[51];
  mono[142] = mono[2] * mono[69];
  mono[143] = mono[1] * mono[71];
  mono[144] = mono[1] * mono[73];
  mono[145] = mono[1] * mono[75];
  mono[146] = mono[1] * mono[77];
  mono[147] = mono[7] * mono[66];
  mono[148] = mono[7] * mono[67];
  mono[149] = mono[7] * mono[68];
  mono[150] = mono[7] * mono[69];
  mono[151] = mono[6] * mono[70];
  mono[152] = mono[6] * mono[71];
  mono[153] = mono[6] * mono[72];
  mono[154] = mono[6] * mono[73];
  mono[155] = mono[5] * mono[74];
  mono[156] = mono[5] * mono[75];
  mono[157] = mono[5] * mono[76];
  mono[158] = mono[5] * mono[77];
  mono[159] = mono[2] * mono[78];
  mono[160] = mono[3] * mono[78];
}

fn f_monomials3(mono: &mut [f64; N_MONOS]) { 
  mono[161] = mono[1] * mono[79];
  mono[162] = mono[3] * mono[79];
  mono[163] = mono[1] * mono[80];
  mono[164] = mono[2] * mono[80];
  mono[165] = mono[2] * mono[81];
  mono[166] = mono[2] * mono[97];
  mono[167] = mono[3] * mono[98];
  mono[168] = mono[2] * mono[99];
  mono[169] = mono[1] * mono[83];
  mono[170] = mono[1] * mono[101];
  mono[171] = mono[3] * mono[102];
  mono[172] = mono[1] * mono[103];
  mono[173] = mono[1] * mono[85];
  mono[174] = mono[2] * mono[85];
  mono[175] = mono[1] * mono[86];
  mono[176] = mono[1] * mono[107];
  mono[177] = mono[2] * mono[108];
  mono[178] = mono[1] * mono[109];
  mono[179] = mono[1] * mono[88];
  mono[180] = mono[2] * mono[111];
  mono[181] = mono[1] * mono[112];
  mono[182] = mono[1] * mono[113];
  mono[183] = mono[2] * mono[91];
  mono[184] = mono[1] * mono[93];
  mono[185] = mono[1] * mono[95];
  mono[186] = mono[2] * mono[98];
  mono[187] = mono[3] * mono[99];
  mono[188] = mono[1] * mono[102];
  mono[189] = mono[3] * mono[103];
  mono[190] = mono[1] * mono[105];
  mono[191] = mono[1] * mono[108];
  mono[192] = mono[2] * mono[109];
  mono[193] = mono[1] * mono[111];
  mono[194] = mono[2] * mono[113];
  mono[195] = mono[3] * mono[114];
  mono[196] = mono[2] * mono[115];
  mono[197] = mono[4] * mono[114];
  mono[198] = mono[4] * mono[115];
  mono[199] = mono[3] * mono[116];
  mono[200] = mono[1] * mono[117];
  mono[201] = mono[4] * mono[116];
  mono[202] = mono[4] * mono[117];
  mono[203] = mono[2] * mono[118];
  mono[204] = mono[3] * mono[118];
  mono[205] = mono[1] * mono[119];
  mono[206] = mono[3] * mono[119];
  mono[207] = mono[2] * mono[120];
  mono[208] = mono[1] * mono[121];
  mono[209] = mono[4] * mono[120];
  mono[210] = mono[4] * mono[121];
}

fn f_monomials4(mono: &mut [f64; N_MONOS]) { 
  mono[211] = mono[2] * mono[122];
  mono[212] = mono[3] * mono[122];
  mono[213] = mono[1] * mono[123];
  mono[214] = mono[2] * mono[123];
  mono[215] = mono[1] * mono[124];
  mono[216] = mono[3] * mono[124];
  mono[217] = mono[1] * mono[125];
  mono[218] = mono[2] * mono[125];
  mono[219] = mono[6] * mono[119];
  mono[220] = mono[5] * mono[123];
  mono[221] = mono[5] * mono[125];
  mono[222] = mono[4] * mono[126];
  mono[223] = mono[3] * mono[127];
  mono[224] = mono[2] * mono[128];
  mono[225] = mono[1] * mono[129];
  mono[226] = mono[1] * mono[78];
  mono[227] = mono[2] * mono[79];
  mono[228] = mono[3] * mono[80];
  mono[229] = mono[1] * mono[81];
  mono[230] = mono[1] * mono[82];
  mono[231] = mono[2] * mono[83];
  mono[232] = mono[2] * mono[84];
  mono[233] = mono[4] * mono[85];
  mono[234] = mono[3] * mono[86];
  mono[235] = mono[3] * mono[87];
  mono[236] = mono[4] * mono[88];
  mono[237] = mono[4] * mono[89];
  mono[238] = mono[2] * mono[130];
  mono[239] = mono[3] * mono[130];
  mono[240] = mono[4] * mono[130];
  mono[241] = mono[1] * mono[131];
  mono[242] = mono[3] * mono[131];
  mono[243] = mono[4] * mono[131];
  mono[244] = mono[1] * mono[132];
  mono[245] = mono[2] * mono[132];
  mono[246] = mono[4] * mono[132];
  mono[247] = mono[1] * mono[133];
  mono[248] = mono[2] * mono[133];
  mono[249] = mono[3] * mono[133];
  mono[250] = mono[5] * mono[115];
  mono[251] = mono[5] * mono[117];
  mono[252] = mono[5] * mono[118];
  mono[253] = mono[5] * mono[119];
  mono[254] = mono[5] * mono[132];
  mono[255] = mono[5] * mono[128];
  mono[256] = mono[6] * mono[123];
  mono[257] = mono[5] * mono[129];
  mono[258] = mono[6] * mono[129];
  mono[259] = mono[5] * mono[133];
  mono[260] = mono[6] * mono[133];
}

fn f_monomials5(mono: &mut [f64; N_MONOS]) { 
  mono[261] = mono[7] * mono[133];
}

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[9];
  mono[2] = r[8];
  mono[3] = r[6];
  mono[4] = r[3];
  mono[5] = r[7];
  mono[6] = r[5];
  mono[7] = r[4];
  mono[8] = r[2];
  mono[9] = r[1];
  mono[10] = r[0];
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

  return mono; 
}


