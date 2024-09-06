// File created from ../pipx/examples/Data/DDEthanol/PIP_deg_2/MOL_1_1_1_2_3_1_2.MONO 
// Total number of monomials = 209 

pub const N_MONOS: usize = 209;

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

  return mono; 
}


