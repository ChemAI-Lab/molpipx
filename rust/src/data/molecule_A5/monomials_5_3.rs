#![allow(unused_variables)]
// File created from data/molecule_A5/MOL_5_3.MONO 
// Total number of monomials = 176 

pub const N_MONOS: usize = 176;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
  mono[11] = mono[3] * mono[4];
  mono[12] = mono[2] * mono[5];
  mono[13] = mono[1] * mono[6];
  mono[14] = mono[3] * mono[7];
  mono[15] = mono[5] * mono[7];
  mono[16] = mono[6] * mono[7];
  mono[17] = mono[2] * mono[8];
  mono[18] = mono[4] * mono[8];
  mono[19] = mono[6] * mono[8];
  mono[20] = mono[1] * mono[9];
  mono[21] = mono[4] * mono[9];
  mono[22] = mono[5] * mono[9];
  mono[23] = mono[1] * mono[10];
  mono[24] = mono[2] * mono[10];
  mono[25] = mono[3] * mono[10];
  mono[26] = mono[1] * mono[2];
  mono[27] = mono[1] * mono[3];
  mono[28] = mono[2] * mono[3];
  mono[29] = mono[1] * mono[4];
  mono[30] = mono[2] * mono[4];
  mono[31] = mono[1] * mono[5];
  mono[32] = mono[3] * mono[5];
  mono[33] = mono[4] * mono[5];
  mono[34] = mono[2] * mono[6];
  mono[35] = mono[3] * mono[6];
  mono[36] = mono[4] * mono[6];
  mono[37] = mono[5] * mono[6];
  mono[38] = mono[1] * mono[7];
  mono[39] = mono[2] * mono[7];
  mono[40] = mono[4] * mono[7];
  mono[41] = mono[1] * mono[8];
  mono[42] = mono[3] * mono[8];
  mono[43] = mono[5] * mono[8];
  mono[44] = mono[7] * mono[8];
  mono[45] = mono[2] * mono[9];
  mono[46] = mono[3] * mono[9];
  mono[47] = mono[6] * mono[9];
  mono[48] = mono[7] * mono[9];
  mono[49] = mono[8] * mono[9];
  mono[50] = mono[4] * mono[10];
  mono[51] = mono[5] * mono[10];
  mono[52] = mono[6] * mono[10];
  mono[53] = mono[7] * mono[10];
  mono[54] = mono[8] * mono[10];
  mono[55] = mono[9] * mono[10];
  mono[56] = mono[3] * mono[40];
  mono[57] = mono[2] * mono[15];
  mono[58] = mono[3] * mono[15];
  mono[59] = mono[1] * mono[16];
  mono[60] = mono[3] * mono[16];
}

fn f_monomials1(mono: &mut [f64; N_MONOS]) { 
  mono[61] = mono[5] * mono[16];
  mono[62] = mono[2] * mono[18];
  mono[63] = mono[3] * mono[18];
  mono[64] = mono[2] * mono[43];
  mono[65] = mono[1] * mono[19];
  mono[66] = mono[2] * mono[19];
  mono[67] = mono[4] * mono[19];
  mono[68] = mono[6] * mono[44];
  mono[69] = mono[1] * mono[21];
  mono[70] = mono[3] * mono[21];
  mono[71] = mono[1] * mono[22];
  mono[72] = mono[2] * mono[22];
  mono[73] = mono[4] * mono[22];
  mono[74] = mono[1] * mono[47];
  mono[75] = mono[5] * mono[48];
  mono[76] = mono[4] * mono[49];
  mono[77] = mono[1] * mono[24];
  mono[78] = mono[1] * mono[25];
  mono[79] = mono[2] * mono[25];
  mono[80] = mono[3] * mono[50];
  mono[81] = mono[2] * mono[51];
  mono[82] = mono[1] * mono[52];
  mono[83] = mono[3] * mono[53];
  mono[84] = mono[2] * mono[54];
  mono[85] = mono[1] * mono[55];
  mono[86] = mono[1] * mono[11];
  mono[87] = mono[2] * mono[11];
  mono[88] = mono[1] * mono[12];
  mono[89] = mono[2] * mono[32];
  mono[90] = mono[2] * mono[33];
  mono[91] = mono[3] * mono[33];
  mono[92] = mono[1] * mono[34];
  mono[93] = mono[1] * mono[35];
  mono[94] = mono[1] * mono[36];
  mono[95] = mono[3] * mono[36];
  mono[96] = mono[1] * mono[37];
  mono[97] = mono[2] * mono[37];
  mono[98] = mono[1] * mono[14];
  mono[99] = mono[2] * mono[14];
  mono[100] = mono[1] * mono[15];
  mono[101] = mono[4] * mono[15];
  mono[102] = mono[2] * mono[16];
  mono[103] = mono[4] * mono[16];
  mono[104] = mono[1] * mono[17];
  mono[105] = mono[2] * mono[42];
  mono[106] = mono[1] * mono[18];
  mono[107] = mono[4] * mono[43];
  mono[108] = mono[3] * mono[19];
  mono[109] = mono[5] * mono[19];
  mono[110] = mono[2] * mono[44];
}

fn f_monomials2(mono: &mut [f64; N_MONOS]) { 
  mono[111] = mono[3] * mono[44];
  mono[112] = mono[4] * mono[44];
  mono[113] = mono[5] * mono[44];
  mono[114] = mono[1] * mono[45];
  mono[115] = mono[1] * mono[46];
  mono[116] = mono[2] * mono[21];
  mono[117] = mono[3] * mono[22];
  mono[118] = mono[4] * mono[47];
  mono[119] = mono[5] * mono[47];
  mono[120] = mono[1] * mono[48];
  mono[121] = mono[3] * mono[48];
  mono[122] = mono[4] * mono[48];
  mono[123] = mono[6] * mono[48];
  mono[124] = mono[1] * mono[49];
  mono[125] = mono[2] * mono[49];
  mono[126] = mono[5] * mono[49];
  mono[127] = mono[6] * mono[49];
  mono[128] = mono[1] * mono[50];
  mono[129] = mono[2] * mono[50];
  mono[130] = mono[1] * mono[51];
  mono[131] = mono[3] * mono[51];
  mono[132] = mono[2] * mono[52];
  mono[133] = mono[3] * mono[52];
  mono[134] = mono[1] * mono[53];
  mono[135] = mono[2] * mono[53];
  mono[136] = mono[5] * mono[53];
  mono[137] = mono[6] * mono[53];
  mono[138] = mono[1] * mono[54];
  mono[139] = mono[3] * mono[54];
  mono[140] = mono[4] * mono[54];
  mono[141] = mono[6] * mono[54];
  mono[142] = mono[2] * mono[55];
  mono[143] = mono[3] * mono[55];
  mono[144] = mono[4] * mono[55];
  mono[145] = mono[5] * mono[55];
  mono[146] = mono[1] * mono[28];
  mono[147] = mono[1] * mono[33];
  mono[148] = mono[2] * mono[36];
  mono[149] = mono[3] * mono[37];
  mono[150] = mono[1] * mono[44];
  mono[151] = mono[2] * mono[48];
  mono[152] = mono[3] * mono[49];
  mono[153] = mono[4] * mono[53];
  mono[154] = mono[5] * mono[54];
  mono[155] = mono[6] * mono[55];
  mono[156] = mono[1] * mono[30];
  mono[157] = mono[1] * mono[32];
  mono[158] = mono[2] * mono[35];
  mono[159] = mono[4] * mono[37];
  mono[160] = mono[1] * mono[39];
}

fn f_monomials3(mono: &mut [f64; N_MONOS]) { 
  mono[161] = mono[1] * mono[40];
  mono[162] = mono[2] * mono[40];
  mono[163] = mono[1] * mono[42];
  mono[164] = mono[1] * mono[43];
  mono[165] = mono[3] * mono[43];
  mono[166] = mono[2] * mono[46];
  mono[167] = mono[2] * mono[47];
  mono[168] = mono[3] * mono[47];
  mono[169] = mono[7] * mono[49];
  mono[170] = mono[4] * mono[51];
  mono[171] = mono[4] * mono[52];
  mono[172] = mono[5] * mono[52];
  mono[173] = mono[7] * mono[54];
  mono[174] = mono[7] * mono[55];
  mono[175] = mono[8] * mono[55];
}

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[9];
  mono[2] = r[8];
  mono[3] = r[7];
  mono[4] = r[6];
  mono[5] = r[5];
  mono[6] = r[4];
  mono[7] = r[3];
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

  return mono; 
}


