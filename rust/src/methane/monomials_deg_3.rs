// File created from ../Data/Methane/PIP_deg_3/MOL_4_1_3.MONO 
// Total number of monomials = 134 

pub const N_MONOS: usize = 134;

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

  return mono; 
}


