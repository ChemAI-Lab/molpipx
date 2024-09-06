#![allow(unused_variables)]
// File created from data/molecule_A3BC/MOL_3_1_1_5.MONO 
// Total number of monomials = 140 

pub const N_MONOS: usize = 140;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
  mono[11] = mono[2] * mono[3];
  mono[12] = mono[2] * mono[4];
  mono[13] = mono[3] * mono[4];
  mono[14] = mono[3] * mono[5];
  mono[15] = mono[2] * mono[6];
  mono[16] = mono[4] * mono[5];
  mono[17] = mono[4] * mono[6];
  mono[18] = mono[2] * mono[7];
  mono[19] = mono[3] * mono[7];
  mono[20] = mono[5] * mono[6];
  mono[21] = mono[5] * mono[7];
  mono[22] = mono[6] * mono[7];
  mono[23] = mono[4] * mono[8];
  mono[24] = mono[3] * mono[9];
  mono[25] = mono[2] * mono[10];
  mono[26] = mono[7] * mono[8];
  mono[27] = mono[6] * mono[9];
  mono[28] = mono[5] * mono[10];
  mono[29] = mono[8] * mono[9];
  mono[30] = mono[8] * mono[10];
  mono[31] = mono[9] * mono[10];
  mono[32] = mono[2] * mono[13];
  mono[33] = mono[3] * mono[16];
  mono[34] = mono[2] * mono[17];
  mono[35] = mono[2] * mono[19];
  mono[36] = mono[4] * mono[20];
  mono[37] = mono[3] * mono[21];
  mono[38] = mono[2] * mono[22];
  mono[39] = mono[5] * mono[22];
  mono[40] = mono[4] * mono[26];
  mono[41] = mono[3] * mono[27];
  mono[42] = mono[2] * mono[28];
  mono[43] = mono[2] * mono[23];
  mono[44] = mono[3] * mono[23];
  mono[45] = mono[2] * mono[24];
  mono[46] = mono[4] * mono[24];
  mono[47] = mono[3] * mono[25];
  mono[48] = mono[4] * mono[25];
  mono[49] = mono[5] * mono[26];
  mono[50] = mono[6] * mono[26];
  mono[51] = mono[5] * mono[27];
  mono[52] = mono[7] * mono[27];
  mono[53] = mono[6] * mono[28];
  mono[54] = mono[7] * mono[28];
  mono[55] = mono[3] * mono[29];
  mono[56] = mono[4] * mono[29];
  mono[57] = mono[2] * mono[30];
  mono[58] = mono[4] * mono[30];
  mono[59] = mono[2] * mono[31];
  mono[60] = mono[3] * mono[31];
}

fn f_monomials1(mono: &mut [f64; N_MONOS]) { 
  mono[61] = mono[6] * mono[29];
  mono[62] = mono[7] * mono[29];
  mono[63] = mono[5] * mono[30];
  mono[64] = mono[7] * mono[30];
  mono[65] = mono[5] * mono[31];
  mono[66] = mono[6] * mono[31];
  mono[67] = mono[8] * mono[31];
  mono[68] = mono[2] * mono[36];
  mono[69] = mono[3] * mono[36];
  mono[70] = mono[2] * mono[37];
  mono[71] = mono[3] * mono[38];
  mono[72] = mono[4] * mono[37];
  mono[73] = mono[4] * mono[38];
  mono[74] = mono[2] * mono[40];
  mono[75] = mono[3] * mono[40];
  mono[76] = mono[2] * mono[41];
  mono[77] = mono[4] * mono[41];
  mono[78] = mono[3] * mono[42];
  mono[79] = mono[4] * mono[42];
  mono[80] = mono[4] * mono[49];
  mono[81] = mono[4] * mono[50];
  mono[82] = mono[3] * mono[51];
  mono[83] = mono[3] * mono[52];
  mono[84] = mono[2] * mono[53];
  mono[85] = mono[2] * mono[54];
  mono[86] = mono[3] * mono[49];
  mono[87] = mono[2] * mono[50];
  mono[88] = mono[4] * mono[51];
  mono[89] = mono[2] * mono[52];
  mono[90] = mono[4] * mono[53];
  mono[91] = mono[3] * mono[54];
  mono[92] = mono[3] * mono[56];
  mono[93] = mono[2] * mono[58];
  mono[94] = mono[2] * mono[60];
  mono[95] = mono[4] * mono[61];
  mono[96] = mono[3] * mono[62];
  mono[97] = mono[4] * mono[63];
  mono[98] = mono[2] * mono[64];
  mono[99] = mono[3] * mono[65];
  mono[100] = mono[2] * mono[66];
  mono[101] = mono[6] * mono[62];
  mono[102] = mono[5] * mono[64];
  mono[103] = mono[5] * mono[66];
  mono[104] = mono[3] * mono[61];
  mono[105] = mono[4] * mono[62];
  mono[106] = mono[2] * mono[63];
  mono[107] = mono[4] * mono[64];
  mono[108] = mono[2] * mono[65];
  mono[109] = mono[3] * mono[66];
  mono[110] = mono[3] * mono[80];
}

fn f_monomials2(mono: &mut [f64; N_MONOS]) { 
  mono[111] = mono[2] * mono[81];
  mono[112] = mono[3] * mono[88];
  mono[113] = mono[2] * mono[83];
  mono[114] = mono[2] * mono[90];
  mono[115] = mono[2] * mono[91];
  mono[116] = mono[2] * mono[80];
  mono[117] = mono[3] * mono[81];
  mono[118] = mono[2] * mono[82];
  mono[119] = mono[4] * mono[83];
  mono[120] = mono[3] * mono[84];
  mono[121] = mono[4] * mono[85];
  mono[122] = mono[3] * mono[95];
  mono[123] = mono[3] * mono[105];
  mono[124] = mono[2] * mono[97];
  mono[125] = mono[2] * mono[107];
  mono[126] = mono[2] * mono[99];
  mono[127] = mono[2] * mono[109];
  mono[128] = mono[3] * mono[101];
  mono[129] = mono[4] * mono[101];
  mono[130] = mono[2] * mono[102];
  mono[131] = mono[4] * mono[102];
  mono[132] = mono[2] * mono[103];
  mono[133] = mono[3] * mono[103];
  mono[134] = mono[2] * mono[95];
  mono[135] = mono[2] * mono[96];
  mono[136] = mono[3] * mono[97];
  mono[137] = mono[3] * mono[98];
  mono[138] = mono[4] * mono[99];
  mono[139] = mono[4] * mono[100];
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
  mono[7] = r[2];
  mono[8] = r[4];
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


