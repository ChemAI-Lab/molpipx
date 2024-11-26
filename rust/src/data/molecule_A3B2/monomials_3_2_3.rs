#![allow(unused_variables)]



// File created from data/molecule_A3B2/MOL_3_2_3.MONO 
pub const N_MONOS: usize = 95;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 95 


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
  return mono;
}


