#![allow(unused_variables)]



// File created from data/molecule_A3BC/MOL_3_1_1_3.MONO 
pub const N_MONOS: usize = 68;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 68 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

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
  mono[61] = mono[6] * mono[29];
  mono[62] = mono[7] * mono[29];
  mono[63] = mono[5] * mono[30];
  mono[64] = mono[7] * mono[30];
  mono[65] = mono[5] * mono[31];
  mono[66] = mono[6] * mono[31];
  mono[67] = mono[8] * mono[31];
  return mono;
}


