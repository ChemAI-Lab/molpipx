#![allow(unused_variables)]



// File created from data/molecule_A2B2C/MOL_2_2_1_8.MONO 
pub const N_MONOS: usize = 72;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 72 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

  mono[0] = 1.0;
  mono[1] = r[9];
  mono[2] = r[8];
  mono[3] = r[7];
  mono[4] = r[6];
  mono[5] = r[3];
  mono[6] = r[5];
  mono[7] = r[4];
  mono[8] = r[2];
  mono[9] = r[1];
  mono[10] = r[0];
  mono[11] = mono[1] * mono[2];
  mono[12] = mono[4] * mono[5];
  mono[13] = mono[2] * mono[6];
  mono[14] = mono[1] * mono[7];
  mono[15] = mono[2] * mono[8];
  mono[16] = mono[1] * mono[9];
  mono[17] = mono[5] * mono[6];
  mono[18] = mono[5] * mono[7];
  mono[19] = mono[4] * mono[8];
  mono[20] = mono[4] * mono[9];
  mono[21] = mono[7] * mono[8];
  mono[22] = mono[6] * mono[9];
  mono[23] = mono[6] * mono[8];
  mono[24] = mono[7] * mono[9];
  mono[25] = mono[6] * mono[7];
  mono[26] = mono[8] * mono[9];
  mono[27] = mono[2] * mono[17];
  mono[28] = mono[1] * mono[18];
  mono[29] = mono[2] * mono[19];
  mono[30] = mono[1] * mono[20];
  mono[31] = mono[2] * mono[23];
  mono[32] = mono[1] * mono[24];
  mono[33] = mono[5] * mono[25];
  mono[34] = mono[4] * mono[26];
  mono[35] = mono[6] * mono[21];
  mono[36] = mono[6] * mono[24];
  mono[37] = mono[6] * mono[26];
  mono[38] = mono[7] * mono[26];
  mono[39] = mono[14] * mono[19];
  mono[40] = mono[15] * mono[18];
  mono[41] = mono[13] * mono[20];
  mono[42] = mono[16] * mono[17];
  mono[43] = mono[2] * mono[35];
  mono[44] = mono[1] * mono[36];
  mono[45] = mono[2] * mono[37];
  mono[46] = mono[1] * mono[38];
  mono[47] = mono[5] * mono[35];
  mono[48] = mono[5] * mono[36];
  mono[49] = mono[4] * mono[37];
  mono[50] = mono[4] * mono[38];
  mono[51] = mono[6] * mono[38];
  mono[52] = mono[2] * mono[47];
  mono[53] = mono[1] * mono[48];
  mono[54] = mono[2] * mono[49];
  mono[55] = mono[1] * mono[50];
  mono[56] = mono[19] * mono[35];
  mono[57] = mono[17] * mono[37];
  mono[58] = mono[18] * mono[38];
  mono[59] = mono[20] * mono[36];
  mono[60] = mono[14] * mono[35];
  mono[61] = mono[13] * mono[36];
  mono[62] = mono[15] * mono[38];
  mono[63] = mono[16] * mono[37];
  mono[64] = mono[2] * mono[56];
  mono[65] = mono[2] * mono[57];
  mono[66] = mono[1] * mono[58];
  mono[67] = mono[1] * mono[59];
  mono[68] = mono[5] * mono[60];
  mono[69] = mono[5] * mono[61];
  mono[70] = mono[4] * mono[62];
  mono[71] = mono[4] * mono[63];
  return mono;
}


