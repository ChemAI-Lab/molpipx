#![allow(unused_variables)]



// File created from data/molecule_A4/MOL_4_3.MONO 
pub const N_MONOS: usize = 42;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 6;
pub const N_ATOMS: usize = 4;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 42 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

  mono[0] = 1.0;
  mono[1] = r[5];
  mono[2] = r[4];
  mono[3] = r[3];
  mono[4] = r[2];
  mono[5] = r[1];
  mono[6] = r[0];
  mono[7] = mono[3] * mono[4];
  mono[8] = mono[2] * mono[5];
  mono[9] = mono[1] * mono[6];
  mono[10] = mono[1] * mono[2];
  mono[11] = mono[1] * mono[3];
  mono[12] = mono[2] * mono[3];
  mono[13] = mono[1] * mono[4];
  mono[14] = mono[2] * mono[4];
  mono[15] = mono[1] * mono[5];
  mono[16] = mono[3] * mono[5];
  mono[17] = mono[4] * mono[5];
  mono[18] = mono[2] * mono[6];
  mono[19] = mono[3] * mono[6];
  mono[20] = mono[4] * mono[6];
  mono[21] = mono[5] * mono[6];
  mono[22] = mono[1] * mono[7];
  mono[23] = mono[2] * mono[7];
  mono[24] = mono[1] * mono[8];
  mono[25] = mono[2] * mono[16];
  mono[26] = mono[2] * mono[17];
  mono[27] = mono[3] * mono[17];
  mono[28] = mono[1] * mono[18];
  mono[29] = mono[1] * mono[19];
  mono[30] = mono[1] * mono[20];
  mono[31] = mono[3] * mono[20];
  mono[32] = mono[1] * mono[21];
  mono[33] = mono[2] * mono[21];
  mono[34] = mono[1] * mono[12];
  mono[35] = mono[1] * mono[17];
  mono[36] = mono[2] * mono[20];
  mono[37] = mono[3] * mono[21];
  mono[38] = mono[1] * mono[14];
  mono[39] = mono[1] * mono[16];
  mono[40] = mono[2] * mono[19];
  mono[41] = mono[4] * mono[21];
  return mono;
}


