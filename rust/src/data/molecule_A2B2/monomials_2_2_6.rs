#![allow(unused_variables)]



// File created from data/molecule_A2B2/MOL_2_2_6.MONO 
pub const N_MONOS: usize = 18;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 6;
pub const N_ATOMS: usize = 4;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 18 


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
  mono[9] = mono[2] * mono[4];
  mono[10] = mono[3] * mono[5];
  mono[11] = mono[2] * mono[3];
  mono[12] = mono[4] * mono[5];
  mono[13] = mono[2] * mono[7];
  mono[14] = mono[2] * mono[10];
  mono[15] = mono[2] * mono[12];
  mono[16] = mono[3] * mono[12];
  mono[17] = mono[2] * mono[16];
  return mono;
}


