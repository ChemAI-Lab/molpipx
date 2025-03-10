#![allow(unused_variables)]



// File created from data/molecule_ABCDE/MOL_1_1_1_1_1_6.MONO 
pub const N_MONOS: usize = 11;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 11 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

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
  return mono;
}


