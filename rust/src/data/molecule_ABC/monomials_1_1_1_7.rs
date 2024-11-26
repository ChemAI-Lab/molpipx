#![allow(unused_variables)]



// File created from data/molecule_ABC/MOL_1_1_1_7.MONO 
pub const N_MONOS: usize = 4;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 3;
pub const N_ATOMS: usize = 3;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 4 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

  mono[0] = 1.0;
  mono[1] = r[2];
  mono[2] = r[1];
  mono[3] = r[0];
  return mono;
}


