#![allow(unused_variables)]
// File created from data/molecule_ABC/MOL_1_1_1_3.MONO 
// Total number of monomials = 4 

pub const N_MONOS: usize = 4;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 3;
pub const N_ATOMS: usize = 3;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[2];
  mono[2] = r[1];
  mono[3] = r[0];
}

pub fn f_monomials(r: &[f64; N_DISTANCES]) -> [f64; N_MONOS] {

  let mut mono = [0.0; N_MONOS];

  f_init_monomials(&mut mono, r);


  return mono; 
}


