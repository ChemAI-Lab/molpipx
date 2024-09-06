#![allow(unused_variables)]
// File created from data/molecule_A2B2/MOL_2_2_3.MONO 
// Total number of monomials = 17 

pub const N_MONOS: usize = 17;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 6;
pub const N_ATOMS: usize = 4;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
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
}

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[5];
  mono[2] = r[4];
  mono[3] = r[3];
  mono[4] = r[2];
  mono[5] = r[1];
  mono[6] = r[0];
}

pub fn f_monomials(r: &[f64; N_DISTANCES]) -> [f64; N_MONOS] {

  let mut mono = [0.0; N_MONOS];

  f_init_monomials(&mut mono, r);

  f_monomials0(&mut mono);

  return mono; 
}


