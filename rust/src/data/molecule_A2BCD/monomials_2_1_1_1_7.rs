#![allow(unused_variables)]
// File created from data/molecule_A2BCD/MOL_2_1_1_1_7.MONO 
// Total number of monomials = 22 

pub const N_MONOS: usize = 22;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
  mono[11] = mono[4] * mono[5];
  mono[12] = mono[5] * mono[6];
  mono[13] = mono[4] * mono[7];
  mono[14] = mono[6] * mono[7];
  mono[15] = mono[5] * mono[8];
  mono[16] = mono[4] * mono[9];
  mono[17] = mono[7] * mono[8];
  mono[18] = mono[6] * mono[9];
  mono[19] = mono[8] * mono[9];
  mono[20] = mono[5] * mono[17];
  mono[21] = mono[4] * mono[18];
}

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[9];
  mono[2] = r[8];
  mono[3] = r[7];
  mono[4] = r[6];
  mono[5] = r[3];
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

  return mono; 
}


