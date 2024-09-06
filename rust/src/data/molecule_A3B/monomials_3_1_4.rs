#![allow(unused_variables)]
// File created from data/molecule_A3B/MOL_3_1_4.MONO 
// Total number of monomials = 33 

pub const N_MONOS: usize = 33;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 6;
pub const N_ATOMS: usize = 4;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
  mono[7] = mono[1] * mono[2];
  mono[8] = mono[1] * mono[3];
  mono[9] = mono[2] * mono[3];
  mono[10] = mono[3] * mono[4];
  mono[11] = mono[2] * mono[5];
  mono[12] = mono[1] * mono[6];
  mono[13] = mono[4] * mono[5];
  mono[14] = mono[4] * mono[6];
  mono[15] = mono[5] * mono[6];
  mono[16] = mono[1] * mono[9];
  mono[17] = mono[1] * mono[10];
  mono[18] = mono[2] * mono[10];
  mono[19] = mono[1] * mono[11];
  mono[20] = mono[3] * mono[11];
  mono[21] = mono[2] * mono[12];
  mono[22] = mono[3] * mono[12];
  mono[23] = mono[2] * mono[13];
  mono[24] = mono[3] * mono[13];
  mono[25] = mono[1] * mono[14];
  mono[26] = mono[3] * mono[14];
  mono[27] = mono[1] * mono[15];
  mono[28] = mono[2] * mono[15];
  mono[29] = mono[4] * mono[15];
  mono[30] = mono[2] * mono[24];
  mono[31] = mono[1] * mono[26];
  mono[32] = mono[1] * mono[28];
}

fn f_init_monomials(mono: &mut [f64; N_MONOS], r: & [f64; N_DISTANCES]) { 
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");
  mono[0] = 1.0;
  mono[1] = r[5];
  mono[2] = r[4];
  mono[3] = r[2];
  mono[4] = r[3];
  mono[5] = r[1];
  mono[6] = r[0];
}

pub fn f_monomials(r: &[f64; N_DISTANCES]) -> [f64; N_MONOS] {

  let mut mono = [0.0; N_MONOS];

  f_init_monomials(&mut mono, r);

  f_monomials0(&mut mono);

  return mono; 
}


