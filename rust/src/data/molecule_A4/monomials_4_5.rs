#![allow(unused_variables)]
// File created from data/molecule_A4/MOL_4_5.MONO 
// Total number of monomials = 63 

pub const N_MONOS: usize = 63;

// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 6;
pub const N_ATOMS: usize = 4;
pub const N_XYZ: usize = N_ATOMS * 3;

fn f_monomials0(mono: &mut [f64; N_MONOS]) { 
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
  mono[42] = mono[2] * mono[27];
  mono[43] = mono[1] * mono[31];
  mono[44] = mono[1] * mono[33];
  mono[45] = mono[1] * mono[23];
  mono[46] = mono[1] * mono[25];
  mono[47] = mono[1] * mono[26];
  mono[48] = mono[1] * mono[27];
  mono[49] = mono[1] * mono[40];
  mono[50] = mono[1] * mono[36];
  mono[51] = mono[2] * mono[31];
  mono[52] = mono[1] * mono[37];
  mono[53] = mono[2] * mono[37];
  mono[54] = mono[1] * mono[41];
  mono[55] = mono[2] * mono[41];
  mono[56] = mono[3] * mono[41];
}

fn f_monomials1(mono: &mut [f64; N_MONOS]) { 
  mono[57] = mono[1] * mono[42];
  mono[58] = mono[1] * mono[51];
  mono[59] = mono[1] * mono[53];
  mono[60] = mono[1] * mono[55];
  mono[61] = mono[1] * mono[56];
  mono[62] = mono[2] * mono[56];
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
  f_monomials1(&mut mono);

  return mono; 
}


