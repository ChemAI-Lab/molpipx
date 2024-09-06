#![allow(unused_variables)]
use super::monomials_2_1_1_3::*;


pub const N_POLYS: usize = 50;

// File created from data/molecule_A2BC/MOL_2_1_1_3.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3];
    poly[3] = mono[4] + mono[5];
    poly[4] = mono[6];
    poly[5] = poly[1] * poly[2];
    poly[6] = mono[7];
    poly[7] = poly[1] * poly[3];
    poly[8] = mono[8] + mono[9];
    poly[9] = mono[10];
    poly[10] = poly[2] * poly[3] - poly[8];
    poly[11] = poly[1] * poly[4];
    poly[12] = poly[4] * poly[2];
    poly[13] = poly[4] * poly[3];
    poly[14] = poly[1] * poly[1];
    poly[15] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[16] = poly[3] * poly[3] - poly[9] - poly[9];
    poly[17] = poly[4] * poly[4];
    poly[18] = poly[1] * poly[6];
    poly[19] = poly[1] * poly[8];
    poly[20] = poly[1] * poly[9];
    poly[21] = poly[1] * poly[10];
    poly[22] = poly[6] * poly[3];
    poly[23] = poly[9] * poly[2];
    poly[24] = poly[1] * poly[12];
    poly[25] = poly[4] * poly[6];
    poly[26] = poly[1] * poly[13];
    poly[27] = poly[4] * poly[8];
    poly[28] = poly[4] * poly[9];
    poly[29] = poly[4] * poly[10];
    poly[30] = poly[1] * poly[5];
    poly[31] = poly[1] * poly[15];
    poly[32] = poly[6] * poly[2];
    poly[33] = poly[1] * poly[7];
    poly[34] = poly[2] * poly[8] - poly[22];
    poly[35] = poly[2] * poly[10] - poly[22];
    poly[36] = poly[1] * poly[16];
    poly[37] = poly[3] * poly[8] - poly[23];
    poly[38] = poly[9] * poly[3];
    poly[39] = poly[2] * poly[16] - poly[37];
    poly[40] = poly[1] * poly[11];
    poly[41] = poly[4] * poly[15];
    poly[42] = poly[4] * poly[16];
    poly[43] = poly[1] * poly[17];
    poly[44] = poly[4] * poly[12];
    poly[45] = poly[4] * poly[13];
    poly[46] = poly[1] * poly[14];
    poly[47] = poly[2] * poly[15] - poly[32];
    poly[48] = poly[3] * poly[16] - poly[38];
    poly[49] = poly[4] * poly[17];
}

// Total number of monomials = 50 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
