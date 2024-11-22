#![allow(unused_variables)]
use super::monomials_2_2_3::*;


pub const N_POLYS: usize = 33;

// File created from data/molecule_A2B2/MOL_2_2_3.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3] + mono[4] + mono[5];
    poly[3] = mono[6];
    poly[4] = poly[1] * poly[2];
    poly[5] = mono[7] + mono[8];
    poly[6] = mono[9] + mono[10];
    poly[7] = mono[11] + mono[12];
    poly[8] = poly[1] * poly[3];
    poly[9] = poly[3] * poly[2];
    poly[10] = poly[1] * poly[1];
    poly[11] = poly[2] * poly[2] - poly[7] - poly[6] - poly[5] - poly[7] - poly[6] - poly[5];
    poly[12] = poly[3] * poly[3];
    poly[13] = poly[1] * poly[5];
    poly[14] = poly[1] * poly[6];
    poly[15] = poly[1] * poly[7];
    poly[16] = mono[13] + mono[14] + mono[15] + mono[16];
    poly[17] = poly[1] * poly[9];
    poly[18] = poly[3] * poly[5];
    poly[19] = poly[3] * poly[6];
    poly[20] = poly[3] * poly[7];
    poly[21] = poly[1] * poly[4];
    poly[22] = poly[1] * poly[11];
    poly[23] = poly[2] * poly[5] - poly[16];
    poly[24] = poly[2] * poly[6] - poly[16];
    poly[25] = poly[2] * poly[7] - poly[16];
    poly[26] = poly[1] * poly[8];
    poly[27] = poly[3] * poly[11];
    poly[28] = poly[1] * poly[12];
    poly[29] = poly[3] * poly[9];
    poly[30] = poly[1] * poly[10];
    poly[31] = poly[2] * poly[11] - poly[25] - poly[24] - poly[23];
    poly[32] = poly[3] * poly[12];
}

// Total number of monomials = 33 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}