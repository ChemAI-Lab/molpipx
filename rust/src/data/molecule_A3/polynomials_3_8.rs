#![allow(unused_variables)]
use super::monomials_3_8::*;


pub const N_POLYS: usize = 41;

// File created from data/molecule_A3/MOL_3_8.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3];
    poly[2] = mono[4] + mono[5] + mono[6];
    poly[3] = poly[1] * poly[1] - poly[2] - poly[2];
    poly[4] = mono[7];
    poly[5] = poly[1] * poly[2] - poly[4] - poly[4] - poly[4];
    poly[6] = poly[1] * poly[3] - poly[5];
    poly[7] = poly[4] * poly[1];
    poly[8] = poly[2] * poly[2] - poly[7] - poly[7];
    poly[9] = poly[2] * poly[3] - poly[7];
    poly[10] = poly[1] * poly[6] - poly[9];
    poly[11] = poly[4] * poly[2];
    poly[12] = poly[4] * poly[3];
    poly[13] = poly[1] * poly[8] - poly[11];
    poly[14] = poly[2] * poly[6] - poly[12];
    poly[15] = poly[1] * poly[10] - poly[14];
    poly[16] = poly[4] * poly[4];
    poly[17] = poly[4] * poly[5];
    poly[18] = poly[4] * poly[6];
    poly[19] = poly[2] * poly[8] - poly[17];
    poly[20] = poly[1] * poly[13] - poly[17] - poly[19] - poly[19];
    poly[21] = poly[2] * poly[10] - poly[18];
    poly[22] = poly[1] * poly[15] - poly[21];
    poly[23] = poly[4] * poly[7];
    poly[24] = poly[4] * poly[8];
    poly[25] = poly[4] * poly[9];
    poly[26] = poly[4] * poly[10];
    poly[27] = poly[1] * poly[19] - poly[24];
    poly[28] = poly[6] * poly[8] - poly[23];
    poly[29] = poly[2] * poly[15] - poly[26];
    poly[30] = poly[1] * poly[22] - poly[29];
    poly[31] = poly[4] * poly[11];
    poly[32] = poly[4] * poly[12];
    poly[33] = poly[4] * poly[13];
    poly[34] = poly[4] * poly[14];
    poly[35] = poly[4] * poly[15];
    poly[36] = poly[2] * poly[19] - poly[33];
    poly[37] = poly[3] * poly[19] - poly[31];
    poly[38] = poly[8] * poly[10] - poly[32];
    poly[39] = poly[2] * poly[22] - poly[35];
    poly[40] = poly[1] * poly[30] - poly[39];
}

// Total number of monomials = 41 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}