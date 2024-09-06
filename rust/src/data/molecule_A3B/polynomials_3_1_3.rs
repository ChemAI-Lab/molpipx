#![allow(unused_variables)]
use super::monomials_3_1_3::*;


pub const N_POLYS: usize = 23;

// File created from data/molecule_A3B/MOL_3_1_3.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3];
    poly[2] = mono[4] + mono[5] + mono[6];
    poly[3] = mono[7] + mono[8] + mono[9];
    poly[4] = mono[10] + mono[11] + mono[12];
    poly[5] = poly[1] * poly[2] - poly[4];
    poly[6] = mono[13] + mono[14] + mono[15];
    poly[7] = poly[1] * poly[1] - poly[3] - poly[3];
    poly[8] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[9] = mono[16];
    poly[10] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
    poly[11] = poly[2] * poly[3] - poly[10];
    poly[12] = mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28];
    poly[13] = mono[29];
    poly[14] = poly[1] * poly[6] - poly[12];
    poly[15] = poly[1] * poly[3] - poly[9] - poly[9] - poly[9];
    poly[16] = poly[1] * poly[4] - poly[10];
    poly[17] = poly[2] * poly[7] - poly[16];
    poly[18] = poly[2] * poly[4] - poly[12];
    poly[19] = poly[1] * poly[8] - poly[18];
    poly[20] = poly[2] * poly[6] - poly[13] - poly[13] - poly[13];
    poly[21] = poly[1] * poly[7] - poly[15];
    poly[22] = poly[2] * poly[8] - poly[20];
}

// Total number of monomials = 23 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
