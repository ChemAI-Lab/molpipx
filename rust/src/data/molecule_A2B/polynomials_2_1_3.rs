#![allow(unused_variables)]
use super::monomials_2_1_3::*;


pub const N_POLYS: usize = 13;

// File created from data/molecule_A2B/MOL_2_1_3.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2];
    poly[2] = mono[3];
    poly[3] = mono[4];
    poly[4] = poly[2] * poly[1];
    poly[5] = poly[1] * poly[1] - poly[3] - poly[3];
    poly[6] = poly[2] * poly[2];
    poly[7] = poly[2] * poly[3];
    poly[8] = poly[3] * poly[1];
    poly[9] = poly[2] * poly[5];
    poly[10] = poly[2] * poly[4];
    poly[11] = poly[1] * poly[5] - poly[8];
    poly[12] = poly[2] * poly[6];
}

// Total number of monomials = 13 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
