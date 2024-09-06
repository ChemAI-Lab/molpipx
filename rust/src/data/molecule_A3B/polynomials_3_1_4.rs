#![allow(unused_variables)]
use super::monomials_3_1_4::*;


pub const N_POLYS: usize = 51;

// File created from data/molecule_A3B/MOL_3_1_4.POLY 


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
    poly[23] = poly[9] * poly[2];
    poly[24] = mono[30] + mono[31] + mono[32];
    poly[25] = poly[3] * poly[6] - poly[24];
    poly[26] = poly[13] * poly[1];
    poly[27] = poly[9] * poly[1];
    poly[28] = poly[3] * poly[4] - poly[23];
    poly[29] = poly[1] * poly[10] - poly[23] - poly[28] - poly[23];
    poly[30] = poly[1] * poly[11] - poly[23];
    poly[31] = poly[1] * poly[12] - poly[25] - poly[24] - poly[24];
    poly[32] = poly[1] * poly[14] - poly[25];
    poly[33] = poly[4] * poly[5] - poly[25] - poly[31];
    poly[34] = poly[2] * poly[11] - poly[25];
    poly[35] = poly[4] * poly[6] - poly[26];
    poly[36] = poly[2] * poly[12] - poly[26] - poly[35] - poly[26];
    poly[37] = poly[13] * poly[2];
    poly[38] = poly[2] * poly[14] - poly[26];
    poly[39] = poly[3] * poly[3] - poly[27] - poly[27];
    poly[40] = poly[3] * poly[7] - poly[27];
    poly[41] = poly[1] * poly[16] - poly[28];
    poly[42] = poly[2] * poly[21] - poly[41];
    poly[43] = poly[1] * poly[18] - poly[33];
    poly[44] = poly[7] * poly[8] - poly[43];
    poly[45] = poly[6] * poly[6] - poly[37] - poly[37];
    poly[46] = poly[2] * poly[18] - poly[35];
    poly[47] = poly[1] * poly[22] - poly[46];
    poly[48] = poly[6] * poly[8] - poly[37];
    poly[49] = poly[1] * poly[21] - poly[40];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[2] * poly[22] - poly[48];
}

// Total number of monomials = 51 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
