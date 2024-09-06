#![allow(unused_variables)]
use super::monomials_1_1_1_5::*;


pub const N_POLYS: usize = 56;

// File created from data/molecule_ABC/MOL_1_1_1_5.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = poly[1] * poly[2];
    poly[5] = poly[1] * poly[3];
    poly[6] = poly[2] * poly[3];
    poly[7] = poly[1] * poly[1];
    poly[8] = poly[2] * poly[2];
    poly[9] = poly[3] * poly[3];
    poly[10] = poly[1] * poly[6];
    poly[11] = poly[1] * poly[4];
    poly[12] = poly[1] * poly[8];
    poly[13] = poly[1] * poly[5];
    poly[14] = poly[2] * poly[6];
    poly[15] = poly[1] * poly[9];
    poly[16] = poly[2] * poly[9];
    poly[17] = poly[1] * poly[7];
    poly[18] = poly[2] * poly[8];
    poly[19] = poly[3] * poly[9];
    poly[20] = poly[1] * poly[10];
    poly[21] = poly[1] * poly[14];
    poly[22] = poly[1] * poly[16];
    poly[23] = poly[1] * poly[11];
    poly[24] = poly[1] * poly[12];
    poly[25] = poly[1] * poly[18];
    poly[26] = poly[1] * poly[13];
    poly[27] = poly[2] * poly[14];
    poly[28] = poly[1] * poly[15];
    poly[29] = poly[2] * poly[16];
    poly[30] = poly[1] * poly[19];
    poly[31] = poly[2] * poly[19];
    poly[32] = poly[1] * poly[17];
    poly[33] = poly[2] * poly[18];
    poly[34] = poly[3] * poly[19];
    poly[35] = poly[1] * poly[20];
    poly[36] = poly[1] * poly[21];
    poly[37] = poly[1] * poly[27];
    poly[38] = poly[1] * poly[22];
    poly[39] = poly[1] * poly[29];
    poly[40] = poly[1] * poly[31];
    poly[41] = poly[1] * poly[23];
    poly[42] = poly[1] * poly[24];
    poly[43] = poly[1] * poly[25];
    poly[44] = poly[1] * poly[33];
    poly[45] = poly[1] * poly[26];
    poly[46] = poly[2] * poly[27];
    poly[47] = poly[1] * poly[28];
    poly[48] = poly[2] * poly[29];
    poly[49] = poly[1] * poly[30];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[2] * poly[31];
    poly[51] = poly[1] * poly[34];
    poly[52] = poly[2] * poly[34];
    poly[53] = poly[1] * poly[32];
    poly[54] = poly[2] * poly[33];
    poly[55] = poly[3] * poly[34];
}

// Total number of monomials = 56 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
