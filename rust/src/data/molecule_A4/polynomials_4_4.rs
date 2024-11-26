#![allow(unused_variables)]
use super::monomials_4_4::*;


pub const N_POLYS: usize = 22;

// File created from data/molecule_A4/MOL_4_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3] + mono[4] + mono[5] + mono[6];
    poly[2] = mono[7] + mono[8] + mono[9];
    poly[3] = mono[10] + mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16] + mono[17] + mono[18] + mono[19] + mono[20] + mono[21];
    poly[4] = poly[1] * poly[1] - poly[3] - poly[2] - poly[3] - poly[2];
    poly[5] = mono[22] + mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31] + mono[32] + mono[33];
    poly[6] = mono[34] + mono[35] + mono[36] + mono[37];
    poly[7] = mono[38] + mono[39] + mono[40] + mono[41];
    poly[8] = poly[1] * poly[2] - poly[5];
    poly[9] = poly[1] * poly[3] - poly[6] - poly[7] - poly[5] - poly[6] - poly[7] - poly[5] - poly[6] - poly[7];
    poly[10] = poly[1] * poly[4] - poly[9] - poly[8];
    poly[11] = mono[42] + mono[43] + mono[44];
    poly[12] = mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54] + mono[55] + mono[56];
    poly[13] = poly[2] * poly[3] - poly[12];
    poly[14] = poly[1] * poly[5] - poly[12] - poly[11] - poly[13] - poly[12] - poly[11] - poly[11] - poly[11];
    poly[15] = poly[1] * poly[6] - poly[12];
    poly[16] = poly[1] * poly[7] - poly[12];
    poly[17] = poly[2] * poly[2] - poly[11] - poly[11];
    poly[18] = poly[3] * poly[3] - poly[12] - poly[11] - poly[15] - poly[16] - poly[14] - poly[12] - poly[11] - poly[15] - poly[16] - poly[14] - poly[12] - poly[11] - poly[12] - poly[11];
    poly[19] = poly[2] * poly[4] - poly[14];
    poly[20] = poly[3] * poly[4] - poly[15] - poly[16] - poly[13];
    poly[21] = poly[1] * poly[10] - poly[20] - poly[19];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
