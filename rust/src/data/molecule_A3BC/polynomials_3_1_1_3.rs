#![allow(unused_variables)]
use super::monomials_3_1_1_3::*;


pub const N_POLYS: usize = 75;

// File created from data/molecule_A3BC/MOL_3_1_1_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3] + mono[4];
    poly[3] = mono[5] + mono[6] + mono[7];
    poly[4] = mono[8] + mono[9] + mono[10];
    poly[5] = poly[1] * poly[2];
    poly[6] = mono[11] + mono[12] + mono[13];
    poly[7] = poly[1] * poly[3];
    poly[8] = mono[14] + mono[15] + mono[16] + mono[17] + mono[18] + mono[19];
    poly[9] = mono[20] + mono[21] + mono[22];
    poly[10] = poly[2] * poly[3] - poly[8];
    poly[11] = poly[1] * poly[4];
    poly[12] = mono[23] + mono[24] + mono[25];
    poly[13] = mono[26] + mono[27] + mono[28];
    poly[14] = poly[2] * poly[4] - poly[12];
    poly[15] = poly[3] * poly[4] - poly[13];
    poly[16] = mono[29] + mono[30] + mono[31];
    poly[17] = poly[1] * poly[1];
    poly[18] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[19] = poly[3] * poly[3] - poly[9] - poly[9];
    poly[20] = poly[4] * poly[4] - poly[16] - poly[16];
    poly[21] = poly[1] * poly[6];
    poly[22] = mono[32];
    poly[23] = poly[1] * poly[8];
    poly[24] = mono[33] + mono[34] + mono[35];
    poly[25] = poly[1] * poly[9];
    poly[26] = mono[36] + mono[37] + mono[38];
    poly[27] = mono[39];
    poly[28] = poly[1] * poly[10];
    poly[29] = poly[3] * poly[6] - poly[24];
    poly[30] = poly[2] * poly[9] - poly[26];
    poly[31] = poly[1] * poly[12];
    poly[32] = poly[1] * poly[13];
    poly[33] = mono[40] + mono[41] + mono[42];
    poly[34] = poly[1] * poly[14];
    poly[35] = mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48];
    poly[36] = poly[2] * poly[13] - poly[33];
    poly[37] = poly[4] * poly[6] - poly[35];
    poly[38] = poly[1] * poly[15];
    poly[39] = poly[3] * poly[12] - poly[33];
    poly[40] = mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54];
    poly[41] = poly[4] * poly[8] - poly[39] - poly[36];
    poly[42] = poly[4] * poly[9] - poly[40];
    poly[43] = poly[4] * poly[10] - poly[33];
    poly[44] = poly[1] * poly[16];
    poly[45] = mono[55] + mono[56] + mono[57] + mono[58] + mono[59] + mono[60];
    poly[46] = mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66];
    poly[47] = mono[67];
    poly[48] = poly[2] * poly[16] - poly[45];
    poly[49] = poly[3] * poly[16] - poly[46];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[5];
    poly[51] = poly[1] * poly[18];
    poly[52] = poly[2] * poly[6] - poly[22] - poly[22] - poly[22];
    poly[53] = poly[1] * poly[7];
    poly[54] = poly[2] * poly[8] - poly[29] - poly[24] - poly[24];
    poly[55] = poly[2] * poly[10] - poly[29];
    poly[56] = poly[1] * poly[19];
    poly[57] = poly[3] * poly[8] - poly[30] - poly[26] - poly[26];
    poly[58] = poly[3] * poly[9] - poly[27] - poly[27] - poly[27];
    poly[59] = poly[2] * poly[19] - poly[57];
    poly[60] = poly[1] * poly[11];
    poly[61] = poly[2] * poly[12] - poly[35];
    poly[62] = poly[3] * poly[13] - poly[40];
    poly[63] = poly[4] * poly[18] - poly[61];
    poly[64] = poly[4] * poly[19] - poly[62];
    poly[65] = poly[1] * poly[20];
    poly[66] = poly[4] * poly[12] - poly[45];
    poly[67] = poly[4] * poly[13] - poly[46];
    poly[68] = poly[2] * poly[20] - poly[66];
    poly[69] = poly[3] * poly[20] - poly[67];
    poly[70] = poly[4] * poly[16] - poly[47] - poly[47] - poly[47];
    poly[71] = poly[1] * poly[17];
    poly[72] = poly[2] * poly[18] - poly[52];
    poly[73] = poly[3] * poly[19] - poly[58];
    poly[74] = poly[4] * poly[20] - poly[70];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
