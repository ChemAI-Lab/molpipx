#![allow(unused_variables)]
use super::monomials_2_2_4::*;


pub const N_POLYS: usize = 75;

// File created from data/molecule_A2B2/MOL_2_2_4.POLY 


#[inline(never)]
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
    poly[33] = poly[1] * poly[16];
    poly[34] = mono[17];
    poly[35] = poly[1] * poly[18];
    poly[36] = poly[1] * poly[19];
    poly[37] = poly[1] * poly[20];
    poly[38] = poly[3] * poly[16];
    poly[39] = poly[1] * poly[13];
    poly[40] = poly[1] * poly[14];
    poly[41] = poly[1] * poly[15];
    poly[42] = poly[1] * poly[23];
    poly[43] = poly[1] * poly[24];
    poly[44] = poly[5] * poly[6];
    poly[45] = poly[1] * poly[25];
    poly[46] = poly[5] * poly[7];
    poly[47] = poly[6] * poly[7];
    poly[48] = poly[1] * poly[17];
    poly[49] = poly[1] * poly[27];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[3] * poly[23];
    poly[51] = poly[3] * poly[24];
    poly[52] = poly[3] * poly[25];
    poly[53] = poly[1] * poly[29];
    poly[54] = poly[3] * poly[18];
    poly[55] = poly[3] * poly[19];
    poly[56] = poly[3] * poly[20];
    poly[57] = poly[1] * poly[21];
    poly[58] = poly[1] * poly[22];
    poly[59] = poly[5] * poly[5] - poly[34] - poly[34];
    poly[60] = poly[6] * poly[6] - poly[34] - poly[34];
    poly[61] = poly[7] * poly[7] - poly[34] - poly[34];
    poly[62] = poly[1] * poly[31];
    poly[63] = poly[5] * poly[11] - poly[47];
    poly[64] = poly[6] * poly[11] - poly[46];
    poly[65] = poly[7] * poly[11] - poly[44];
    poly[66] = poly[1] * poly[26];
    poly[67] = poly[3] * poly[31];
    poly[68] = poly[1] * poly[28];
    poly[69] = poly[3] * poly[27];
    poly[70] = poly[1] * poly[32];
    poly[71] = poly[3] * poly[29];
    poly[72] = poly[1] * poly[30];
    poly[73] = poly[2] * poly[31] - poly[65] - poly[64] - poly[63];
    poly[74] = poly[3] * poly[32];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
