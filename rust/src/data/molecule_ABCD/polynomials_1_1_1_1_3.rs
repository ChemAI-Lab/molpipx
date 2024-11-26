#![allow(unused_variables)]
use super::monomials_1_1_1_1_3::*;


pub const N_POLYS: usize = 84;

// File created from data/molecule_ABCD/MOL_1_1_1_1_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4];
    poly[5] = mono[5];
    poly[6] = mono[6];
    poly[7] = poly[1] * poly[2];
    poly[8] = poly[1] * poly[3];
    poly[9] = poly[2] * poly[3];
    poly[10] = poly[1] * poly[4];
    poly[11] = poly[2] * poly[4];
    poly[12] = poly[3] * poly[4];
    poly[13] = poly[1] * poly[5];
    poly[14] = poly[2] * poly[5];
    poly[15] = poly[3] * poly[5];
    poly[16] = poly[4] * poly[5];
    poly[17] = poly[1] * poly[6];
    poly[18] = poly[2] * poly[6];
    poly[19] = poly[3] * poly[6];
    poly[20] = poly[4] * poly[6];
    poly[21] = poly[5] * poly[6];
    poly[22] = poly[1] * poly[1];
    poly[23] = poly[2] * poly[2];
    poly[24] = poly[3] * poly[3];
    poly[25] = poly[4] * poly[4];
    poly[26] = poly[5] * poly[5];
    poly[27] = poly[6] * poly[6];
    poly[28] = poly[1] * poly[9];
    poly[29] = poly[1] * poly[11];
    poly[30] = poly[1] * poly[12];
    poly[31] = poly[2] * poly[12];
    poly[32] = poly[1] * poly[14];
    poly[33] = poly[1] * poly[15];
    poly[34] = poly[2] * poly[15];
    poly[35] = poly[1] * poly[16];
    poly[36] = poly[2] * poly[16];
    poly[37] = poly[3] * poly[16];
    poly[38] = poly[1] * poly[18];
    poly[39] = poly[1] * poly[19];
    poly[40] = poly[2] * poly[19];
    poly[41] = poly[1] * poly[20];
    poly[42] = poly[2] * poly[20];
    poly[43] = poly[3] * poly[20];
    poly[44] = poly[1] * poly[21];
    poly[45] = poly[2] * poly[21];
    poly[46] = poly[3] * poly[21];
    poly[47] = poly[4] * poly[21];
    poly[48] = poly[1] * poly[7];
    poly[49] = poly[1] * poly[23];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[8];
    poly[51] = poly[2] * poly[9];
    poly[52] = poly[1] * poly[24];
    poly[53] = poly[2] * poly[24];
    poly[54] = poly[1] * poly[10];
    poly[55] = poly[2] * poly[11];
    poly[56] = poly[3] * poly[12];
    poly[57] = poly[1] * poly[25];
    poly[58] = poly[2] * poly[25];
    poly[59] = poly[3] * poly[25];
    poly[60] = poly[1] * poly[13];
    poly[61] = poly[2] * poly[14];
    poly[62] = poly[3] * poly[15];
    poly[63] = poly[4] * poly[16];
    poly[64] = poly[1] * poly[26];
    poly[65] = poly[2] * poly[26];
    poly[66] = poly[3] * poly[26];
    poly[67] = poly[4] * poly[26];
    poly[68] = poly[1] * poly[17];
    poly[69] = poly[2] * poly[18];
    poly[70] = poly[3] * poly[19];
    poly[71] = poly[4] * poly[20];
    poly[72] = poly[5] * poly[21];
    poly[73] = poly[1] * poly[27];
    poly[74] = poly[2] * poly[27];
    poly[75] = poly[3] * poly[27];
    poly[76] = poly[4] * poly[27];
    poly[77] = poly[5] * poly[27];
    poly[78] = poly[1] * poly[22];
    poly[79] = poly[2] * poly[23];
    poly[80] = poly[3] * poly[24];
    poly[81] = poly[4] * poly[25];
    poly[82] = poly[5] * poly[26];
    poly[83] = poly[6] * poly[27];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
