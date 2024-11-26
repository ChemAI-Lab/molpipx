#![allow(unused_variables)]
use super::monomials_2_2_1_3::*;


pub const N_POLYS: usize = 102;

// File created from data/molecule_A2B2C/MOL_2_2_1_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2];
    poly[2] = mono[3];
    poly[3] = mono[4] + mono[5];
    poly[4] = mono[6] + mono[7] + mono[8] + mono[9];
    poly[5] = mono[10];
    poly[6] = mono[11];
    poly[7] = poly[2] * poly[1];
    poly[8] = poly[1] * poly[3];
    poly[9] = poly[2] * poly[3];
    poly[10] = mono[12];
    poly[11] = mono[13] + mono[14] + mono[15] + mono[16];
    poly[12] = poly[1] * poly[4] - poly[11];
    poly[13] = poly[2] * poly[4];
    poly[14] = mono[17] + mono[18] + mono[19] + mono[20];
    poly[15] = mono[21] + mono[22];
    poly[16] = mono[23] + mono[24];
    poly[17] = poly[3] * poly[4] - poly[14];
    poly[18] = mono[25] + mono[26];
    poly[19] = poly[5] * poly[1];
    poly[20] = poly[2] * poly[5];
    poly[21] = poly[5] * poly[3];
    poly[22] = poly[5] * poly[4];
    poly[23] = poly[1] * poly[1] - poly[6] - poly[6];
    poly[24] = poly[2] * poly[2];
    poly[25] = poly[3] * poly[3] - poly[10] - poly[10];
    poly[26] = poly[4] * poly[4] - poly[18] - poly[16] - poly[15] - poly[18] - poly[16] - poly[15];
    poly[27] = poly[5] * poly[5];
    poly[28] = poly[2] * poly[6];
    poly[29] = poly[6] * poly[3];
    poly[30] = poly[2] * poly[8];
    poly[31] = poly[10] * poly[1];
    poly[32] = poly[2] * poly[10];
    poly[33] = poly[6] * poly[4];
    poly[34] = poly[2] * poly[11];
    poly[35] = poly[2] * poly[12];
    poly[36] = mono[27] + mono[28] + mono[29] + mono[30];
    poly[37] = poly[1] * poly[14] - poly[36];
    poly[38] = poly[2] * poly[14];
    poly[39] = poly[1] * poly[15];
    poly[40] = poly[2] * poly[15];
    poly[41] = mono[31] + mono[32];
    poly[42] = poly[1] * poly[16] - poly[41];
    poly[43] = poly[2] * poly[16];
    poly[44] = poly[3] * poly[11] - poly[36];
    poly[45] = poly[1] * poly[17] - poly[44];
    poly[46] = poly[2] * poly[17];
    poly[47] = poly[10] * poly[4];
    poly[48] = poly[3] * poly[15];
    poly[49] = poly[3] * poly[16];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[18];
    poly[51] = poly[2] * poly[18];
    poly[52] = mono[33] + mono[34];
    poly[53] = mono[35] + mono[36] + mono[37] + mono[38];
    poly[54] = poly[3] * poly[18] - poly[52];
    poly[55] = poly[5] * poly[6];
    poly[56] = poly[2] * poly[19];
    poly[57] = poly[5] * poly[8];
    poly[58] = poly[2] * poly[21];
    poly[59] = poly[5] * poly[10];
    poly[60] = poly[5] * poly[11];
    poly[61] = poly[5] * poly[12];
    poly[62] = poly[2] * poly[22];
    poly[63] = poly[5] * poly[14];
    poly[64] = poly[5] * poly[15];
    poly[65] = poly[5] * poly[16];
    poly[66] = poly[5] * poly[17];
    poly[67] = poly[5] * poly[18];
    poly[68] = poly[6] * poly[1];
    poly[69] = poly[2] * poly[23];
    poly[70] = poly[2] * poly[7];
    poly[71] = poly[3] * poly[23];
    poly[72] = poly[2] * poly[9];
    poly[73] = poly[1] * poly[25];
    poly[74] = poly[2] * poly[25];
    poly[75] = poly[10] * poly[3];
    poly[76] = poly[1] * poly[11] - poly[33];
    poly[77] = poly[1] * poly[12] - poly[33];
    poly[78] = poly[2] * poly[13];
    poly[79] = poly[3] * poly[14] - poly[47];
    poly[80] = poly[3] * poly[17] - poly[47];
    poly[81] = poly[4] * poly[11] - poly[50] - poly[41] - poly[39] - poly[41];
    poly[82] = poly[1] * poly[26] - poly[81];
    poly[83] = poly[2] * poly[26];
    poly[84] = poly[4] * poly[14] - poly[52] - poly[49] - poly[48] - poly[52];
    poly[85] = poly[4] * poly[15] - poly[53];
    poly[86] = poly[4] * poly[16] - poly[53];
    poly[87] = poly[3] * poly[26] - poly[84];
    poly[88] = poly[4] * poly[18] - poly[53];
    poly[89] = poly[5] * poly[23];
    poly[90] = poly[2] * poly[20];
    poly[91] = poly[5] * poly[25];
    poly[92] = poly[5] * poly[26];
    poly[93] = poly[5] * poly[19];
    poly[94] = poly[2] * poly[27];
    poly[95] = poly[5] * poly[21];
    poly[96] = poly[5] * poly[22];
    poly[97] = poly[1] * poly[23] - poly[68];
    poly[98] = poly[2] * poly[24];
    poly[99] = poly[3] * poly[25] - poly[75];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[4] * poly[26] - poly[88] - poly[86] - poly[85];
    poly[101] = poly[5] * poly[27];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);

    return poly;
}
