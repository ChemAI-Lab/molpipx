#![allow(unused_variables)]
use super::monomials_3_1_5::*;


pub const N_POLYS: usize = 103;

// File created from data/molecule_A3B/MOL_3_1_5.POLY 


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
    poly[51] = poly[9] * poly[6];
    poly[52] = poly[13] * poly[3];
    poly[53] = poly[9] * poly[4];
    poly[54] = poly[9] * poly[5];
    poly[55] = poly[1] * poly[24] - poly[51];
    poly[56] = poly[3] * poly[12] - poly[51] - poly[55] - poly[51];
    poly[57] = poly[3] * poly[14] - poly[51];
    poly[58] = poly[13] * poly[7];
    poly[59] = poly[9] * poly[8];
    poly[60] = poly[2] * poly[24] - poly[52];
    poly[61] = poly[13] * poly[4];
    poly[62] = poly[4] * poly[14] - poly[58];
    poly[63] = poly[6] * poly[11] - poly[52];
    poly[64] = poly[13] * poly[5];
    poly[65] = poly[9] * poly[3];
    poly[66] = poly[9] * poly[7];
    poly[67] = poly[3] * poly[16] - poly[53];
    poly[68] = poly[4] * poly[15] - poly[54] - poly[67];
    poly[69] = poly[2] * poly[39] - poly[68];
    poly[70] = poly[1] * poly[29] - poly[54] - poly[68];
    poly[71] = poly[7] * poly[11] - poly[53];
    poly[72] = poly[1] * poly[31] - poly[56] - poly[55];
    poly[73] = poly[1] * poly[32] - poly[57];
    poly[74] = poly[3] * poly[18] - poly[59];
    poly[75] = poly[2] * poly[29] - poly[57] - poly[55];
    poly[76] = poly[1] * poly[34] - poly[59];
    poly[77] = poly[6] * poly[16] - poly[58];
    poly[78] = poly[1] * poly[36] - poly[63] - poly[60];
    poly[79] = poly[2] * poly[32] - poly[58];
    poly[80] = poly[6] * poly[12] - poly[64] - poly[61] - poly[61];
    poly[81] = poly[13] * poly[6];
    poly[82] = poly[1] * poly[45] - poly[80];
    poly[83] = poly[2] * poly[33] - poly[62] - poly[60];
    poly[84] = poly[2] * poly[34] - poly[63];
    poly[85] = poly[6] * poly[18] - poly[61];
    poly[86] = poly[2] * poly[36] - poly[64] - poly[80];
    poly[87] = poly[13] * poly[8];
    poly[88] = poly[8] * poly[14] - poly[61];
    poly[89] = poly[1] * poly[39] - poly[65];
    poly[90] = poly[3] * poly[21] - poly[66];
    poly[91] = poly[1] * poly[41] - poly[67];
    poly[92] = poly[2] * poly[49] - poly[91];
    poly[93] = poly[1] * poly[43] - poly[74];
    poly[94] = poly[8] * poly[21] - poly[93];
    poly[95] = poly[1] * poly[46] - poly[83];
    poly[96] = poly[7] * poly[22] - poly[95];
    poly[97] = poly[2] * poly[45] - poly[81];
    poly[98] = poly[2] * poly[46] - poly[85];
    poly[99] = poly[1] * poly[50] - poly[98];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[6] * poly[22] - poly[87];
    poly[101] = poly[1] * poly[49] - poly[90];
    poly[102] = poly[2] * poly[50] - poly[100];
}

// Total number of monomials = 103 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);

    return poly;
}
