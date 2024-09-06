#![allow(unused_variables)]
use super::monomials_2_1_8::*;


pub const N_POLYS: usize = 95;

// File created from data/molecule_A2B/MOL_2_1_8.POLY 


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
    poly[13] = poly[2] * poly[8];
    poly[14] = poly[2] * poly[7];
    poly[15] = poly[3] * poly[3];
    poly[16] = poly[3] * poly[5];
    poly[17] = poly[2] * poly[11];
    poly[18] = poly[2] * poly[9];
    poly[19] = poly[2] * poly[10];
    poly[20] = poly[1] * poly[11] - poly[16];
    poly[21] = poly[2] * poly[12];
    poly[22] = poly[2] * poly[15];
    poly[23] = poly[2] * poly[16];
    poly[24] = poly[2] * poly[13];
    poly[25] = poly[2] * poly[14];
    poly[26] = poly[3] * poly[8];
    poly[27] = poly[3] * poly[11];
    poly[28] = poly[2] * poly[20];
    poly[29] = poly[2] * poly[17];
    poly[30] = poly[2] * poly[18];
    poly[31] = poly[2] * poly[19];
    poly[32] = poly[1] * poly[20] - poly[27];
    poly[33] = poly[2] * poly[21];
    poly[34] = poly[2] * poly[26];
    poly[35] = poly[2] * poly[27];
    poly[36] = poly[2] * poly[22];
    poly[37] = poly[2] * poly[23];
    poly[38] = poly[2] * poly[24];
    poly[39] = poly[2] * poly[25];
    poly[40] = poly[3] * poly[15];
    poly[41] = poly[3] * poly[16];
    poly[42] = poly[3] * poly[20];
    poly[43] = poly[2] * poly[32];
    poly[44] = poly[2] * poly[28];
    poly[45] = poly[2] * poly[29];
    poly[46] = poly[2] * poly[30];
    poly[47] = poly[2] * poly[31];
    poly[48] = poly[1] * poly[32] - poly[42];
    poly[49] = poly[2] * poly[33];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[2] * poly[40];
    poly[51] = poly[2] * poly[41];
    poly[52] = poly[2] * poly[42];
    poly[53] = poly[2] * poly[34];
    poly[54] = poly[2] * poly[35];
    poly[55] = poly[2] * poly[36];
    poly[56] = poly[2] * poly[37];
    poly[57] = poly[2] * poly[38];
    poly[58] = poly[2] * poly[39];
    poly[59] = poly[3] * poly[26];
    poly[60] = poly[3] * poly[27];
    poly[61] = poly[3] * poly[32];
    poly[62] = poly[2] * poly[48];
    poly[63] = poly[2] * poly[43];
    poly[64] = poly[2] * poly[44];
    poly[65] = poly[2] * poly[45];
    poly[66] = poly[2] * poly[46];
    poly[67] = poly[2] * poly[47];
    poly[68] = poly[1] * poly[48] - poly[61];
    poly[69] = poly[2] * poly[49];
    poly[70] = poly[2] * poly[59];
    poly[71] = poly[2] * poly[60];
    poly[72] = poly[2] * poly[61];
    poly[73] = poly[2] * poly[50];
    poly[74] = poly[2] * poly[51];
    poly[75] = poly[2] * poly[52];
    poly[76] = poly[2] * poly[53];
    poly[77] = poly[2] * poly[54];
    poly[78] = poly[2] * poly[55];
    poly[79] = poly[2] * poly[56];
    poly[80] = poly[2] * poly[57];
    poly[81] = poly[2] * poly[58];
    poly[82] = poly[3] * poly[40];
    poly[83] = poly[3] * poly[41];
    poly[84] = poly[3] * poly[42];
    poly[85] = poly[3] * poly[48];
    poly[86] = poly[2] * poly[68];
    poly[87] = poly[2] * poly[62];
    poly[88] = poly[2] * poly[63];
    poly[89] = poly[2] * poly[64];
    poly[90] = poly[2] * poly[65];
    poly[91] = poly[2] * poly[66];
    poly[92] = poly[2] * poly[67];
    poly[93] = poly[1] * poly[68] - poly[85];
    poly[94] = poly[2] * poly[69];
}

// Total number of monomials = 95 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
