#![allow(unused_variables)]
use super::monomials_1_1_1_6::*;


pub const N_POLYS: usize = 84;

// File created from data/molecule_ABC/MOL_1_1_1_6.POLY 


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
    poly[56] = poly[1] * poly[35];
    poly[57] = poly[1] * poly[36];
    poly[58] = poly[1] * poly[37];
    poly[59] = poly[1] * poly[46];
    poly[60] = poly[1] * poly[38];
    poly[61] = poly[1] * poly[39];
    poly[62] = poly[1] * poly[48];
    poly[63] = poly[1] * poly[40];
    poly[64] = poly[1] * poly[50];
    poly[65] = poly[1] * poly[52];
    poly[66] = poly[1] * poly[41];
    poly[67] = poly[1] * poly[42];
    poly[68] = poly[1] * poly[43];
    poly[69] = poly[1] * poly[44];
    poly[70] = poly[1] * poly[54];
    poly[71] = poly[1] * poly[45];
    poly[72] = poly[2] * poly[46];
    poly[73] = poly[1] * poly[47];
    poly[74] = poly[2] * poly[48];
    poly[75] = poly[1] * poly[49];
    poly[76] = poly[2] * poly[50];
    poly[77] = poly[1] * poly[51];
    poly[78] = poly[2] * poly[52];
    poly[79] = poly[1] * poly[55];
    poly[80] = poly[2] * poly[55];
    poly[81] = poly[1] * poly[53];
    poly[82] = poly[2] * poly[54];
    poly[83] = poly[3] * poly[55];
}

// Total number of monomials = 84 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
