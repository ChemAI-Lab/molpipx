#![allow(unused_variables)]
use super::monomials_3_2_3::*;


pub const N_POLYS: usize = 48;

// File created from data/molecule_A3B2/MOL_3_2_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3] + mono[4] + mono[5] + mono[6] + mono[7];
    poly[3] = mono[8] + mono[9] + mono[10];
    poly[4] = poly[1] * poly[2];
    poly[5] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16];
    poly[6] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
    poly[7] = mono[23] + mono[24] + mono[25];
    poly[8] = poly[1] * poly[3];
    poly[9] = mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31];
    poly[10] = poly[2] * poly[3] - poly[9];
    poly[11] = mono[32] + mono[33] + mono[34];
    poly[12] = poly[1] * poly[1];
    poly[13] = poly[2] * poly[2] - poly[7] - poly[6] - poly[5] - poly[7] - poly[6] - poly[5];
    poly[14] = poly[3] * poly[3] - poly[11] - poly[11];
    poly[15] = poly[1] * poly[5];
    poly[16] = poly[1] * poly[6];
    poly[17] = mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40];
    poly[18] = mono[41] + mono[42];
    poly[19] = poly[1] * poly[7];
    poly[20] = mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54];
    poly[21] = poly[1] * poly[9];
    poly[22] = mono[55] + mono[56] + mono[57];
    poly[23] = poly[1] * poly[10];
    poly[24] = mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69];
    poly[25] = mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81];
    poly[26] = poly[3] * poly[5] - poly[24];
    poly[27] = poly[3] * poly[6] - poly[25];
    poly[28] = poly[3] * poly[7] - poly[22];
    poly[29] = poly[1] * poly[11];
    poly[30] = mono[82] + mono[83] + mono[84] + mono[85] + mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93];
    poly[31] = mono[94];
    poly[32] = poly[2] * poly[11] - poly[30];
    poly[33] = poly[1] * poly[4];
    poly[34] = poly[1] * poly[13];
    poly[35] = poly[2] * poly[5] - poly[20] - poly[17] - poly[17];
    poly[36] = poly[2] * poly[6] - poly[20] - poly[18] - poly[17] - poly[18] - poly[18];
    poly[37] = poly[2] * poly[7] - poly[20];
    poly[38] = poly[1] * poly[8];
    poly[39] = poly[2] * poly[9] - poly[25] - poly[24] - poly[22] - poly[22];
    poly[40] = poly[3] * poly[13] - poly[39];
    poly[41] = poly[1] * poly[14];
    poly[42] = poly[3] * poly[9] - poly[30];
    poly[43] = poly[2] * poly[14] - poly[42];
    poly[44] = poly[3] * poly[11] - poly[31] - poly[31] - poly[31];
    poly[45] = poly[1] * poly[12];
    poly[46] = poly[2] * poly[13] - poly[37] - poly[36] - poly[35];
    poly[47] = poly[3] * poly[14] - poly[44];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}
