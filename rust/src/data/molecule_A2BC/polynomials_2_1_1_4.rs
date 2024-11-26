#![allow(unused_variables)]
use super::monomials_2_1_1_4::*;


pub const N_POLYS: usize = 120;

// File created from data/molecule_A2BC/MOL_2_1_1_4.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2] + mono[3];
    poly[3] = mono[4] + mono[5];
    poly[4] = mono[6];
    poly[5] = poly[1] * poly[2];
    poly[6] = mono[7];
    poly[7] = poly[1] * poly[3];
    poly[8] = mono[8] + mono[9];
    poly[9] = mono[10];
    poly[10] = poly[2] * poly[3] - poly[8];
    poly[11] = poly[1] * poly[4];
    poly[12] = poly[4] * poly[2];
    poly[13] = poly[4] * poly[3];
    poly[14] = poly[1] * poly[1];
    poly[15] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[16] = poly[3] * poly[3] - poly[9] - poly[9];
    poly[17] = poly[4] * poly[4];
    poly[18] = poly[1] * poly[6];
    poly[19] = poly[1] * poly[8];
    poly[20] = poly[1] * poly[9];
    poly[21] = poly[1] * poly[10];
    poly[22] = poly[6] * poly[3];
    poly[23] = poly[9] * poly[2];
    poly[24] = poly[1] * poly[12];
    poly[25] = poly[4] * poly[6];
    poly[26] = poly[1] * poly[13];
    poly[27] = poly[4] * poly[8];
    poly[28] = poly[4] * poly[9];
    poly[29] = poly[4] * poly[10];
    poly[30] = poly[1] * poly[5];
    poly[31] = poly[1] * poly[15];
    poly[32] = poly[6] * poly[2];
    poly[33] = poly[1] * poly[7];
    poly[34] = poly[2] * poly[8] - poly[22];
    poly[35] = poly[2] * poly[10] - poly[22];
    poly[36] = poly[1] * poly[16];
    poly[37] = poly[3] * poly[8] - poly[23];
    poly[38] = poly[9] * poly[3];
    poly[39] = poly[2] * poly[16] - poly[37];
    poly[40] = poly[1] * poly[11];
    poly[41] = poly[4] * poly[15];
    poly[42] = poly[4] * poly[16];
    poly[43] = poly[1] * poly[17];
    poly[44] = poly[4] * poly[12];
    poly[45] = poly[4] * poly[13];
    poly[46] = poly[1] * poly[14];
    poly[47] = poly[2] * poly[15] - poly[32];
    poly[48] = poly[3] * poly[16] - poly[38];
    poly[49] = poly[4] * poly[17];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[22];
    poly[51] = poly[1] * poly[23];
    poly[52] = poly[6] * poly[9];
    poly[53] = poly[1] * poly[25];
    poly[54] = poly[1] * poly[27];
    poly[55] = poly[1] * poly[28];
    poly[56] = poly[1] * poly[29];
    poly[57] = poly[4] * poly[22];
    poly[58] = poly[4] * poly[23];
    poly[59] = poly[1] * poly[18];
    poly[60] = poly[1] * poly[32];
    poly[61] = poly[1] * poly[19];
    poly[62] = poly[1] * poly[34];
    poly[63] = poly[1] * poly[20];
    poly[64] = poly[1] * poly[21];
    poly[65] = poly[6] * poly[8];
    poly[66] = poly[1] * poly[35];
    poly[67] = poly[6] * poly[10];
    poly[68] = poly[9] * poly[15];
    poly[69] = poly[1] * poly[37];
    poly[70] = poly[1] * poly[38];
    poly[71] = poly[9] * poly[8];
    poly[72] = poly[1] * poly[39];
    poly[73] = poly[6] * poly[16];
    poly[74] = poly[9] * poly[10];
    poly[75] = poly[1] * poly[24];
    poly[76] = poly[1] * poly[41];
    poly[77] = poly[4] * poly[32];
    poly[78] = poly[1] * poly[26];
    poly[79] = poly[4] * poly[34];
    poly[80] = poly[4] * poly[35];
    poly[81] = poly[1] * poly[42];
    poly[82] = poly[4] * poly[37];
    poly[83] = poly[4] * poly[38];
    poly[84] = poly[4] * poly[39];
    poly[85] = poly[1] * poly[44];
    poly[86] = poly[4] * poly[25];
    poly[87] = poly[1] * poly[45];
    poly[88] = poly[4] * poly[27];
    poly[89] = poly[4] * poly[28];
    poly[90] = poly[4] * poly[29];
    poly[91] = poly[1] * poly[30];
    poly[92] = poly[1] * poly[31];
    poly[93] = poly[6] * poly[6];
    poly[94] = poly[1] * poly[47];
    poly[95] = poly[6] * poly[15];
    poly[96] = poly[1] * poly[33];
    poly[97] = poly[2] * poly[34] - poly[65];
    poly[98] = poly[2] * poly[35] - poly[67];
    poly[99] = poly[1] * poly[36];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[2] * poly[37] - poly[73];
    poly[101] = poly[9] * poly[9];
    poly[102] = poly[2] * poly[39] - poly[73];
    poly[103] = poly[1] * poly[48];
    poly[104] = poly[3] * poly[37] - poly[71];
    poly[105] = poly[9] * poly[16];
    poly[106] = poly[2] * poly[48] - poly[104];
    poly[107] = poly[1] * poly[40];
    poly[108] = poly[4] * poly[47];
    poly[109] = poly[4] * poly[48];
    poly[110] = poly[1] * poly[43];
    poly[111] = poly[4] * poly[41];
    poly[112] = poly[4] * poly[42];
    poly[113] = poly[1] * poly[49];
    poly[114] = poly[4] * poly[44];
    poly[115] = poly[4] * poly[45];
    poly[116] = poly[1] * poly[46];
    poly[117] = poly[2] * poly[47] - poly[95];
    poly[118] = poly[3] * poly[48] - poly[105];
    poly[119] = poly[4] * poly[49];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);

    return poly;
}
