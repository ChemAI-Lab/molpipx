#![allow(unused_variables)]
use super::monomials_4_7::*;


pub const N_POLYS: usize = 120;

// File created from data/molecule_A4/MOL_4_7.POLY 


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
    poly[22] = mono[57] + mono[58] + mono[59] + mono[60] + mono[61] + mono[62];
    poly[23] = poly[1] * poly[11] - poly[22];
    poly[24] = poly[2] * poly[6];
    poly[25] = poly[2] * poly[7];
    poly[26] = poly[1] * poly[12] - poly[22] - poly[24] - poly[25] - poly[22] - poly[22] - poly[22];
    poly[27] = poly[2] * poly[5] - poly[22] - poly[23] - poly[22];
    poly[28] = poly[3] * poly[5] - poly[22] - poly[26] - poly[24] - poly[25] - poly[23] - poly[22] - poly[24] - poly[25] - poly[23] - poly[22] - poly[22];
    poly[29] = poly[3] * poly[6] - poly[22] - poly[26] - poly[22];
    poly[30] = poly[3] * poly[7] - poly[22] - poly[26] - poly[22];
    poly[31] = poly[2] * poly[9] - poly[26] - poly[28];
    poly[32] = poly[1] * poly[14] - poly[26] - poly[23] - poly[28];
    poly[33] = poly[4] * poly[6] - poly[25];
    poly[34] = poly[4] * poly[7] - poly[24];
    poly[35] = poly[1] * poly[17] - poly[27];
    poly[36] = poly[1] * poly[18] - poly[29] - poly[30] - poly[28];
    poly[37] = poly[2] * poly[10] - poly[32];
    poly[38] = poly[3] * poly[10] - poly[33] - poly[34] - poly[31];
    poly[39] = poly[1] * poly[21] - poly[38] - poly[37];
    poly[40] = mono[63];
    poly[41] = mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69] + mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85] + mono[86] + mono[87];
    poly[42] = poly[1] * poly[22] - poly[40] - poly[41] - poly[40] - poly[40] - poly[40] - poly[40] - poly[40];
    poly[43] = poly[2] * poly[11] - poly[40] - poly[40] - poly[40];
    poly[44] = poly[2] * poly[12] - poly[41];
    poly[45] = poly[3] * poly[11] - poly[41];
    poly[46] = poly[5] * poly[6] - poly[41];
    poly[47] = poly[5] * poly[7] - poly[41];
    poly[48] = poly[6] * poly[7] - poly[40] - poly[40] - poly[40] - poly[40];
    poly[49] = poly[4] * poly[11] - poly[42];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[2] * poly[15] - poly[46];
    poly[51] = poly[2] * poly[16] - poly[47];
    poly[52] = poly[4] * poly[12] - poly[41] - poly[50] - poly[51];
    poly[53] = poly[2] * poly[14] - poly[42] - poly[49] - poly[42];
    poly[54] = poly[6] * poly[6] - poly[42] - poly[42];
    poly[55] = poly[7] * poly[7] - poly[42] - poly[42];
    poly[56] = poly[3] * poly[17] - poly[44];
    poly[57] = poly[2] * poly[18] - poly[48];
    poly[58] = poly[3] * poly[14] - poly[41] - poly[52] - poly[46] - poly[47] - poly[45] - poly[45];
    poly[59] = poly[6] * poly[9] - poly[41] - poly[52] - poly[47];
    poly[60] = poly[7] * poly[9] - poly[41] - poly[52] - poly[46];
    poly[61] = poly[2] * poly[20] - poly[52] - poly[58];
    poly[62] = poly[1] * poly[32] - poly[52] - poly[49] - poly[58];
    poly[63] = poly[6] * poly[10] - poly[51];
    poly[64] = poly[7] * poly[10] - poly[50];
    poly[65] = poly[2] * poly[17] - poly[43];
    poly[66] = poly[3] * poly[18] - poly[46] - poly[47] - poly[45] - poly[59] - poly[60] - poly[58];
    poly[67] = poly[2] * poly[19] - poly[49];
    poly[68] = poly[1] * poly[36] - poly[59] - poly[60] - poly[58] - poly[57] - poly[66] - poly[66];
    poly[69] = poly[2] * poly[21] - poly[62];
    poly[70] = poly[3] * poly[21] - poly[63] - poly[64] - poly[61];
    poly[71] = poly[1] * poly[39] - poly[70] - poly[69];
    poly[72] = poly[40] * poly[1];
    poly[73] = poly[2] * poly[22] - poly[72];
    poly[74] = poly[6] * poly[11];
    poly[75] = poly[7] * poly[11];
    poly[76] = poly[3] * poly[22] - poly[72] - poly[74] - poly[75] - poly[72] - poly[72] - poly[72];
    poly[77] = mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93] + mono[94] + mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100] + mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106] + mono[107] + mono[108] + mono[109] + mono[110] + mono[111];
    poly[78] = poly[1] * poly[42] - poly[72] - poly[76];
    poly[79] = poly[5] * poly[11] - poly[72] - poly[73] - poly[72];
    poly[80] = poly[2] * poly[26] - poly[76] - poly[77];
    poly[81] = poly[6] * poly[12] - poly[72] - poly[76] - poly[72];
    poly[82] = poly[7] * poly[12] - poly[72] - poly[76] - poly[72];
    poly[83] = poly[8] * poly[11] - poly[72];
    poly[84] = poly[6] * poly[17];
    poly[85] = poly[7] * poly[17];
    poly[86] = poly[9] * poly[11] - poly[76] - poly[77];
    poly[87] = poly[2] * poly[29] - poly[81];
    poly[88] = poly[6] * poly[14] - poly[75] - poly[75];
    poly[89] = poly[2] * poly[30] - poly[82];
    poly[90] = poly[7] * poly[14] - poly[74] - poly[74];
    poly[91] = poly[1] * poly[48] - poly[76] - poly[81] - poly[82];
    poly[92] = poly[10] * poly[11] - poly[78];
    poly[93] = poly[2] * poly[33] - poly[88];
    poly[94] = poly[2] * poly[34] - poly[90];
    poly[95] = poly[10] * poly[12] - poly[77] - poly[93] - poly[94];
    poly[96] = poly[2] * poly[27] - poly[73] - poly[79];
    poly[97] = poly[2] * poly[28] - poly[76] - poly[86];
    poly[98] = poly[1] * poly[53] - poly[80] - poly[79] - poly[97];
    poly[99] = poly[1] * poly[54] - poly[81];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[55] - poly[82];
    poly[101] = poly[5] * poly[18] - poly[76] - poly[91] - poly[87] - poly[89] - poly[86];
    poly[102] = poly[6] * poly[18] - poly[74] - poly[90];
    poly[103] = poly[7] * poly[18] - poly[75] - poly[88];
    poly[104] = poly[2] * poly[31] - poly[77] - poly[86];
    poly[105] = poly[2] * poly[36] - poly[91] - poly[101];
    poly[106] = poly[3] * poly[32] - poly[77] - poly[95] - poly[88] - poly[90] - poly[86];
    poly[107] = poly[4] * poly[29] - poly[82] - poly[80] - poly[99];
    poly[108] = poly[4] * poly[30] - poly[81] - poly[80] - poly[100];
    poly[109] = poly[2] * poly[38] - poly[95] - poly[106];
    poly[110] = poly[1] * poly[62] - poly[95] - poly[92] - poly[106];
    poly[111] = poly[6] * poly[21] - poly[94];
    poly[112] = poly[7] * poly[21] - poly[93];
    poly[113] = poly[1] * poly[65] - poly[96];
    poly[114] = poly[1] * poly[66] - poly[102] - poly[103] - poly[101];
    poly[115] = poly[2] * poly[37] - poly[92];
    poly[116] = poly[10] * poly[18] - poly[99] - poly[100] - poly[97];
    poly[117] = poly[2] * poly[39] - poly[110];
    poly[118] = poly[3] * poly[39] - poly[111] - poly[112] - poly[109];
    poly[119] = poly[1] * poly[71] - poly[118] - poly[117];
}

// Total number of monomials = 120 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);

    return poly;
}
