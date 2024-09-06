#![allow(unused_variables)]
use super::monomials_2_2_5::*;


pub const N_POLYS: usize = 153;

// File created from data/molecule_A2B2/MOL_2_2_5.POLY 


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
    poly[75] = poly[1] * poly[34];
    poly[76] = poly[1] * poly[38];
    poly[77] = poly[3] * poly[34];
    poly[78] = poly[1] * poly[33];
    poly[79] = poly[1] * poly[44];
    poly[80] = poly[1] * poly[46];
    poly[81] = poly[1] * poly[47];
    poly[82] = poly[34] * poly[2];
    poly[83] = poly[1] * poly[35];
    poly[84] = poly[1] * poly[36];
    poly[85] = poly[1] * poly[37];
    poly[86] = poly[1] * poly[50];
    poly[87] = poly[1] * poly[51];
    poly[88] = poly[3] * poly[44];
    poly[89] = poly[1] * poly[52];
    poly[90] = poly[3] * poly[46];
    poly[91] = poly[3] * poly[47];
    poly[92] = poly[1] * poly[54];
    poly[93] = poly[1] * poly[55];
    poly[94] = poly[1] * poly[56];
    poly[95] = poly[3] * poly[38];
    poly[96] = poly[1] * poly[39];
    poly[97] = poly[1] * poly[40];
    poly[98] = poly[1] * poly[41];
    poly[99] = poly[1] * poly[42];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[59];
    poly[101] = poly[1] * poly[43];
    poly[102] = poly[1] * poly[60];
    poly[103] = poly[1] * poly[45];
    poly[104] = poly[5] * poly[16] - poly[82];
    poly[105] = poly[6] * poly[16] - poly[82];
    poly[106] = poly[1] * poly[61];
    poly[107] = poly[7] * poly[16] - poly[82];
    poly[108] = poly[1] * poly[63];
    poly[109] = poly[1] * poly[64];
    poly[110] = poly[5] * poly[24] - poly[105];
    poly[111] = poly[1] * poly[65];
    poly[112] = poly[5] * poly[25] - poly[107];
    poly[113] = poly[6] * poly[25] - poly[107];
    poly[114] = poly[1] * poly[48];
    poly[115] = poly[1] * poly[49];
    poly[116] = poly[3] * poly[59];
    poly[117] = poly[3] * poly[60];
    poly[118] = poly[3] * poly[61];
    poly[119] = poly[1] * poly[67];
    poly[120] = poly[3] * poly[63];
    poly[121] = poly[3] * poly[64];
    poly[122] = poly[3] * poly[65];
    poly[123] = poly[1] * poly[53];
    poly[124] = poly[1] * poly[69];
    poly[125] = poly[3] * poly[50];
    poly[126] = poly[3] * poly[51];
    poly[127] = poly[3] * poly[52];
    poly[128] = poly[1] * poly[71];
    poly[129] = poly[3] * poly[54];
    poly[130] = poly[3] * poly[55];
    poly[131] = poly[3] * poly[56];
    poly[132] = poly[1] * poly[57];
    poly[133] = poly[1] * poly[58];
    poly[134] = poly[1] * poly[62];
    poly[135] = poly[2] * poly[59] - poly[104];
    poly[136] = poly[2] * poly[60] - poly[105];
    poly[137] = poly[2] * poly[61] - poly[107];
    poly[138] = poly[1] * poly[73];
    poly[139] = poly[5] * poly[31] - poly[113];
    poly[140] = poly[6] * poly[31] - poly[112];
    poly[141] = poly[7] * poly[31] - poly[110];
    poly[142] = poly[1] * poly[66];
    poly[143] = poly[3] * poly[73];
    poly[144] = poly[1] * poly[68];
    poly[145] = poly[3] * poly[67];
    poly[146] = poly[1] * poly[70];
    poly[147] = poly[3] * poly[69];
    poly[148] = poly[1] * poly[74];
    poly[149] = poly[3] * poly[71];
}

fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[72];
    poly[151] = poly[2] * poly[73] - poly[141] - poly[140] - poly[139];
    poly[152] = poly[3] * poly[74];
}

// Total number of monomials = 153 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);

    return poly;
}
