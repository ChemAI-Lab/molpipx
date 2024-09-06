#![allow(unused_variables)]
use super::monomials_2_1_1_1_3::*;


pub const N_POLYS: usize = 168;

// File created from data/molecule_A2BCD/MOL_2_1_1_1_3.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4] + mono[5];
    poly[5] = mono[6] + mono[7];
    poly[6] = mono[8] + mono[9];
    poly[7] = mono[10];
    poly[8] = poly[1] * poly[2];
    poly[9] = poly[1] * poly[3];
    poly[10] = poly[2] * poly[3];
    poly[11] = poly[1] * poly[4];
    poly[12] = poly[2] * poly[4];
    poly[13] = poly[3] * poly[4];
    poly[14] = mono[11];
    poly[15] = poly[1] * poly[5];
    poly[16] = poly[2] * poly[5];
    poly[17] = poly[3] * poly[5];
    poly[18] = mono[12] + mono[13];
    poly[19] = mono[14];
    poly[20] = poly[4] * poly[5] - poly[18];
    poly[21] = poly[1] * poly[6];
    poly[22] = poly[2] * poly[6];
    poly[23] = poly[3] * poly[6];
    poly[24] = mono[15] + mono[16];
    poly[25] = mono[17] + mono[18];
    poly[26] = mono[19];
    poly[27] = poly[4] * poly[6] - poly[24];
    poly[28] = poly[5] * poly[6] - poly[25];
    poly[29] = poly[1] * poly[7];
    poly[30] = poly[2] * poly[7];
    poly[31] = poly[3] * poly[7];
    poly[32] = poly[7] * poly[4];
    poly[33] = poly[7] * poly[5];
    poly[34] = poly[7] * poly[6];
    poly[35] = poly[1] * poly[1];
    poly[36] = poly[2] * poly[2];
    poly[37] = poly[3] * poly[3];
    poly[38] = poly[4] * poly[4] - poly[14] - poly[14];
    poly[39] = poly[5] * poly[5] - poly[19] - poly[19];
    poly[40] = poly[6] * poly[6] - poly[26] - poly[26];
    poly[41] = poly[7] * poly[7];
    poly[42] = poly[1] * poly[10];
    poly[43] = poly[1] * poly[12];
    poly[44] = poly[1] * poly[13];
    poly[45] = poly[2] * poly[13];
    poly[46] = poly[1] * poly[14];
    poly[47] = poly[2] * poly[14];
    poly[48] = poly[3] * poly[14];
    poly[49] = poly[1] * poly[16];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[1] * poly[17];
    poly[51] = poly[2] * poly[17];
    poly[52] = poly[1] * poly[18];
    poly[53] = poly[2] * poly[18];
    poly[54] = poly[3] * poly[18];
    poly[55] = poly[1] * poly[19];
    poly[56] = poly[2] * poly[19];
    poly[57] = poly[3] * poly[19];
    poly[58] = poly[1] * poly[20];
    poly[59] = poly[2] * poly[20];
    poly[60] = poly[3] * poly[20];
    poly[61] = poly[14] * poly[5];
    poly[62] = poly[19] * poly[4];
    poly[63] = poly[1] * poly[22];
    poly[64] = poly[1] * poly[23];
    poly[65] = poly[2] * poly[23];
    poly[66] = poly[1] * poly[24];
    poly[67] = poly[2] * poly[24];
    poly[68] = poly[3] * poly[24];
    poly[69] = poly[1] * poly[25];
    poly[70] = poly[2] * poly[25];
    poly[71] = poly[3] * poly[25];
    poly[72] = mono[20] + mono[21];
    poly[73] = poly[1] * poly[26];
    poly[74] = poly[2] * poly[26];
    poly[75] = poly[3] * poly[26];
    poly[76] = poly[1] * poly[27];
    poly[77] = poly[2] * poly[27];
    poly[78] = poly[3] * poly[27];
    poly[79] = poly[14] * poly[6];
    poly[80] = poly[4] * poly[25] - poly[72];
    poly[81] = poly[26] * poly[4];
    poly[82] = poly[1] * poly[28];
    poly[83] = poly[2] * poly[28];
    poly[84] = poly[3] * poly[28];
    poly[85] = poly[5] * poly[24] - poly[72];
    poly[86] = poly[19] * poly[6];
    poly[87] = poly[26] * poly[5];
    poly[88] = poly[4] * poly[28] - poly[85];
    poly[89] = poly[1] * poly[30];
    poly[90] = poly[1] * poly[31];
    poly[91] = poly[2] * poly[31];
    poly[92] = poly[1] * poly[32];
    poly[93] = poly[2] * poly[32];
    poly[94] = poly[3] * poly[32];
    poly[95] = poly[7] * poly[14];
    poly[96] = poly[1] * poly[33];
    poly[97] = poly[2] * poly[33];
    poly[98] = poly[3] * poly[33];
    poly[99] = poly[7] * poly[18];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[7] * poly[19];
    poly[101] = poly[7] * poly[20];
    poly[102] = poly[1] * poly[34];
    poly[103] = poly[2] * poly[34];
    poly[104] = poly[3] * poly[34];
    poly[105] = poly[7] * poly[24];
    poly[106] = poly[7] * poly[25];
    poly[107] = poly[7] * poly[26];
    poly[108] = poly[7] * poly[27];
    poly[109] = poly[7] * poly[28];
    poly[110] = poly[1] * poly[8];
    poly[111] = poly[1] * poly[36];
    poly[112] = poly[1] * poly[9];
    poly[113] = poly[2] * poly[10];
    poly[114] = poly[1] * poly[37];
    poly[115] = poly[2] * poly[37];
    poly[116] = poly[1] * poly[11];
    poly[117] = poly[2] * poly[12];
    poly[118] = poly[3] * poly[13];
    poly[119] = poly[1] * poly[38];
    poly[120] = poly[2] * poly[38];
    poly[121] = poly[3] * poly[38];
    poly[122] = poly[14] * poly[4];
    poly[123] = poly[1] * poly[15];
    poly[124] = poly[2] * poly[16];
    poly[125] = poly[3] * poly[17];
    poly[126] = poly[4] * poly[18] - poly[61];
    poly[127] = poly[4] * poly[20] - poly[61];
    poly[128] = poly[1] * poly[39];
    poly[129] = poly[2] * poly[39];
    poly[130] = poly[3] * poly[39];
    poly[131] = poly[5] * poly[18] - poly[62];
    poly[132] = poly[19] * poly[5];
    poly[133] = poly[4] * poly[39] - poly[131];
    poly[134] = poly[1] * poly[21];
    poly[135] = poly[2] * poly[22];
    poly[136] = poly[3] * poly[23];
    poly[137] = poly[4] * poly[24] - poly[79];
    poly[138] = poly[5] * poly[25] - poly[86];
    poly[139] = poly[4] * poly[27] - poly[79];
    poly[140] = poly[5] * poly[28] - poly[86];
    poly[141] = poly[1] * poly[40];
    poly[142] = poly[2] * poly[40];
    poly[143] = poly[3] * poly[40];
    poly[144] = poly[6] * poly[24] - poly[81];
    poly[145] = poly[6] * poly[25] - poly[87];
    poly[146] = poly[26] * poly[6];
    poly[147] = poly[4] * poly[40] - poly[144];
    poly[148] = poly[5] * poly[40] - poly[145];
    poly[149] = poly[1] * poly[29];
}

fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[2] * poly[30];
    poly[151] = poly[3] * poly[31];
    poly[152] = poly[7] * poly[38];
    poly[153] = poly[7] * poly[39];
    poly[154] = poly[7] * poly[40];
    poly[155] = poly[1] * poly[41];
    poly[156] = poly[2] * poly[41];
    poly[157] = poly[3] * poly[41];
    poly[158] = poly[7] * poly[32];
    poly[159] = poly[7] * poly[33];
    poly[160] = poly[7] * poly[34];
    poly[161] = poly[1] * poly[35];
    poly[162] = poly[2] * poly[36];
    poly[163] = poly[3] * poly[37];
    poly[164] = poly[4] * poly[38] - poly[122];
    poly[165] = poly[5] * poly[39] - poly[132];
    poly[166] = poly[6] * poly[40] - poly[146];
    poly[167] = poly[7] * poly[41];
}

// Total number of monomials = 168 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);

    return poly;
}
