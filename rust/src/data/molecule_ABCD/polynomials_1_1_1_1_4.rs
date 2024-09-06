#![allow(unused_variables)]
use super::monomials_1_1_1_1_4::*;


pub const N_POLYS: usize = 210;

// File created from data/molecule_ABCD/MOL_1_1_1_1_4.POLY 


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
    poly[84] = poly[1] * poly[31];
    poly[85] = poly[1] * poly[34];
    poly[86] = poly[1] * poly[36];
    poly[87] = poly[1] * poly[37];
    poly[88] = poly[2] * poly[37];
    poly[89] = poly[1] * poly[40];
    poly[90] = poly[1] * poly[42];
    poly[91] = poly[1] * poly[43];
    poly[92] = poly[2] * poly[43];
    poly[93] = poly[1] * poly[45];
    poly[94] = poly[1] * poly[46];
    poly[95] = poly[2] * poly[46];
    poly[96] = poly[1] * poly[47];
    poly[97] = poly[2] * poly[47];
    poly[98] = poly[3] * poly[47];
    poly[99] = poly[1] * poly[28];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[51];
    poly[101] = poly[1] * poly[53];
    poly[102] = poly[1] * poly[29];
    poly[103] = poly[1] * poly[55];
    poly[104] = poly[1] * poly[30];
    poly[105] = poly[2] * poly[31];
    poly[106] = poly[1] * poly[56];
    poly[107] = poly[2] * poly[56];
    poly[108] = poly[1] * poly[58];
    poly[109] = poly[1] * poly[59];
    poly[110] = poly[2] * poly[59];
    poly[111] = poly[1] * poly[32];
    poly[112] = poly[1] * poly[61];
    poly[113] = poly[1] * poly[33];
    poly[114] = poly[2] * poly[34];
    poly[115] = poly[1] * poly[62];
    poly[116] = poly[2] * poly[62];
    poly[117] = poly[1] * poly[35];
    poly[118] = poly[2] * poly[36];
    poly[119] = poly[3] * poly[37];
    poly[120] = poly[1] * poly[63];
    poly[121] = poly[2] * poly[63];
    poly[122] = poly[3] * poly[63];
    poly[123] = poly[1] * poly[65];
    poly[124] = poly[1] * poly[66];
    poly[125] = poly[2] * poly[66];
    poly[126] = poly[1] * poly[67];
    poly[127] = poly[2] * poly[67];
    poly[128] = poly[3] * poly[67];
    poly[129] = poly[1] * poly[38];
    poly[130] = poly[1] * poly[69];
    poly[131] = poly[1] * poly[39];
    poly[132] = poly[2] * poly[40];
    poly[133] = poly[1] * poly[70];
    poly[134] = poly[2] * poly[70];
    poly[135] = poly[1] * poly[41];
    poly[136] = poly[2] * poly[42];
    poly[137] = poly[3] * poly[43];
    poly[138] = poly[1] * poly[71];
    poly[139] = poly[2] * poly[71];
    poly[140] = poly[3] * poly[71];
    poly[141] = poly[1] * poly[44];
    poly[142] = poly[2] * poly[45];
    poly[143] = poly[3] * poly[46];
    poly[144] = poly[4] * poly[47];
    poly[145] = poly[1] * poly[72];
    poly[146] = poly[2] * poly[72];
    poly[147] = poly[3] * poly[72];
    poly[148] = poly[4] * poly[72];
    poly[149] = poly[1] * poly[74];
}

fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[75];
    poly[151] = poly[2] * poly[75];
    poly[152] = poly[1] * poly[76];
    poly[153] = poly[2] * poly[76];
    poly[154] = poly[3] * poly[76];
    poly[155] = poly[1] * poly[77];
    poly[156] = poly[2] * poly[77];
    poly[157] = poly[3] * poly[77];
    poly[158] = poly[4] * poly[77];
    poly[159] = poly[1] * poly[48];
    poly[160] = poly[1] * poly[49];
    poly[161] = poly[1] * poly[79];
    poly[162] = poly[1] * poly[50];
    poly[163] = poly[2] * poly[51];
    poly[164] = poly[1] * poly[52];
    poly[165] = poly[2] * poly[53];
    poly[166] = poly[1] * poly[80];
    poly[167] = poly[2] * poly[80];
    poly[168] = poly[1] * poly[54];
    poly[169] = poly[2] * poly[55];
    poly[170] = poly[3] * poly[56];
    poly[171] = poly[1] * poly[57];
    poly[172] = poly[2] * poly[58];
    poly[173] = poly[3] * poly[59];
    poly[174] = poly[1] * poly[81];
    poly[175] = poly[2] * poly[81];
    poly[176] = poly[3] * poly[81];
    poly[177] = poly[1] * poly[60];
    poly[178] = poly[2] * poly[61];
    poly[179] = poly[3] * poly[62];
    poly[180] = poly[4] * poly[63];
    poly[181] = poly[1] * poly[64];
    poly[182] = poly[2] * poly[65];
    poly[183] = poly[3] * poly[66];
    poly[184] = poly[4] * poly[67];
    poly[185] = poly[1] * poly[82];
    poly[186] = poly[2] * poly[82];
    poly[187] = poly[3] * poly[82];
    poly[188] = poly[4] * poly[82];
    poly[189] = poly[1] * poly[68];
    poly[190] = poly[2] * poly[69];
    poly[191] = poly[3] * poly[70];
    poly[192] = poly[4] * poly[71];
    poly[193] = poly[5] * poly[72];
    poly[194] = poly[1] * poly[73];
    poly[195] = poly[2] * poly[74];
    poly[196] = poly[3] * poly[75];
    poly[197] = poly[4] * poly[76];
    poly[198] = poly[5] * poly[77];
    poly[199] = poly[1] * poly[83];
}

fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[2] * poly[83];
    poly[201] = poly[3] * poly[83];
    poly[202] = poly[4] * poly[83];
    poly[203] = poly[5] * poly[83];
    poly[204] = poly[1] * poly[78];
    poly[205] = poly[2] * poly[79];
    poly[206] = poly[3] * poly[80];
    poly[207] = poly[4] * poly[81];
    poly[208] = poly[5] * poly[82];
    poly[209] = poly[6] * poly[83];
}

// Total number of monomials = 210 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);
    f_polynomials4(&mut poly, &mono);

    return poly;
}
