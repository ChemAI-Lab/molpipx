#![allow(unused_variables)]
use super::monomials_deg_2::*;


pub const N_POLYS: usize = 208;

// File created from ../pipx/examples/Data/DDEthanol/PIP_deg_2/MOL_1_1_1_2_3_1_2.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3];
    poly[2] = mono[4] + mono[5] + mono[6];
    poly[3] = mono[7] + mono[8];
    poly[4] = mono[9] + mono[10] + mono[11] + mono[12] + mono[13] + mono[14];
    poly[5] = mono[15];
    poly[6] = mono[16];
    poly[7] = mono[17] + mono[18] + mono[19];
    poly[8] = mono[20] + mono[21];
    poly[9] = mono[22];
    poly[10] = mono[23] + mono[24] + mono[25];
    poly[11] = mono[26] + mono[27];
    poly[12] = mono[28];
    poly[13] = mono[29];
    poly[14] = mono[30] + mono[31] + mono[32];
    poly[15] = mono[33] + mono[34];
    poly[16] = mono[35];
    poly[17] = mono[36];
    poly[18] = mono[37] + mono[38] + mono[39];
    poly[19] = mono[40] + mono[41] + mono[42];
    poly[20] = poly[1] * poly[2] - poly[19];
    poly[21] = mono[43] + mono[44] + mono[45];
    poly[22] = poly[1] * poly[3];
    poly[23] = poly[2] * poly[3];
    poly[24] = mono[46];
    poly[25] = mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54] + mono[55] + mono[56] + mono[57] + mono[58];
    poly[26] = mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64];
    poly[27] = poly[1] * poly[4] - poly[25];
    poly[28] = poly[2] * poly[4] - poly[26];
    poly[29] = mono[65] + mono[66] + mono[67] + mono[68] + mono[69] + mono[70];
    poly[30] = mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76];
    poly[31] = mono[77] + mono[78] + mono[79];
    poly[32] = poly[3] * poly[4] - poly[29];
    poly[33] = mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85];
    poly[34] = poly[5] * poly[1];
    poly[35] = poly[5] * poly[2];
    poly[36] = poly[5] * poly[3];
    poly[37] = poly[5] * poly[4];
    poly[38] = poly[6] * poly[1];
    poly[39] = poly[6] * poly[2];
    poly[40] = poly[6] * poly[3];
    poly[41] = poly[6] * poly[4];
    poly[42] = poly[5] * poly[6];
    poly[43] = mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91];
    poly[44] = mono[92] + mono[93] + mono[94];
    poly[45] = poly[1] * poly[7] - poly[43];
    poly[46] = poly[2] * poly[7] - poly[44];
    poly[47] = poly[3] * poly[7];
    poly[48] = mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100] + mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106];
    poly[49] = poly[4] * poly[7] - poly[48];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[5] * poly[7];
    poly[51] = poly[6] * poly[7];
    poly[52] = mono[107] + mono[108] + mono[109];
    poly[53] = poly[1] * poly[8];
    poly[54] = poly[2] * poly[8];
    poly[55] = mono[110] + mono[111];
    poly[56] = mono[112] + mono[113] + mono[114] + mono[115] + mono[116] + mono[117];
    poly[57] = poly[3] * poly[8] - poly[55];
    poly[58] = poly[4] * poly[8] - poly[56];
    poly[59] = poly[5] * poly[8];
    poly[60] = poly[6] * poly[8];
    poly[61] = poly[7] * poly[8];
    poly[62] = mono[118];
    poly[63] = poly[9] * poly[1];
    poly[64] = poly[9] * poly[2];
    poly[65] = poly[9] * poly[3];
    poly[66] = poly[9] * poly[4];
    poly[67] = poly[5] * poly[9];
    poly[68] = poly[6] * poly[9];
    poly[69] = poly[9] * poly[7];
    poly[70] = poly[9] * poly[8];
    poly[71] = mono[119] + mono[120] + mono[121] + mono[122] + mono[123] + mono[124];
    poly[72] = mono[125] + mono[126] + mono[127];
    poly[73] = poly[1] * poly[10] - poly[71];
    poly[74] = poly[2] * poly[10] - poly[72];
    poly[75] = poly[3] * poly[10];
    poly[76] = mono[128] + mono[129] + mono[130] + mono[131] + mono[132] + mono[133] + mono[134] + mono[135] + mono[136] + mono[137] + mono[138] + mono[139];
    poly[77] = poly[4] * poly[10] - poly[76];
    poly[78] = poly[5] * poly[10];
    poly[79] = poly[6] * poly[10];
    poly[80] = mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145];
    poly[81] = poly[7] * poly[10] - poly[80];
    poly[82] = poly[8] * poly[10];
    poly[83] = poly[9] * poly[10];
    poly[84] = mono[146] + mono[147] + mono[148];
    poly[85] = poly[1] * poly[11];
    poly[86] = poly[2] * poly[11];
    poly[87] = mono[149] + mono[150];
    poly[88] = mono[151] + mono[152] + mono[153] + mono[154] + mono[155] + mono[156];
    poly[89] = poly[3] * poly[11] - poly[87];
    poly[90] = poly[4] * poly[11] - poly[88];
    poly[91] = poly[5] * poly[11];
    poly[92] = poly[6] * poly[11];
    poly[93] = poly[7] * poly[11];
    poly[94] = mono[157] + mono[158];
    poly[95] = poly[8] * poly[11] - poly[94];
    poly[96] = poly[9] * poly[11];
    poly[97] = poly[10] * poly[11];
    poly[98] = mono[159];
    poly[99] = poly[12] * poly[1];
}

fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[12] * poly[2];
    poly[101] = poly[12] * poly[3];
    poly[102] = poly[12] * poly[4];
    poly[103] = poly[5] * poly[12];
    poly[104] = poly[6] * poly[12];
    poly[105] = poly[12] * poly[7];
    poly[106] = poly[12] * poly[8];
    poly[107] = poly[9] * poly[12];
    poly[108] = poly[12] * poly[10];
    poly[109] = poly[12] * poly[11];
    poly[110] = poly[13] * poly[1];
    poly[111] = poly[13] * poly[2];
    poly[112] = poly[13] * poly[3];
    poly[113] = poly[13] * poly[4];
    poly[114] = poly[5] * poly[13];
    poly[115] = poly[6] * poly[13];
    poly[116] = poly[13] * poly[7];
    poly[117] = poly[13] * poly[8];
    poly[118] = poly[9] * poly[13];
    poly[119] = poly[13] * poly[10];
    poly[120] = poly[13] * poly[11];
    poly[121] = poly[12] * poly[13];
    poly[122] = mono[160] + mono[161] + mono[162] + mono[163] + mono[164] + mono[165];
    poly[123] = mono[166] + mono[167] + mono[168];
    poly[124] = poly[1] * poly[14] - poly[122];
    poly[125] = poly[2] * poly[14] - poly[123];
    poly[126] = poly[3] * poly[14];
    poly[127] = mono[169] + mono[170] + mono[171] + mono[172] + mono[173] + mono[174] + mono[175] + mono[176] + mono[177] + mono[178] + mono[179] + mono[180];
    poly[128] = poly[4] * poly[14] - poly[127];
    poly[129] = poly[5] * poly[14];
    poly[130] = poly[6] * poly[14];
    poly[131] = mono[181] + mono[182] + mono[183] + mono[184] + mono[185] + mono[186];
    poly[132] = poly[7] * poly[14] - poly[131];
    poly[133] = poly[8] * poly[14];
    poly[134] = poly[9] * poly[14];
    poly[135] = mono[187] + mono[188] + mono[189] + mono[190] + mono[191] + mono[192];
    poly[136] = poly[10] * poly[14] - poly[135];
    poly[137] = poly[11] * poly[14];
    poly[138] = poly[12] * poly[14];
    poly[139] = poly[13] * poly[14];
    poly[140] = mono[193] + mono[194] + mono[195];
    poly[141] = poly[1] * poly[15];
    poly[142] = poly[2] * poly[15];
    poly[143] = mono[196] + mono[197];
    poly[144] = mono[198] + mono[199] + mono[200] + mono[201] + mono[202] + mono[203];
    poly[145] = poly[3] * poly[15] - poly[143];
    poly[146] = poly[4] * poly[15] - poly[144];
    poly[147] = poly[5] * poly[15];
    poly[148] = poly[6] * poly[15];
    poly[149] = poly[7] * poly[15];
}

fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = mono[204] + mono[205];
    poly[151] = poly[8] * poly[15] - poly[150];
    poly[152] = poly[9] * poly[15];
    poly[153] = poly[10] * poly[15];
    poly[154] = mono[206] + mono[207];
    poly[155] = poly[11] * poly[15] - poly[154];
    poly[156] = poly[12] * poly[15];
    poly[157] = poly[13] * poly[15];
    poly[158] = poly[14] * poly[15];
    poly[159] = mono[208];
    poly[160] = poly[16] * poly[1];
    poly[161] = poly[16] * poly[2];
    poly[162] = poly[16] * poly[3];
    poly[163] = poly[16] * poly[4];
    poly[164] = poly[5] * poly[16];
    poly[165] = poly[6] * poly[16];
    poly[166] = poly[16] * poly[7];
    poly[167] = poly[16] * poly[8];
    poly[168] = poly[9] * poly[16];
    poly[169] = poly[16] * poly[10];
    poly[170] = poly[16] * poly[11];
    poly[171] = poly[12] * poly[16];
    poly[172] = poly[13] * poly[16];
    poly[173] = poly[16] * poly[14];
    poly[174] = poly[16] * poly[15];
    poly[175] = poly[17] * poly[1];
    poly[176] = poly[17] * poly[2];
    poly[177] = poly[17] * poly[3];
    poly[178] = poly[17] * poly[4];
    poly[179] = poly[5] * poly[17];
    poly[180] = poly[6] * poly[17];
    poly[181] = poly[17] * poly[7];
    poly[182] = poly[17] * poly[8];
    poly[183] = poly[9] * poly[17];
    poly[184] = poly[17] * poly[10];
    poly[185] = poly[17] * poly[11];
    poly[186] = poly[12] * poly[17];
    poly[187] = poly[13] * poly[17];
    poly[188] = poly[17] * poly[14];
    poly[189] = poly[17] * poly[15];
    poly[190] = poly[16] * poly[17];
    poly[191] = poly[1] * poly[1] - poly[18] - poly[18];
    poly[192] = poly[2] * poly[2] - poly[21] - poly[21];
    poly[193] = poly[3] * poly[3] - poly[24] - poly[24];
    poly[194] = poly[4] * poly[4] - poly[33] - poly[31] - poly[30] - poly[33] - poly[31] - poly[30];
    poly[195] = poly[5] * poly[5];
    poly[196] = poly[6] * poly[6];
    poly[197] = poly[7] * poly[7] - poly[52] - poly[52];
    poly[198] = poly[8] * poly[8] - poly[62] - poly[62];
    poly[199] = poly[9] * poly[9];
}

fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[10] * poly[10] - poly[84] - poly[84];
    poly[201] = poly[11] * poly[11] - poly[98] - poly[98];
    poly[202] = poly[12] * poly[12];
    poly[203] = poly[13] * poly[13];
    poly[204] = poly[14] * poly[14] - poly[140] - poly[140];
    poly[205] = poly[15] * poly[15] - poly[159] - poly[159];
    poly[206] = poly[16] * poly[16];
    poly[207] = poly[17] * poly[17];
}

// Total number of monomials = 208 

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
