#![allow(unused_variables)]
use super::monomials_2_1_1_5::*;


pub const N_POLYS: usize = 256;

// File created from data/molecule_A2BC/MOL_2_1_1_5.POLY 


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
    poly[120] = poly[1] * poly[52];
    poly[121] = poly[1] * poly[57];
    poly[122] = poly[1] * poly[58];
    poly[123] = poly[4] * poly[52];
    poly[124] = poly[1] * poly[50];
    poly[125] = poly[1] * poly[65];
    poly[126] = poly[1] * poly[51];
    poly[127] = poly[1] * poly[67];
    poly[128] = poly[1] * poly[68];
    poly[129] = poly[6] * poly[23];
    poly[130] = poly[1] * poly[71];
    poly[131] = poly[1] * poly[73];
    poly[132] = poly[1] * poly[74];
    poly[133] = poly[6] * poly[38];
    poly[134] = poly[1] * poly[53];
    poly[135] = poly[1] * poly[77];
    poly[136] = poly[1] * poly[54];
    poly[137] = poly[1] * poly[79];
    poly[138] = poly[1] * poly[55];
    poly[139] = poly[1] * poly[56];
    poly[140] = poly[4] * poly[65];
    poly[141] = poly[1] * poly[80];
    poly[142] = poly[4] * poly[67];
    poly[143] = poly[4] * poly[68];
    poly[144] = poly[1] * poly[82];
    poly[145] = poly[1] * poly[83];
    poly[146] = poly[4] * poly[71];
    poly[147] = poly[1] * poly[84];
    poly[148] = poly[4] * poly[73];
    poly[149] = poly[4] * poly[74];
}

fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[86];
    poly[151] = poly[1] * poly[88];
    poly[152] = poly[1] * poly[89];
    poly[153] = poly[1] * poly[90];
    poly[154] = poly[4] * poly[57];
    poly[155] = poly[4] * poly[58];
    poly[156] = poly[1] * poly[59];
    poly[157] = poly[1] * poly[60];
    poly[158] = poly[1] * poly[93];
    poly[159] = poly[1] * poly[95];
    poly[160] = poly[1] * poly[61];
    poly[161] = poly[1] * poly[62];
    poly[162] = poly[1] * poly[97];
    poly[163] = poly[1] * poly[63];
    poly[164] = poly[1] * poly[64];
    poly[165] = poly[6] * poly[34];
    poly[166] = poly[1] * poly[66];
    poly[167] = poly[6] * poly[22];
    poly[168] = poly[1] * poly[98];
    poly[169] = poly[6] * poly[35];
    poly[170] = poly[9] * poly[47];
    poly[171] = poly[1] * poly[69];
    poly[172] = poly[1] * poly[100];
    poly[173] = poly[1] * poly[70];
    poly[174] = poly[9] * poly[34];
    poly[175] = poly[1] * poly[101];
    poly[176] = poly[1] * poly[72];
    poly[177] = poly[6] * poly[37];
    poly[178] = poly[9] * poly[23];
    poly[179] = poly[1] * poly[102];
    poly[180] = poly[6] * poly[39];
    poly[181] = poly[9] * poly[35];
    poly[182] = poly[1] * poly[104];
    poly[183] = poly[1] * poly[105];
    poly[184] = poly[9] * poly[37];
    poly[185] = poly[1] * poly[106];
    poly[186] = poly[6] * poly[48];
    poly[187] = poly[9] * poly[39];
    poly[188] = poly[1] * poly[75];
    poly[189] = poly[1] * poly[76];
    poly[190] = poly[4] * poly[93];
    poly[191] = poly[1] * poly[108];
    poly[192] = poly[4] * poly[95];
    poly[193] = poly[1] * poly[78];
    poly[194] = poly[4] * poly[97];
    poly[195] = poly[4] * poly[98];
    poly[196] = poly[1] * poly[81];
    poly[197] = poly[4] * poly[100];
    poly[198] = poly[4] * poly[101];
    poly[199] = poly[4] * poly[102];
}

fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[1] * poly[109];
    poly[201] = poly[4] * poly[104];
    poly[202] = poly[4] * poly[105];
    poly[203] = poly[4] * poly[106];
    poly[204] = poly[1] * poly[85];
    poly[205] = poly[1] * poly[111];
    poly[206] = poly[4] * poly[77];
    poly[207] = poly[1] * poly[87];
    poly[208] = poly[4] * poly[79];
    poly[209] = poly[4] * poly[80];
    poly[210] = poly[1] * poly[112];
    poly[211] = poly[4] * poly[82];
    poly[212] = poly[4] * poly[83];
    poly[213] = poly[4] * poly[84];
    poly[214] = poly[1] * poly[114];
    poly[215] = poly[4] * poly[86];
    poly[216] = poly[1] * poly[115];
    poly[217] = poly[4] * poly[88];
    poly[218] = poly[4] * poly[89];
    poly[219] = poly[4] * poly[90];
    poly[220] = poly[1] * poly[91];
    poly[221] = poly[1] * poly[92];
    poly[222] = poly[1] * poly[94];
    poly[223] = poly[6] * poly[32];
    poly[224] = poly[1] * poly[117];
    poly[225] = poly[6] * poly[47];
    poly[226] = poly[1] * poly[96];
    poly[227] = poly[2] * poly[97] - poly[165];
    poly[228] = poly[2] * poly[98] - poly[169];
    poly[229] = poly[1] * poly[99];
    poly[230] = poly[2] * poly[100] - poly[177];
    poly[231] = poly[2] * poly[102] - poly[180];
    poly[232] = poly[1] * poly[103];
    poly[233] = poly[2] * poly[104] - poly[186];
    poly[234] = poly[9] * poly[38];
    poly[235] = poly[2] * poly[106] - poly[186];
    poly[236] = poly[1] * poly[118];
    poly[237] = poly[3] * poly[104] - poly[184];
    poly[238] = poly[9] * poly[48];
    poly[239] = poly[2] * poly[118] - poly[237];
    poly[240] = poly[1] * poly[107];
    poly[241] = poly[4] * poly[117];
    poly[242] = poly[4] * poly[118];
    poly[243] = poly[1] * poly[110];
    poly[244] = poly[4] * poly[108];
    poly[245] = poly[4] * poly[109];
    poly[246] = poly[1] * poly[113];
    poly[247] = poly[4] * poly[111];
    poly[248] = poly[4] * poly[112];
    poly[249] = poly[1] * poly[119];
}

fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[4] * poly[114];
    poly[251] = poly[4] * poly[115];
    poly[252] = poly[1] * poly[116];
    poly[253] = poly[2] * poly[117] - poly[225];
    poly[254] = poly[3] * poly[118] - poly[238];
    poly[255] = poly[4] * poly[119];
}

// Total number of monomials = 256 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);
    f_polynomials4(&mut poly, &mono);
    f_polynomials5(&mut poly, &mono);

    return poly;
}
