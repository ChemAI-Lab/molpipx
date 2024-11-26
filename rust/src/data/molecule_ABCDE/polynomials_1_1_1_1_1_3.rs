#![allow(unused_variables)]
use super::monomials_1_1_1_1_1_3::*;


pub const N_POLYS: usize = 286;

// File created from data/molecule_ABCDE/MOL_1_1_1_1_1_3.POLY 


#[inline(never)]
fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1];
    poly[2] = mono[2];
    poly[3] = mono[3];
    poly[4] = mono[4];
    poly[5] = mono[5];
    poly[6] = mono[6];
    poly[7] = mono[7];
    poly[8] = mono[8];
    poly[9] = mono[9];
    poly[10] = mono[10];
    poly[11] = poly[1] * poly[2];
    poly[12] = poly[1] * poly[3];
    poly[13] = poly[2] * poly[3];
    poly[14] = poly[1] * poly[4];
    poly[15] = poly[2] * poly[4];
    poly[16] = poly[3] * poly[4];
    poly[17] = poly[1] * poly[5];
    poly[18] = poly[2] * poly[5];
    poly[19] = poly[3] * poly[5];
    poly[20] = poly[4] * poly[5];
    poly[21] = poly[1] * poly[6];
    poly[22] = poly[2] * poly[6];
    poly[23] = poly[3] * poly[6];
    poly[24] = poly[4] * poly[6];
    poly[25] = poly[5] * poly[6];
    poly[26] = poly[1] * poly[7];
    poly[27] = poly[2] * poly[7];
    poly[28] = poly[3] * poly[7];
    poly[29] = poly[4] * poly[7];
    poly[30] = poly[5] * poly[7];
    poly[31] = poly[6] * poly[7];
    poly[32] = poly[1] * poly[8];
    poly[33] = poly[2] * poly[8];
    poly[34] = poly[3] * poly[8];
    poly[35] = poly[4] * poly[8];
    poly[36] = poly[5] * poly[8];
    poly[37] = poly[6] * poly[8];
    poly[38] = poly[7] * poly[8];
    poly[39] = poly[1] * poly[9];
    poly[40] = poly[2] * poly[9];
    poly[41] = poly[3] * poly[9];
    poly[42] = poly[4] * poly[9];
    poly[43] = poly[5] * poly[9];
    poly[44] = poly[6] * poly[9];
    poly[45] = poly[7] * poly[9];
    poly[46] = poly[8] * poly[9];
    poly[47] = poly[1] * poly[10];
    poly[48] = poly[2] * poly[10];
    poly[49] = poly[3] * poly[10];
}

#[inline(never)]
fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[4] * poly[10];
    poly[51] = poly[5] * poly[10];
    poly[52] = poly[6] * poly[10];
    poly[53] = poly[7] * poly[10];
    poly[54] = poly[8] * poly[10];
    poly[55] = poly[9] * poly[10];
    poly[56] = poly[1] * poly[1];
    poly[57] = poly[2] * poly[2];
    poly[58] = poly[3] * poly[3];
    poly[59] = poly[4] * poly[4];
    poly[60] = poly[5] * poly[5];
    poly[61] = poly[6] * poly[6];
    poly[62] = poly[7] * poly[7];
    poly[63] = poly[8] * poly[8];
    poly[64] = poly[9] * poly[9];
    poly[65] = poly[10] * poly[10];
    poly[66] = poly[1] * poly[13];
    poly[67] = poly[1] * poly[15];
    poly[68] = poly[1] * poly[16];
    poly[69] = poly[2] * poly[16];
    poly[70] = poly[1] * poly[18];
    poly[71] = poly[1] * poly[19];
    poly[72] = poly[2] * poly[19];
    poly[73] = poly[1] * poly[20];
    poly[74] = poly[2] * poly[20];
    poly[75] = poly[3] * poly[20];
    poly[76] = poly[1] * poly[22];
    poly[77] = poly[1] * poly[23];
    poly[78] = poly[2] * poly[23];
    poly[79] = poly[1] * poly[24];
    poly[80] = poly[2] * poly[24];
    poly[81] = poly[3] * poly[24];
    poly[82] = poly[1] * poly[25];
    poly[83] = poly[2] * poly[25];
    poly[84] = poly[3] * poly[25];
    poly[85] = poly[4] * poly[25];
    poly[86] = poly[1] * poly[27];
    poly[87] = poly[1] * poly[28];
    poly[88] = poly[2] * poly[28];
    poly[89] = poly[1] * poly[29];
    poly[90] = poly[2] * poly[29];
    poly[91] = poly[3] * poly[29];
    poly[92] = poly[1] * poly[30];
    poly[93] = poly[2] * poly[30];
    poly[94] = poly[3] * poly[30];
    poly[95] = poly[4] * poly[30];
    poly[96] = poly[1] * poly[31];
    poly[97] = poly[2] * poly[31];
    poly[98] = poly[3] * poly[31];
    poly[99] = poly[4] * poly[31];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[5] * poly[31];
    poly[101] = poly[1] * poly[33];
    poly[102] = poly[1] * poly[34];
    poly[103] = poly[2] * poly[34];
    poly[104] = poly[1] * poly[35];
    poly[105] = poly[2] * poly[35];
    poly[106] = poly[3] * poly[35];
    poly[107] = poly[1] * poly[36];
    poly[108] = poly[2] * poly[36];
    poly[109] = poly[3] * poly[36];
    poly[110] = poly[4] * poly[36];
    poly[111] = poly[1] * poly[37];
    poly[112] = poly[2] * poly[37];
    poly[113] = poly[3] * poly[37];
    poly[114] = poly[4] * poly[37];
    poly[115] = poly[5] * poly[37];
    poly[116] = poly[1] * poly[38];
    poly[117] = poly[2] * poly[38];
    poly[118] = poly[3] * poly[38];
    poly[119] = poly[4] * poly[38];
    poly[120] = poly[5] * poly[38];
    poly[121] = poly[6] * poly[38];
    poly[122] = poly[1] * poly[40];
    poly[123] = poly[1] * poly[41];
    poly[124] = poly[2] * poly[41];
    poly[125] = poly[1] * poly[42];
    poly[126] = poly[2] * poly[42];
    poly[127] = poly[3] * poly[42];
    poly[128] = poly[1] * poly[43];
    poly[129] = poly[2] * poly[43];
    poly[130] = poly[3] * poly[43];
    poly[131] = poly[4] * poly[43];
    poly[132] = poly[1] * poly[44];
    poly[133] = poly[2] * poly[44];
    poly[134] = poly[3] * poly[44];
    poly[135] = poly[4] * poly[44];
    poly[136] = poly[5] * poly[44];
    poly[137] = poly[1] * poly[45];
    poly[138] = poly[2] * poly[45];
    poly[139] = poly[3] * poly[45];
    poly[140] = poly[4] * poly[45];
    poly[141] = poly[5] * poly[45];
    poly[142] = poly[6] * poly[45];
    poly[143] = poly[1] * poly[46];
    poly[144] = poly[2] * poly[46];
    poly[145] = poly[3] * poly[46];
    poly[146] = poly[4] * poly[46];
    poly[147] = poly[5] * poly[46];
    poly[148] = poly[6] * poly[46];
    poly[149] = poly[7] * poly[46];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[48];
    poly[151] = poly[1] * poly[49];
    poly[152] = poly[2] * poly[49];
    poly[153] = poly[1] * poly[50];
    poly[154] = poly[2] * poly[50];
    poly[155] = poly[3] * poly[50];
    poly[156] = poly[1] * poly[51];
    poly[157] = poly[2] * poly[51];
    poly[158] = poly[3] * poly[51];
    poly[159] = poly[4] * poly[51];
    poly[160] = poly[1] * poly[52];
    poly[161] = poly[2] * poly[52];
    poly[162] = poly[3] * poly[52];
    poly[163] = poly[4] * poly[52];
    poly[164] = poly[5] * poly[52];
    poly[165] = poly[1] * poly[53];
    poly[166] = poly[2] * poly[53];
    poly[167] = poly[3] * poly[53];
    poly[168] = poly[4] * poly[53];
    poly[169] = poly[5] * poly[53];
    poly[170] = poly[6] * poly[53];
    poly[171] = poly[1] * poly[54];
    poly[172] = poly[2] * poly[54];
    poly[173] = poly[3] * poly[54];
    poly[174] = poly[4] * poly[54];
    poly[175] = poly[5] * poly[54];
    poly[176] = poly[6] * poly[54];
    poly[177] = poly[7] * poly[54];
    poly[178] = poly[1] * poly[55];
    poly[179] = poly[2] * poly[55];
    poly[180] = poly[3] * poly[55];
    poly[181] = poly[4] * poly[55];
    poly[182] = poly[5] * poly[55];
    poly[183] = poly[6] * poly[55];
    poly[184] = poly[7] * poly[55];
    poly[185] = poly[8] * poly[55];
    poly[186] = poly[1] * poly[11];
    poly[187] = poly[1] * poly[57];
    poly[188] = poly[1] * poly[12];
    poly[189] = poly[2] * poly[13];
    poly[190] = poly[1] * poly[58];
    poly[191] = poly[2] * poly[58];
    poly[192] = poly[1] * poly[14];
    poly[193] = poly[2] * poly[15];
    poly[194] = poly[3] * poly[16];
    poly[195] = poly[1] * poly[59];
    poly[196] = poly[2] * poly[59];
    poly[197] = poly[3] * poly[59];
    poly[198] = poly[1] * poly[17];
    poly[199] = poly[2] * poly[18];
}

#[inline(never)]
fn f_polynomials4(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[200] = poly[3] * poly[19];
    poly[201] = poly[4] * poly[20];
    poly[202] = poly[1] * poly[60];
    poly[203] = poly[2] * poly[60];
    poly[204] = poly[3] * poly[60];
    poly[205] = poly[4] * poly[60];
    poly[206] = poly[1] * poly[21];
    poly[207] = poly[2] * poly[22];
    poly[208] = poly[3] * poly[23];
    poly[209] = poly[4] * poly[24];
    poly[210] = poly[5] * poly[25];
    poly[211] = poly[1] * poly[61];
    poly[212] = poly[2] * poly[61];
    poly[213] = poly[3] * poly[61];
    poly[214] = poly[4] * poly[61];
    poly[215] = poly[5] * poly[61];
    poly[216] = poly[1] * poly[26];
    poly[217] = poly[2] * poly[27];
    poly[218] = poly[3] * poly[28];
    poly[219] = poly[4] * poly[29];
    poly[220] = poly[5] * poly[30];
    poly[221] = poly[6] * poly[31];
    poly[222] = poly[1] * poly[62];
    poly[223] = poly[2] * poly[62];
    poly[224] = poly[3] * poly[62];
    poly[225] = poly[4] * poly[62];
    poly[226] = poly[5] * poly[62];
    poly[227] = poly[6] * poly[62];
    poly[228] = poly[1] * poly[32];
    poly[229] = poly[2] * poly[33];
    poly[230] = poly[3] * poly[34];
    poly[231] = poly[4] * poly[35];
    poly[232] = poly[5] * poly[36];
    poly[233] = poly[6] * poly[37];
    poly[234] = poly[7] * poly[38];
    poly[235] = poly[1] * poly[63];
    poly[236] = poly[2] * poly[63];
    poly[237] = poly[3] * poly[63];
    poly[238] = poly[4] * poly[63];
    poly[239] = poly[5] * poly[63];
    poly[240] = poly[6] * poly[63];
    poly[241] = poly[7] * poly[63];
    poly[242] = poly[1] * poly[39];
    poly[243] = poly[2] * poly[40];
    poly[244] = poly[3] * poly[41];
    poly[245] = poly[4] * poly[42];
    poly[246] = poly[5] * poly[43];
    poly[247] = poly[6] * poly[44];
    poly[248] = poly[7] * poly[45];
    poly[249] = poly[8] * poly[46];
}

#[inline(never)]
fn f_polynomials5(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[250] = poly[1] * poly[64];
    poly[251] = poly[2] * poly[64];
    poly[252] = poly[3] * poly[64];
    poly[253] = poly[4] * poly[64];
    poly[254] = poly[5] * poly[64];
    poly[255] = poly[6] * poly[64];
    poly[256] = poly[7] * poly[64];
    poly[257] = poly[8] * poly[64];
    poly[258] = poly[1] * poly[47];
    poly[259] = poly[2] * poly[48];
    poly[260] = poly[3] * poly[49];
    poly[261] = poly[4] * poly[50];
    poly[262] = poly[5] * poly[51];
    poly[263] = poly[6] * poly[52];
    poly[264] = poly[7] * poly[53];
    poly[265] = poly[8] * poly[54];
    poly[266] = poly[9] * poly[55];
    poly[267] = poly[1] * poly[65];
    poly[268] = poly[2] * poly[65];
    poly[269] = poly[3] * poly[65];
    poly[270] = poly[4] * poly[65];
    poly[271] = poly[5] * poly[65];
    poly[272] = poly[6] * poly[65];
    poly[273] = poly[7] * poly[65];
    poly[274] = poly[8] * poly[65];
    poly[275] = poly[9] * poly[65];
    poly[276] = poly[1] * poly[56];
    poly[277] = poly[2] * poly[57];
    poly[278] = poly[3] * poly[58];
    poly[279] = poly[4] * poly[59];
    poly[280] = poly[5] * poly[60];
    poly[281] = poly[6] * poly[61];
    poly[282] = poly[7] * poly[62];
    poly[283] = poly[8] * poly[63];
    poly[284] = poly[9] * poly[64];
    poly[285] = poly[10] * poly[65];
}

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
