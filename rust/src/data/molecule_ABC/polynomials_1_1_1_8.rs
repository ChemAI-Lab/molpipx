#![allow(unused_variables)]
use super::monomials_1_1_1_8::*;


pub const N_POLYS: usize = 165;

// File created from data/molecule_ABC/MOL_1_1_1_8.POLY 


#[inline(never)]
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

#[inline(never)]
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
    poly[84] = poly[1] * poly[56];
    poly[85] = poly[1] * poly[57];
    poly[86] = poly[1] * poly[58];
    poly[87] = poly[1] * poly[59];
    poly[88] = poly[1] * poly[72];
    poly[89] = poly[1] * poly[60];
    poly[90] = poly[1] * poly[61];
    poly[91] = poly[1] * poly[62];
    poly[92] = poly[1] * poly[74];
    poly[93] = poly[1] * poly[63];
    poly[94] = poly[1] * poly[64];
    poly[95] = poly[1] * poly[76];
    poly[96] = poly[1] * poly[65];
    poly[97] = poly[1] * poly[78];
    poly[98] = poly[1] * poly[80];
    poly[99] = poly[1] * poly[66];
}

#[inline(never)]
fn f_polynomials2(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[100] = poly[1] * poly[67];
    poly[101] = poly[1] * poly[68];
    poly[102] = poly[1] * poly[69];
    poly[103] = poly[1] * poly[70];
    poly[104] = poly[1] * poly[82];
    poly[105] = poly[1] * poly[71];
    poly[106] = poly[2] * poly[72];
    poly[107] = poly[1] * poly[73];
    poly[108] = poly[2] * poly[74];
    poly[109] = poly[1] * poly[75];
    poly[110] = poly[2] * poly[76];
    poly[111] = poly[1] * poly[77];
    poly[112] = poly[2] * poly[78];
    poly[113] = poly[1] * poly[79];
    poly[114] = poly[2] * poly[80];
    poly[115] = poly[1] * poly[83];
    poly[116] = poly[2] * poly[83];
    poly[117] = poly[1] * poly[81];
    poly[118] = poly[2] * poly[82];
    poly[119] = poly[3] * poly[83];
    poly[120] = poly[1] * poly[84];
    poly[121] = poly[1] * poly[85];
    poly[122] = poly[1] * poly[86];
    poly[123] = poly[1] * poly[87];
    poly[124] = poly[1] * poly[88];
    poly[125] = poly[1] * poly[106];
    poly[126] = poly[1] * poly[89];
    poly[127] = poly[1] * poly[90];
    poly[128] = poly[1] * poly[91];
    poly[129] = poly[1] * poly[92];
    poly[130] = poly[1] * poly[108];
    poly[131] = poly[1] * poly[93];
    poly[132] = poly[1] * poly[94];
    poly[133] = poly[1] * poly[95];
    poly[134] = poly[1] * poly[110];
    poly[135] = poly[1] * poly[96];
    poly[136] = poly[1] * poly[97];
    poly[137] = poly[1] * poly[112];
    poly[138] = poly[1] * poly[98];
    poly[139] = poly[1] * poly[114];
    poly[140] = poly[1] * poly[116];
    poly[141] = poly[1] * poly[99];
    poly[142] = poly[1] * poly[100];
    poly[143] = poly[1] * poly[101];
    poly[144] = poly[1] * poly[102];
    poly[145] = poly[1] * poly[103];
    poly[146] = poly[1] * poly[104];
    poly[147] = poly[1] * poly[118];
    poly[148] = poly[1] * poly[105];
    poly[149] = poly[2] * poly[106];
}

#[inline(never)]
fn f_polynomials3(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[150] = poly[1] * poly[107];
    poly[151] = poly[2] * poly[108];
    poly[152] = poly[1] * poly[109];
    poly[153] = poly[2] * poly[110];
    poly[154] = poly[1] * poly[111];
    poly[155] = poly[2] * poly[112];
    poly[156] = poly[1] * poly[113];
    poly[157] = poly[2] * poly[114];
    poly[158] = poly[1] * poly[115];
    poly[159] = poly[2] * poly[116];
    poly[160] = poly[1] * poly[119];
    poly[161] = poly[2] * poly[119];
    poly[162] = poly[1] * poly[117];
    poly[163] = poly[2] * poly[118];
    poly[164] = poly[3] * poly[119];
}

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);
    f_polynomials2(&mut poly, &mono);
    f_polynomials3(&mut poly, &mono);

    return poly;
}
