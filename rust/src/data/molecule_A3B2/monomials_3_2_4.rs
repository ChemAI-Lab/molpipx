#![allow(unused_variables)]



// File created from data/molecule_A3B2/MOL_3_2_4.MONO 
pub const N_MONOS: usize = 170;
// N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;
pub const N_DISTANCES: usize = 10;
pub const N_ATOMS: usize = 5;
pub const N_XYZ: usize = N_ATOMS * 3;

// Total number of monomials = 170 


pub fn f_monomials(r: & [f64; N_DISTANCES]) -> [f64; N_MONOS] {
  assert!(2 * N_DISTANCES == N_ATOMS * (N_ATOMS - 1), "library author error!");

  let mut mono = [0.0; N_MONOS];

  mono[0] = 1.0;
  mono[1] = r[9];
  mono[2] = r[8];
  mono[3] = r[7];
  mono[4] = r[6];
  mono[5] = r[5];
  mono[6] = r[3];
  mono[7] = r[2];
  mono[8] = r[4];
  mono[9] = r[1];
  mono[10] = r[0];
  mono[11] = mono[3] * mono[4];
  mono[12] = mono[2] * mono[5];
  mono[13] = mono[3] * mono[6];
  mono[14] = mono[5] * mono[6];
  mono[15] = mono[2] * mono[7];
  mono[16] = mono[4] * mono[7];
  mono[17] = mono[2] * mono[4];
  mono[18] = mono[3] * mono[5];
  mono[19] = mono[2] * mono[6];
  mono[20] = mono[4] * mono[6];
  mono[21] = mono[3] * mono[7];
  mono[22] = mono[5] * mono[7];
  mono[23] = mono[2] * mono[3];
  mono[24] = mono[4] * mono[5];
  mono[25] = mono[6] * mono[7];
  mono[26] = mono[6] * mono[8];
  mono[27] = mono[7] * mono[8];
  mono[28] = mono[4] * mono[9];
  mono[29] = mono[5] * mono[9];
  mono[30] = mono[2] * mono[10];
  mono[31] = mono[3] * mono[10];
  mono[32] = mono[8] * mono[9];
  mono[33] = mono[8] * mono[10];
  mono[34] = mono[9] * mono[10];
  mono[35] = mono[3] * mono[20];
  mono[36] = mono[2] * mono[14];
  mono[37] = mono[3] * mono[14];
  mono[38] = mono[2] * mono[16];
  mono[39] = mono[3] * mono[16];
  mono[40] = mono[2] * mono[22];
  mono[41] = mono[2] * mono[20];
  mono[42] = mono[3] * mono[22];
  mono[43] = mono[2] * mono[11];
  mono[44] = mono[2] * mono[18];
  mono[45] = mono[2] * mono[24];
  mono[46] = mono[3] * mono[24];
  mono[47] = mono[2] * mono[13];
  mono[48] = mono[4] * mono[14];
  mono[49] = mono[2] * mono[21];
  mono[50] = mono[4] * mono[22];
  mono[51] = mono[2] * mono[25];
  mono[52] = mono[3] * mono[25];
  mono[53] = mono[4] * mono[25];
  mono[54] = mono[5] * mono[25];
  mono[55] = mono[6] * mono[27];
  mono[56] = mono[4] * mono[29];
  mono[57] = mono[2] * mono[31];
  mono[58] = mono[3] * mono[26];
  mono[59] = mono[5] * mono[26];
  mono[60] = mono[2] * mono[27];
  mono[61] = mono[4] * mono[27];
  mono[62] = mono[3] * mono[28];
  mono[63] = mono[2] * mono[29];
  mono[64] = mono[6] * mono[29];
  mono[65] = mono[7] * mono[28];
  mono[66] = mono[4] * mono[31];
  mono[67] = mono[5] * mono[30];
  mono[68] = mono[6] * mono[31];
  mono[69] = mono[7] * mono[30];
  mono[70] = mono[2] * mono[26];
  mono[71] = mono[4] * mono[26];
  mono[72] = mono[3] * mono[27];
  mono[73] = mono[5] * mono[27];
  mono[74] = mono[2] * mono[28];
  mono[75] = mono[3] * mono[29];
  mono[76] = mono[6] * mono[28];
  mono[77] = mono[7] * mono[29];
  mono[78] = mono[4] * mono[30];
  mono[79] = mono[5] * mono[31];
  mono[80] = mono[6] * mono[30];
  mono[81] = mono[7] * mono[31];
  mono[82] = mono[4] * mono[32];
  mono[83] = mono[5] * mono[32];
  mono[84] = mono[6] * mono[32];
  mono[85] = mono[7] * mono[32];
  mono[86] = mono[2] * mono[33];
  mono[87] = mono[3] * mono[33];
  mono[88] = mono[6] * mono[33];
  mono[89] = mono[7] * mono[33];
  mono[90] = mono[2] * mono[34];
  mono[91] = mono[3] * mono[34];
  mono[92] = mono[4] * mono[34];
  mono[93] = mono[5] * mono[34];
  mono[94] = mono[8] * mono[34];
  mono[95] = mono[2] * mono[37];
  mono[96] = mono[3] * mono[48];
  mono[97] = mono[2] * mono[39];
  mono[98] = mono[2] * mono[50];
  mono[99] = mono[3] * mono[53];
  mono[100] = mono[2] * mono[54];
  mono[101] = mono[2] * mono[35];
  mono[102] = mono[2] * mono[48];
  mono[103] = mono[2] * mono[42];
  mono[104] = mono[3] * mono[50];
  mono[105] = mono[2] * mono[53];
  mono[106] = mono[3] * mono[54];
  mono[107] = mono[2] * mono[46];
  mono[108] = mono[2] * mono[52];
  mono[109] = mono[4] * mono[54];
  mono[110] = mono[2] * mono[55];
  mono[111] = mono[3] * mono[55];
  mono[112] = mono[4] * mono[55];
  mono[113] = mono[5] * mono[55];
  mono[114] = mono[2] * mono[56];
  mono[115] = mono[3] * mono[56];
  mono[116] = mono[4] * mono[64];
  mono[117] = mono[4] * mono[77];
  mono[118] = mono[2] * mono[66];
  mono[119] = mono[2] * mono[79];
  mono[120] = mono[2] * mono[68];
  mono[121] = mono[2] * mono[81];
  mono[122] = mono[3] * mono[71];
  mono[123] = mono[2] * mono[59];
  mono[124] = mono[3] * mono[61];
  mono[125] = mono[2] * mono[73];
  mono[126] = mono[3] * mono[76];
  mono[127] = mono[3] * mono[64];
  mono[128] = mono[2] * mono[65];
  mono[129] = mono[2] * mono[77];
  mono[130] = mono[5] * mono[80];
  mono[131] = mono[5] * mono[68];
  mono[132] = mono[4] * mono[69];
  mono[133] = mono[4] * mono[81];
  mono[134] = mono[2] * mono[58];
  mono[135] = mono[4] * mono[59];
  mono[136] = mono[2] * mono[72];
  mono[137] = mono[4] * mono[73];
  mono[138] = mono[2] * mono[62];
  mono[139] = mono[2] * mono[75];
  mono[140] = mono[6] * mono[65];
  mono[141] = mono[6] * mono[77];
  mono[142] = mono[4] * mono[67];
  mono[143] = mono[4] * mono[79];
  mono[144] = mono[6] * mono[69];
  mono[145] = mono[6] * mono[81];
  mono[146] = mono[5] * mono[84];
  mono[147] = mono[4] * mono[85];
  mono[148] = mono[3] * mono[88];
  mono[149] = mono[2] * mono[89];
  mono[150] = mono[3] * mono[92];
  mono[151] = mono[2] * mono[93];
  mono[152] = mono[4] * mono[84];
  mono[153] = mono[5] * mono[85];
  mono[154] = mono[2] * mono[88];
  mono[155] = mono[3] * mono[89];
  mono[156] = mono[2] * mono[92];
  mono[157] = mono[3] * mono[93];
  mono[158] = mono[4] * mono[83];
  mono[159] = mono[6] * mono[85];
  mono[160] = mono[2] * mono[87];
  mono[161] = mono[6] * mono[89];
  mono[162] = mono[2] * mono[91];
  mono[163] = mono[4] * mono[93];
  mono[164] = mono[3] * mono[35];
  mono[165] = mono[5] * mono[36];
  mono[166] = mono[6] * mono[37];
  mono[167] = mono[4] * mono[39];
  mono[168] = mono[2] * mono[40];
  mono[169] = mono[7] * mono[38];
  return mono;
}


