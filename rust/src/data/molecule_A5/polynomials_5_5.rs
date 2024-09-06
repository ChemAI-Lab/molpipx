#![allow(unused_variables)]
use super::monomials_5_5::*;


pub const N_POLYS: usize = 64;

// File created from data/molecule_A5/MOL_5_5.POLY 


fn f_polynomials0(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3] + mono[4] + mono[5] + mono[6] + mono[7] + mono[8] + mono[9] + mono[10];
    poly[2] = mono[11] + mono[12] + mono[13] + mono[14] + mono[15] + mono[16] + mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22] + mono[23] + mono[24] + mono[25];
    poly[3] = mono[26] + mono[27] + mono[28] + mono[29] + mono[30] + mono[31] + mono[32] + mono[33] + mono[34] + mono[35] + mono[36] + mono[37] + mono[38] + mono[39] + mono[40] + mono[41] + mono[42] + mono[43] + mono[44] + mono[45] + mono[46] + mono[47] + mono[48] + mono[49] + mono[50] + mono[51] + mono[52] + mono[53] + mono[54] + mono[55];
    poly[4] = poly[1] * poly[1] - poly[3] - poly[2] - poly[3] - poly[2];
    poly[5] = mono[56] + mono[57] + mono[58] + mono[59] + mono[60] + mono[61] + mono[62] + mono[63] + mono[64] + mono[65] + mono[66] + mono[67] + mono[68] + mono[69] + mono[70] + mono[71] + mono[72] + mono[73] + mono[74] + mono[75] + mono[76] + mono[77] + mono[78] + mono[79] + mono[80] + mono[81] + mono[82] + mono[83] + mono[84] + mono[85];
    poly[6] = mono[86] + mono[87] + mono[88] + mono[89] + mono[90] + mono[91] + mono[92] + mono[93] + mono[94] + mono[95] + mono[96] + mono[97] + mono[98] + mono[99] + mono[100] + mono[101] + mono[102] + mono[103] + mono[104] + mono[105] + mono[106] + mono[107] + mono[108] + mono[109] + mono[110] + mono[111] + mono[112] + mono[113] + mono[114] + mono[115] + mono[116] + mono[117] + mono[118] + mono[119] + mono[120] + mono[121] + mono[122] + mono[123] + mono[124] + mono[125] + mono[126] + mono[127] + mono[128] + mono[129] + mono[130] + mono[131] + mono[132] + mono[133] + mono[134] + mono[135] + mono[136] + mono[137] + mono[138] + mono[139] + mono[140] + mono[141] + mono[142] + mono[143] + mono[144] + mono[145];
    poly[7] = mono[146] + mono[147] + mono[148] + mono[149] + mono[150] + mono[151] + mono[152] + mono[153] + mono[154] + mono[155];
    poly[8] = mono[156] + mono[157] + mono[158] + mono[159] + mono[160] + mono[161] + mono[162] + mono[163] + mono[164] + mono[165] + mono[166] + mono[167] + mono[168] + mono[169] + mono[170] + mono[171] + mono[172] + mono[173] + mono[174] + mono[175];
    poly[9] = poly[1] * poly[2] - poly[6] - poly[5] - poly[5];
    poly[10] = poly[1] * poly[3] - poly[7] - poly[8] - poly[6] - poly[5] - poly[7] - poly[8] - poly[6] - poly[7] - poly[8];
    poly[11] = poly[1] * poly[4] - poly[10] - poly[9];
    poly[12] = mono[176] + mono[177] + mono[178] + mono[179] + mono[180] + mono[181] + mono[182] + mono[183] + mono[184] + mono[185] + mono[186] + mono[187] + mono[188] + mono[189] + mono[190] + mono[191] + mono[192] + mono[193] + mono[194] + mono[195] + mono[196] + mono[197] + mono[198] + mono[199] + mono[200] + mono[201] + mono[202] + mono[203] + mono[204] + mono[205] + mono[206] + mono[207] + mono[208] + mono[209] + mono[210] + mono[211] + mono[212] + mono[213] + mono[214] + mono[215] + mono[216] + mono[217] + mono[218] + mono[219] + mono[220] + mono[221] + mono[222] + mono[223] + mono[224] + mono[225] + mono[226] + mono[227] + mono[228] + mono[229] + mono[230] + mono[231] + mono[232] + mono[233] + mono[234] + mono[235];
    poly[13] = mono[236] + mono[237] + mono[238] + mono[239] + mono[240] + mono[241] + mono[242] + mono[243] + mono[244] + mono[245] + mono[246] + mono[247] + mono[248] + mono[249] + mono[250];
    poly[14] = mono[251] + mono[252] + mono[253] + mono[254] + mono[255] + mono[256] + mono[257] + mono[258] + mono[259] + mono[260];
    poly[15] = mono[261] + mono[262] + mono[263] + mono[264] + mono[265] + mono[266] + mono[267] + mono[268] + mono[269] + mono[270] + mono[271] + mono[272] + mono[273] + mono[274] + mono[275] + mono[276] + mono[277] + mono[278] + mono[279] + mono[280] + mono[281] + mono[282] + mono[283] + mono[284] + mono[285] + mono[286] + mono[287] + mono[288] + mono[289] + mono[290] + mono[291] + mono[292] + mono[293] + mono[294] + mono[295] + mono[296] + mono[297] + mono[298] + mono[299] + mono[300] + mono[301] + mono[302] + mono[303] + mono[304] + mono[305] + mono[306] + mono[307] + mono[308] + mono[309] + mono[310] + mono[311] + mono[312] + mono[313] + mono[314] + mono[315] + mono[316] + mono[317] + mono[318] + mono[319] + mono[320];
    poly[16] = mono[321] + mono[322] + mono[323] + mono[324] + mono[325] + mono[326] + mono[327] + mono[328] + mono[329] + mono[330] + mono[331] + mono[332] + mono[333] + mono[334] + mono[335] + mono[336] + mono[337] + mono[338] + mono[339] + mono[340] + mono[341] + mono[342] + mono[343] + mono[344] + mono[345] + mono[346] + mono[347] + mono[348] + mono[349] + mono[350] + mono[351] + mono[352] + mono[353] + mono[354] + mono[355] + mono[356] + mono[357] + mono[358] + mono[359] + mono[360] + mono[361] + mono[362] + mono[363] + mono[364] + mono[365] + mono[366] + mono[367] + mono[368] + mono[369] + mono[370] + mono[371] + mono[372] + mono[373] + mono[374] + mono[375] + mono[376] + mono[377] + mono[378] + mono[379] + mono[380];
    poly[17] = mono[381] + mono[382] + mono[383] + mono[384] + mono[385];
    poly[18] = mono[386] + mono[387] + mono[388] + mono[389] + mono[390] + mono[391] + mono[392] + mono[393] + mono[394] + mono[395] + mono[396] + mono[397] + mono[398] + mono[399] + mono[400] + mono[401] + mono[402] + mono[403] + mono[404] + mono[405] + mono[406] + mono[407] + mono[408] + mono[409] + mono[410] + mono[411] + mono[412] + mono[413] + mono[414] + mono[415];
    poly[19] = poly[5] * poly[1] - poly[15] - poly[12] - poly[14] - poly[18] - poly[12] - poly[14] - poly[14];
    poly[20] = poly[2] * poly[3] - poly[16] - poly[15] - poly[12] - poly[14] - poly[19] - poly[15] - poly[14] - poly[14];
    poly[21] = poly[1] * poly[6] - poly[16] - poly[13] - poly[15] - poly[12] - poly[20] - poly[16] - poly[13] - poly[15] - poly[12] - poly[13] - poly[13];
    poly[22] = poly[1] * poly[7] - poly[16] - poly[14];
    poly[23] = poly[1] * poly[8] - poly[16] - poly[17] - poly[15] - poly[17] - poly[17] - poly[17];
    poly[24] = poly[2] * poly[2] - poly[13] - poly[12] - poly[18] - poly[13] - poly[12] - poly[18];
    poly[25] = poly[3] * poly[3] - poly[16] - poly[13] - poly[17] - poly[15] - poly[12] - poly[22] - poly[23] - poly[21] - poly[16] - poly[13] - poly[17] - poly[15] - poly[12] - poly[22] - poly[23] - poly[21] - poly[16] - poly[13] - poly[17] - poly[16] - poly[13] - poly[17] - poly[17] - poly[17];
    poly[26] = poly[2] * poly[4] - poly[21] - poly[19];
    poly[27] = poly[3] * poly[4] - poly[22] - poly[23] - poly[20] - poly[18];
    poly[28] = poly[1] * poly[11] - poly[27] - poly[26];
    poly[29] = mono[416] + mono[417] + mono[418] + mono[419] + mono[420] + mono[421] + mono[422] + mono[423] + mono[424] + mono[425] + mono[426] + mono[427];
    poly[30] = mono[428] + mono[429] + mono[430] + mono[431] + mono[432] + mono[433] + mono[434] + mono[435] + mono[436] + mono[437] + mono[438] + mono[439] + mono[440] + mono[441] + mono[442] + mono[443] + mono[444] + mono[445] + mono[446] + mono[447] + mono[448] + mono[449] + mono[450] + mono[451] + mono[452] + mono[453] + mono[454] + mono[455] + mono[456] + mono[457] + mono[458] + mono[459] + mono[460] + mono[461] + mono[462] + mono[463] + mono[464] + mono[465] + mono[466] + mono[467] + mono[468] + mono[469] + mono[470] + mono[471] + mono[472] + mono[473] + mono[474] + mono[475] + mono[476] + mono[477] + mono[478] + mono[479] + mono[480] + mono[481] + mono[482] + mono[483] + mono[484] + mono[485] + mono[486] + mono[487];
    poly[31] = mono[488] + mono[489] + mono[490] + mono[491] + mono[492] + mono[493] + mono[494] + mono[495] + mono[496] + mono[497] + mono[498] + mono[499] + mono[500] + mono[501] + mono[502] + mono[503] + mono[504] + mono[505] + mono[506] + mono[507] + mono[508] + mono[509] + mono[510] + mono[511] + mono[512] + mono[513] + mono[514] + mono[515] + mono[516] + mono[517] + mono[518] + mono[519] + mono[520] + mono[521] + mono[522] + mono[523] + mono[524] + mono[525] + mono[526] + mono[527] + mono[528] + mono[529] + mono[530] + mono[531] + mono[532] + mono[533] + mono[534] + mono[535] + mono[536] + mono[537] + mono[538] + mono[539] + mono[540] + mono[541] + mono[542] + mono[543] + mono[544] + mono[545] + mono[546] + mono[547];
    poly[32] = mono[548] + mono[549] + mono[550] + mono[551] + mono[552] + mono[553] + mono[554] + mono[555] + mono[556] + mono[557] + mono[558] + mono[559] + mono[560] + mono[561] + mono[562] + mono[563] + mono[564] + mono[565] + mono[566] + mono[567] + mono[568] + mono[569] + mono[570] + mono[571] + mono[572] + mono[573] + mono[574] + mono[575] + mono[576] + mono[577] + mono[578] + mono[579] + mono[580] + mono[581] + mono[582] + mono[583] + mono[584] + mono[585] + mono[586] + mono[587] + mono[588] + mono[589] + mono[590] + mono[591] + mono[592] + mono[593] + mono[594] + mono[595] + mono[596] + mono[597] + mono[598] + mono[599] + mono[600] + mono[601] + mono[602] + mono[603] + mono[604] + mono[605] + mono[606] + mono[607];
    poly[33] = mono[608] + mono[609] + mono[610] + mono[611] + mono[612] + mono[613] + mono[614] + mono[615] + mono[616] + mono[617] + mono[618] + mono[619] + mono[620] + mono[621] + mono[622] + mono[623] + mono[624] + mono[625] + mono[626] + mono[627] + mono[628] + mono[629] + mono[630] + mono[631] + mono[632] + mono[633] + mono[634] + mono[635] + mono[636] + mono[637];
    poly[34] = mono[638] + mono[639] + mono[640] + mono[641] + mono[642] + mono[643] + mono[644] + mono[645] + mono[646] + mono[647] + mono[648] + mono[649] + mono[650] + mono[651] + mono[652] + mono[653] + mono[654] + mono[655] + mono[656] + mono[657] + mono[658] + mono[659] + mono[660] + mono[661] + mono[662] + mono[663] + mono[664] + mono[665] + mono[666] + mono[667];
    poly[35] = mono[668] + mono[669] + mono[670] + mono[671] + mono[672] + mono[673] + mono[674] + mono[675] + mono[676] + mono[677];
    poly[36] = mono[678] + mono[679] + mono[680] + mono[681] + mono[682] + mono[683] + mono[684] + mono[685] + mono[686] + mono[687] + mono[688] + mono[689] + mono[690] + mono[691] + mono[692] + mono[693] + mono[694] + mono[695] + mono[696] + mono[697] + mono[698] + mono[699] + mono[700] + mono[701] + mono[702] + mono[703] + mono[704] + mono[705] + mono[706] + mono[707] + mono[708] + mono[709] + mono[710] + mono[711] + mono[712] + mono[713] + mono[714] + mono[715] + mono[716] + mono[717] + mono[718] + mono[719] + mono[720] + mono[721] + mono[722] + mono[723] + mono[724] + mono[725] + mono[726] + mono[727] + mono[728] + mono[729] + mono[730] + mono[731] + mono[732] + mono[733] + mono[734] + mono[735] + mono[736] + mono[737] + mono[738] + mono[739] + mono[740] + mono[741] + mono[742] + mono[743] + mono[744] + mono[745] + mono[746] + mono[747] + mono[748] + mono[749] + mono[750] + mono[751] + mono[752] + mono[753] + mono[754] + mono[755] + mono[756] + mono[757] + mono[758] + mono[759] + mono[760] + mono[761] + mono[762] + mono[763] + mono[764] + mono[765] + mono[766] + mono[767] + mono[768] + mono[769] + mono[770] + mono[771] + mono[772] + mono[773] + mono[774] + mono[775] + mono[776] + mono[777] + mono[778] + mono[779] + mono[780] + mono[781] + mono[782] + mono[783] + mono[784] + mono[785] + mono[786] + mono[787] + mono[788] + mono[789] + mono[790] + mono[791] + mono[792] + mono[793] + mono[794] + mono[795] + mono[796] + mono[797];
    poly[37] = mono[798] + mono[799] + mono[800] + mono[801] + mono[802] + mono[803] + mono[804] + mono[805] + mono[806] + mono[807] + mono[808] + mono[809] + mono[810] + mono[811] + mono[812] + mono[813] + mono[814] + mono[815] + mono[816] + mono[817] + mono[818] + mono[819] + mono[820] + mono[821] + mono[822] + mono[823] + mono[824] + mono[825] + mono[826] + mono[827] + mono[828] + mono[829] + mono[830] + mono[831] + mono[832] + mono[833] + mono[834] + mono[835] + mono[836] + mono[837] + mono[838] + mono[839] + mono[840] + mono[841] + mono[842] + mono[843] + mono[844] + mono[845] + mono[846] + mono[847] + mono[848] + mono[849] + mono[850] + mono[851] + mono[852] + mono[853] + mono[854] + mono[855] + mono[856] + mono[857];
    poly[38] = poly[1] * poly[12] - poly[32] - poly[30] - poly[31] - poly[29] - poly[36] - poly[30] - poly[31] - poly[29] - poly[29] - poly[29] - poly[29];
    poly[39] = poly[1] * poly[13] - poly[33] - poly[30];
    poly[40] = poly[1] * poly[14] - poly[31] - poly[35];
    poly[41] = poly[2] * poly[7] - poly[32] - poly[40];
    poly[42] = mono[858] + mono[859] + mono[860] + mono[861] + mono[862] + mono[863] + mono[864] + mono[865] + mono[866] + mono[867] + mono[868] + mono[869] + mono[870] + mono[871] + mono[872] + mono[873] + mono[874] + mono[875] + mono[876] + mono[877] + mono[878] + mono[879] + mono[880] + mono[881] + mono[882] + mono[883] + mono[884] + mono[885] + mono[886] + mono[887] + mono[888] + mono[889] + mono[890] + mono[891] + mono[892] + mono[893] + mono[894] + mono[895] + mono[896] + mono[897] + mono[898] + mono[899] + mono[900] + mono[901] + mono[902] + mono[903] + mono[904] + mono[905] + mono[906] + mono[907] + mono[908] + mono[909] + mono[910] + mono[911] + mono[912] + mono[913] + mono[914] + mono[915] + mono[916] + mono[917] + mono[918] + mono[919] + mono[920] + mono[921] + mono[922] + mono[923] + mono[924] + mono[925] + mono[926] + mono[927] + mono[928] + mono[929] + mono[930] + mono[931] + mono[932] + mono[933] + mono[934] + mono[935] + mono[936] + mono[937] + mono[938] + mono[939] + mono[940] + mono[941] + mono[942] + mono[943] + mono[944] + mono[945] + mono[946] + mono[947] + mono[948] + mono[949] + mono[950] + mono[951] + mono[952] + mono[953] + mono[954] + mono[955] + mono[956] + mono[957] + mono[958] + mono[959] + mono[960] + mono[961] + mono[962] + mono[963] + mono[964] + mono[965] + mono[966] + mono[967] + mono[968] + mono[969] + mono[970] + mono[971] + mono[972] + mono[973] + mono[974] + mono[975] + mono[976] + mono[977];
    poly[43] = poly[2] * poly[8] - poly[34] - poly[31] - poly[42] - poly[34];
    poly[44] = poly[1] * poly[15] - poly[34] - poly[32] - poly[30] - poly[31] - poly[37] - poly[42] - poly[34] - poly[32] - poly[30];
    poly[45] = poly[1] * poly[16] - poly[33] - poly[34] - poly[32] - poly[31] - poly[41] - poly[43] - poly[33] - poly[34] - poly[32] - poly[33] - poly[33];
    poly[46] = poly[1] * poly[17] - poly[34];
    poly[47] = poly[2] * poly[5] - poly[30] - poly[31] - poly[29] - poly[36] - poly[35] - poly[30] - poly[29] - poly[35] - poly[29] - poly[35] - poly[29] - poly[29];
    poly[48] = poly[2] * poly[6] - poly[33] - poly[32] - poly[30] - poly[31] - poly[39] - poly[37] - poly[38] - poly[36] - poly[33] - poly[32] - poly[30] - poly[31] - poly[37];
    poly[49] = poly[3] * poly[5] - poly[34] - poly[32] - poly[30] - poly[31] - poly[37] - poly[42] - poly[38] - poly[36] - poly[40] - poly[32] - poly[31] - poly[40] - poly[31];
}

fn f_polynomials1(poly: &mut [f64; N_POLYS],mono: &[f64; N_MONOS]) {
    poly[50] = poly[3] * poly[6] - poly[33] - poly[34] - poly[32] - poly[30] - poly[31] - poly[29] - poly[45] - poly[41] - poly[43] - poly[39] - poly[44] - poly[42] - poly[38] - poly[33] - poly[34] - poly[32] - poly[30] - poly[31] - poly[29] - poly[41] - poly[43] - poly[39] - poly[44] - poly[33] - poly[34] - poly[32] - poly[30] - poly[29] - poly[33] - poly[34] - poly[30] - poly[29] - poly[29];
    poly[51] = poly[3] * poly[7] - poly[33] - poly[34] - poly[31] - poly[45] - poly[33];
    poly[52] = poly[3] * poly[8] - poly[33] - poly[34] - poly[32] - poly[30] - poly[45] - poly[46] - poly[44] - poly[33] - poly[34] - poly[32] - poly[46] - poly[46];
    poly[53] = poly[1] * poly[18] - poly[37] - poly[36] - poly[35] - poly[47] - poly[35] - poly[35];
    poly[54] = poly[4] * poly[5] - poly[44] - poly[38] - poly[40] - poly[53];
    poly[55] = poly[2] * poly[10] - poly[45] - poly[44] - poly[42] - poly[38] - poly[40] - poly[50] - poly[54] - poly[49] - poly[44] - poly[40] - poly[49];
    poly[56] = poly[4] * poly[6] - poly[45] - poly[39] - poly[42] - poly[36] - poly[55];
    poly[57] = poly[4] * poly[7] - poly[43] - poly[35];
    poly[58] = poly[4] * poly[8] - poly[41] - poly[46] - poly[37];
    poly[59] = poly[1] * poly[24] - poly[48] - poly[47];
    poly[60] = poly[1] * poly[25] - poly[51] - poly[52] - poly[50] - poly[49];
    poly[61] = poly[2] * poly[11] - poly[56] - poly[54];
    poly[62] = poly[3] * poly[11] - poly[57] - poly[58] - poly[55] - poly[53];
    poly[63] = poly[1] * poly[28] - poly[62] - poly[61];
}

// Total number of monomials = 64 

pub fn f_polynomials(r: &[f64; N_DISTANCES]) -> [f64; N_POLYS] {

    let mono: [f64; N_MONOS] = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);
    f_polynomials1(&mut poly, &mono);

    return poly;
}
