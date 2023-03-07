#![feature(bench_black_box)]
use autodiff::autodiff;

const N_R: usize = 6;
const N_R: usize = 12;
const N_MONOS: usize = 51;
const N_POLYS: usize = 51;

fn f_monomials(r: &[f32; 6]) -> [f32; N_MONOS] {
    let mut mono = [0.0; N_MONOS];

    mono[0] = 1.0;
    mono[1] = r[5];
    mono[2] = r[4];
    mono[3] = r[3];
    mono[4] = r[2];
    mono[5] = r[1];
    mono[6] = r[0];

    mono[7] = mono[1] * mono[2];
    mono[8] = mono[1] * mono[3];
    mono[9] = mono[2] * mono[3];
    mono[10] = mono[3] * mono[4];
    mono[11] = mono[2] * mono[5];
    mono[12] = mono[1] * mono[6];
    mono[13] = mono[4] * mono[5];
    mono[14] = mono[4] * mono[6];
    mono[15] = mono[5] * mono[6];
    mono[16] = mono[1] * mono[9];
    mono[17] = mono[1] * mono[10];
    mono[18] = mono[2] * mono[10];
    mono[19] = mono[1] * mono[11];
    mono[20] = mono[3] * mono[11];
    mono[21] = mono[2] * mono[12];
    mono[22] = mono[3] * mono[12];
    mono[23] = mono[2] * mono[13];
    mono[24] = mono[3] * mono[13];
    mono[25] = mono[1] * mono[14];
    mono[26] = mono[3] * mono[14];
    mono[27] = mono[1] * mono[15];
    mono[28] = mono[2] * mono[15];
    mono[29] = mono[4] * mono[15];
    mono[30] = mono[2] * mono[24];
    mono[31] = mono[1] * mono[26];
    mono[32] = mono[1] * mono[28];

    return mono;
}

fn f_polynomials0(poly: &mut [f32;N_POLY], mono: &[f32;N_MONO]) {
    poly[0] = mono[0];
    poly[1] = mono[1] + mono[2] + mono[3];
    poly[2] = mono[4] + mono[5] + mono[6];
    poly[3] = mono[7] + mono[8] + mono[9];
    poly[4] = mono[10] + mono[11] + mono[12];
    poly[5] = poly[1] * poly[2] - poly[4];
    poly[6] = mono[13] + mono[14] + mono[15];
    poly[7] = poly[1] * poly[1] - poly[3] - poly[3];
    poly[8] = poly[2] * poly[2] - poly[6] - poly[6];
    poly[9] = mono[16];
    poly[10] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
    poly[11] = poly[2] * poly[3] - poly[10];
    poly[12] = mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28];
    poly[13] = poly[1] * poly[6] - poly[12];
    poly[14] = mono[29];
    poly[15] = poly[1] * poly[3] - poly[9] - poly[9] - poly[9];
    poly[16] = poly[1] * poly[4] - poly[10];
    poly[17] = poly[2] * poly[7] - poly[16];
    poly[18] = poly[2] * poly[4] - poly[12];
    poly[19] = poly[1] * poly[8] - poly[18];
    poly[20] = poly[2] * poly[6] - poly[14] - poly[14] - poly[14];
    poly[21] = poly[1] * poly[7] - poly[15];
    poly[22] = poly[2] * poly[8] - poly[20];
    poly[23] = poly[9] * poly[2];
    poly[24] = mono[30] + mono[31] + mono[32];
    poly[25] = poly[3] * poly[6] - poly[24];
    poly[26] = poly[14] * poly[1];
    poly[27] = poly[9] * poly[1];
    poly[28] = poly[3] * poly[4] - poly[23];
    poly[29] = poly[1] * poly[10] - poly[23] - poly[28] - poly[23];
    poly[30] = poly[1] * poly[11] - poly[23];
    poly[31] = poly[1] * poly[12] - poly[25] - poly[24] - poly[24];
    poly[32] = poly[1] * poly[13] - poly[25];
    poly[33] = poly[4] * poly[5] - poly[25] - poly[31];
    poly[34] = poly[2] * poly[11] - poly[25];
    poly[35] = poly[4] * poly[6] - poly[26];
    poly[36] = poly[2] * poly[12] - poly[26] - poly[35] - poly[26];
    poly[37] = poly[2] * poly[13] - poly[26];
    poly[38] = poly[14] * poly[2];
    poly[39] = poly[3] * poly[3] - poly[27] - poly[27];
    poly[40] = poly[3] * poly[7] - poly[27];
    poly[41] = poly[1] * poly[16] - poly[28];
    poly[42] = poly[2] * poly[21] - poly[41];
    poly[43] = poly[1] * poly[18] - poly[33];
    poly[44] = poly[7] * poly[8] - poly[43];
    poly[45] = poly[6] * poly[6] - poly[38] - poly[38];
    poly[46] = poly[2] * poly[18] - poly[35];
    poly[47] = poly[1] * poly[22] - poly[46];
    poly[48] = poly[6] * poly[8] - poly[38];
    poly[49] = poly[1] * poly[21] - poly[40];
    poly[50] = poly[2] * poly[22] - poly[48];
}

fn f_polynomials(r: &[f32;6]) -> [f32;N_POLYS] {
    let mono = f_monomials(r);
    let mut poly = [0.0; N_POLYS];

    f_polynomials0(&mut poly, &mono);

    return poly;
}

fn point_dist(positions: &[f32;N_R], x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

// 4*3/2
fn dist(positions: &[f32;N_R]) -> [f32;6] {
    assert!(positions.len() % 3 == 0);
    const entries: usize = 3*4/3;
    let mut r = [0.0; entries * (entries - 1) / 2];
    let mut pos = 0;

    for i in 0..entries {
        for j in (i + 1)..entries {
            r[pos] = point_dist(&positions, 3 * i, 3 * j);
            pos += 1;
        }
    }
    return r;
}

fn morse(r: &mut [f32], l: f32) {
    for x in r {
        *x = f32::exp(-*x / l);
    }
    //r = r.iter().map(|x| f32::exp(-x / l)).collect();
}

#[autodiff(d_energy_rev, Reverse, Active)]
fn f_energy(#[dup] inputs: &[f32;N_R], weights: &[f32;N_POLYS]) -> f32 {
    let mut distances = dist(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == weights.len());

    let mut res = 0.0;
    for i in 0..N {
        res += weights[i] * outs[i];
    }
    return res;
}


fn main() {
    let x: [f32;N_R] = [
        0.00000000,
        0.00000000,
        14.00307401,
        0.00000000,
        -1.98144397,
        1.00782504,
        -1.71598081,
        0.99072198,
        1.00782504,
        1.71598081,
        0.99072198,
        1.00782504,
    ];

    let weights = [1.0; N_POLYS];

    // Primary - correct
    let energy = f_energy(&x, &weights);
    println!("Energy: {energy}");

    // Jacobian - correct
    let mut dx = [0.0; N_R];
    let out = d_energy_rev(&x, &mut dx, &weights, 1.0);
    for (i, val) in dx.iter().enumerate() {
        if i % 3 == 0 {
            println!("");
        }
        print!("{:10.3e} ", val);
    }
    println!("");
    // let dx1: Vec<f32> = vec![0.0; x.len()];
    // jac_fwd(x, dx, weights);
    // jac_rev(x, dx1, weights);
    // std::cout << std::endl << "jac fwd: ";
    // for (size_t i = 0; i < dx.size(); i++) {
    //   if (i % 3 == 0)
    //     std::cout << std::endl;
    //   std::cout << dx[i] << " \t";
    // }
    // std::cout << std::endl << std::endl << "jac rev: ";
    // for (size_t i = 0; i < dx1.size(); i++) {
    //   if (i % 3 == 0)
    //     std::cout << std::endl;
    //   std::cout << dx1[i] << " \t";
    // }
    // std::cout << std::endl;

    // // Hessian - incorrect
    // vd dd_x(x.size() * x.size());
    // hess_fwdfwd2(x, dd_x, weights);

    // for (size_t i = 0; i < dd_x.size(); i++) {
    //   if (i % x.size() == 0)
    //     std::cout << std::endl;
    //   std::cout << "\t" << i << ": \t" << dd_x[i];
    // }
}
