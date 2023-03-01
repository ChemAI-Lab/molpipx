#![feature(bench_black_box)]
use autodiff::autodiff;

const N: usize = 51;

fn f_monomials(r: &Vec<f32>) -> Vec<f32> {
    let mut mono: Vec<f32> = vec![0.0; 33];

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

fn f_polynomials0(poly: &mut Vec<f32>, mono: &Vec<f32>) {
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

// Total number of monomials = 51

fn f_polynomials(r: &Vec<f32>) -> Vec<f32> {
    let mono = f_monomials(r);
    let mut poly: Vec<f32> = vec![0.0; N];

    f_polynomials0(&mut poly, &mono);

    return poly;
}

fn point_dist(positions: &Vec<f32>, x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

fn dist(positions: &Vec<f32>) -> Vec<f32> {
    assert!(positions.len() % 3 == 0);
    let entries = positions.len() / 3;
    let mut r = vec![0.0; entries * (entries - 1) / 2];
    let mut pos = 0;

    for i in 0..entries {
        for j in (i + 1)..entries {
            r[pos] = point_dist(&positions, 3 * i, 3 * j);
            pos += 1;
        }
    }
    return r;
}

fn morse(r: Vec<f32>, l: f32) -> Vec<f32> {
    r.iter().map(|x| f32::exp(-x / l)).collect();
}

#[autodiff(d_energy_fwd, Reverse, Active)]
fn f_energy(#[dup] inputs: &Vec<f32>, weights: &Vec<f32>) -> f32 {
    let distances: Vec<f32> = dist(inputs);
    let z: Vec<f32> = morse(distances, 1.0);
    let outs: Vec<f32> = f_polynomials(&z);
    assert!(outs.len() == weights.len());

    let mut res = 0.0;
    for i in 0..N {
        res += weights[i] * outs[i];
    }
    return res;
}
// #[autodiff(d_energy_fwd, Reverse, Active)]
// fn rosenbrock(#[dup] x: &[f64; 2]) -> f64 {
//     let mut res = 0.0;
//     for i in 0..(x.len() - 1) {
//         let a = x[i + 1] - x[i] * x[i];
//         let b = x[i] - 1.0;
//         res += 100.0 * a * a + b * b;
//     }
//     res
// }


fn main() {
    let x: Vec<f32> = vec![
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

    let weights: Vec<f32> = vec![1.0; N];

    // Primary - correct
    let energy = f_energy(&x, &weights);
    println!("Energy: {energy}");

    // Jacobian - correct
    let dx: Vec<f32> = vec![0.0; x.len()];
    let dx1: Vec<f32> = vec![0.0; x.len()];
    let out = d_energy_fwd(&x, &mut dx, &weights);
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
