#![allow(dead_code, unused)]
#![feature(bench_black_box)]
use autodiff::autodiff;
mod monomials_ethanol;
mod polynomials_ethanol;
use monomials_ethanol::*;
use polynomials_ethanol::*;

use pipenzyme::*;

use std::path::PathBuf;

// compiling single poly fn
// and all monomial fns takes 24s
// and one monomial fn  takes  9s
//
// compiling all monomial fns
// and a single poly fn  takes 24s
// and 50       poly fns takes 37s

fn point_dist(positions: &[f32; N_R], x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

fn dist(positions: &[f32; N_R]) -> [f32; N_DISTANCES] {
    assert!(positions.len() % 3 == 0);
    let mut r = [0.0; N_DISTANCES];
    let mut pos = 0;

    for i in 0..N_POINTS {
        for j in (i + 1)..N_POINTS {
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
fn f_energy(#[dup] inputs: &[f32; N_R], weights: &[f32; N_POLYS]) -> f32 {
    let mut distances = dist(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == weights.len());

    let mut res = 0.0;
    for i in 0..N_POLYS {
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

// const vector<double> r1 = {
//   1.5534098, 4.9921384,  5.9013925, 3.1407506,  2.3746452, 2.3064625,
//   2.4243245, 0.9831121,  4.9590926, 3.1897664,  1.7556409, 6.280233,
//   7.3188167, 3.9259133,  2.6457272, 3.8236654,  3.7039938, 0.9613827,
//   6.3667183, 3.6145263,  3.2313368, 1.5631986,  3.747352,  4.6520596,
//   3.8534281, 5.2598147,  5.3290095, 0.99386555, 4.2686257, 4.4923944,
//   4.554017,  5.7103868,  4.369564,  5.5798626,  6.4054008, 0.96123296,
//   5.2826056, 4.9923925,  1.5310733, 3.7889235,  4.372375,  3.1611478,
//   3.7787771, 0.98322767, 3.5199509, 3.8987951,  4.347726,  1.9725064,
//   4.831806,  0.9698857,  3.5025046, 1.5626531,  3.1857383, 3.5315819,
//   4.3042774, 0.99871117, 3.4016144, 4.806191,   4.8917413, 0.9615672,
//   5.450229,  2.9030426,  2.7246506, 4.482426,   4.144017,  4.0145044};
//
// const vector<double> r2(6, 1.0);
//
// const vector<double> pos_vec = {
//     -0.59298831, 2.17569828,  1.30499411,  -1.99557149, 2.68355870,
//     1.73849177,  1.59105551,  -1.43353951, -1.36423826, 3.00556111,
//     -1.16198051, -1.97170806, -0.65665430, 1.53734851,  -1.76954162,
//     -1.75324965, 1.73531950,  -0.71955508, 1.56205964,  1.35383236,
//     1.29633057,  1.70314455,  2.86626911,  1.66309309,  -1.55682957,
//     1.99913681,  1.22533596,  2.32177281,  -0.76277858, -1.42669415,
//     -1.61325145, 1.37598705,  -1.60947585, 1.15495706,  2.25580525,
//     1.16165924};

fn main() {
    // let x: [f32; N_R] = [
    //     0.00000000,
    //     0.00000000,
    //     14.00307401,
    //     0.00000000,
    //     -1.98144397,
    //     1.00782504,
    //     -1.71598081,
    //     0.99072198,
    //     1.00782504,
    //     1.71598081,
    //     0.99072198,
    //     1.00782504,
    // ];

    let x: [f32; N_R] = [// 8 * 8 + 2 = 66
        1.5534098, 4.9921384, 5.9013925, 3.1407506, 2.3746452, 2.3064625, 2.4243245, 0.9831121,
        4.9590926, 3.1897664, 1.7556409, 6.280233, 7.3188167, 3.9259133, 2.6457272, 3.8236654,
        3.7039938, 0.9613827, 6.3667183, 3.6145263, 3.2313368, 1.5631986, 3.747352, 4.6520596,
        3.8534281, 5.2598147, 5.3290095, 0.99386555, 4.2686257, 4.4923944, 4.554017, 5.7103868,
        4.369564, 5.5798626, 6.4054008, 0.96123296, 5.2826056, 4.9923925, 1.5310733, 3.7889235,
        4.372375, 3.1611478, 3.7787771, 0.98322767, 3.5199509, 3.8987951, 4.347726, 1.9725064,
        4.831806, 0.9698857, 3.5025046, 1.5626531, 3.1857383, 3.5315819, 4.3042774, 0.99871117,
        3.4016144, 4.806191, 4.8917413, 0.9615672, 5.450229, 2.9030426, 2.7246506, 4.482426,
        4.144017, 4.0145044,
    ];

    let weights = [1.0; N_POLYS];

    let ethanol_path = PathBuf::from("/h/344/drehwald/prog/msa_data/DDEthanol/Train_data_50000.xyz");
    let mut mat = read_xyz(ethanol_path, N_POINTS);

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
