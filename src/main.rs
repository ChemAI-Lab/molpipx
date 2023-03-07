#![feature(bench_black_box)]
use autodiff::autodiff;
mod monomials_;
mod polynomials_;
use polynomials_::*;
use monomials_::*;

fn point_dist(positions: &[f32; N_R], x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

fn dist(positions: &[f32;N_R]) -> [f32; N_DISTANCES] {
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


fn main() {
    let x: [f32; N_R] = [
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
