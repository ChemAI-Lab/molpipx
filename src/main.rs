#![allow(dead_code, unused)]
#![feature(slice_as_chunks)]
use autodiff::autodiff;
mod monomials_ethanol;
mod polynomials_ethanol;
use monomials_ethanol::*;
use polynomials_ethanol::*;

use faer::*;
use pipenzyme::*;

use std::path::PathBuf;

// compiling single poly fn
// and all monomial fns takes 24s
// and one monomial fn  takes  9s
//
// compiling all monomial fns
// and a single poly fn  takes 24s
// and 50       poly fns takes 37s

fn point_dist(positions: &[f64; N_XYZ], x: usize, y: usize) -> f64 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

fn dist(positions: &[f64; N_XYZ]) -> [f64; N_DISTANCES] {
    assert!(positions.len() % 3 == 0);
    let mut r = [0.0; N_DISTANCES];
    let mut pos = 0;

    for i in 0..N_ATOMS {
        for j in (i + 1)..N_ATOMS {
            r[pos] = point_dist(&positions, 3 * i, 3 * j);
            pos += 1;
        }
    }
    return r;
}

fn morse(r: &mut [f64], l: f64) {
    for x in r {
        *x = f64::exp(-*x / l);
    }
    //r = r.iter().map(|x| f64::exp(-x / l)).collect();
}

#[autodiff(d_f, Reverse, Const)]
fn f(#[dup] inputs: &[f64; N_XYZ], #[dup] poly_outs: &mut [f64; N_POLYS]) {
    let mut distances = dist(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == poly_outs.len());
    poly_outs.copy_from_slice(&outs);
}
//#[autodiff(d_energy_rev, Reverse, Active)]
//fn f_energy(#[dup] inputs: &[f64; N_XYZ], poly_outs: &mut [f64; N_POLYS], weights: &[f64; N_POLYS]) -> f64 {
//    let mut distances = dist(inputs);
//    morse(&mut distances, 1.0);
//    let outs = f_polynomials(&distances);
//    assert!(outs.len() == weights.len());
//    poly_outs.copy_from_slice(&outs);
//
//    let mut res = 0.0;
//    for i in 0..N_POLYS {
//        res += weights[i] * outs[i];
//    }
//    return res;
//}

pub const N: usize = 50_000;

#[allow(non_snake_case)]
fn main() {

    let weights = [1.0; N_POLYS];

    let use_grads = false;
    let ethanol_path = PathBuf::from("/h/344/drehwald/prog/msa_data/DDEthanol/Train_data_50000.xyz");
    assert!(ethanol_path.exists());
    let (mut inputs, b) = read_xyz(ethanol_path, N, use_grads);
    assert!(inputs.len() == N * 3 * N_ATOMS);
    println!("read {N} lines");

    let m = if use_grads { 3*N_ATOMS*N+N } else { N };
    let n = N_POLYS;
    println!("Allocating: {} GB for A", m * n * 4 / 1024 / 1024 / 1024);
    let mut A = Mat::zeros(m, n);
    assert!(A.nrows() == m);
    assert!(A.ncols() == n);

    // Primary - correct
    unsafe {
      for (i, input) in inputs.as_chunks_unchecked::<N_XYZ>().iter().enumerate() {
          let mut polys = [0.0; N_POLYS];
          f(&input, &mut polys);
          for j in 0..N_POLYS {
              A[(i, j)] = polys[j];
          }
      }
    }
    if use_grads {
    }
    println!("Computed Polys and energy");
    for i in 0..5 {
        for j in 0..N_ATOMS {
            println!("{} {} {}", inputs[i * 3 * N_ATOMS + 3 * j], inputs[i * 3 * N_ATOMS + 3 * j + 1], inputs[i * 3 * N_ATOMS + 3 * j + 2]);
        }
    }

    let svd = A.svd();
    println!("Computed SVD");
    let s_diag = svd.s_diagonal();
    for i in 0..Ord::min(m, n) {
        println!("diagonal: {} {}", i, s_diag[(i, 0)]);
    }
    let mut s_inv = Mat::zeros(n, m);
    for i in 0..Ord::min(m, n) {
        s_inv[(i, i)] = 1.0 / s_diag[(i, 0)];
    }
    let pseudoinv = svd.v() * &s_inv * svd.u().adjoint();
    println!("Computed Pseudoinverse");
    
    assert_matrix_eq!(
        &pseudoinv * &A,
        Mat::identity(n, n),
        comp = abs,
        tol = 1e-10
    );

    let mut b_vec = Mat::zeros(N, 1);
    for i in 0..N {
        b_vec[(i, 0)] = b[i];
    }

    let x_min = pseudoinv.clone() * b_vec.clone();
    println!("Computed x_min");

    let I = pseudoinv * A.clone();
    for i in 0..I.nrows() {
        println!("{}", I[(i, i)]);
    }
    println!("Computed I");

    let b_hat = A * x_min.clone();

    for i in 0..N_XYZ {
       println!("{} : {}", b_hat[(i, 0)], b[i]);
    }

    // Jacobian - correct
    //let mut x = [0.0; N_XYZ];
    //for i in 0..N_XYZ {
    //    x[i] = inputs[i];
    //}
    //let mut dx = [0.0; N_XYZ];
    //let mut polys2 = [0.0; N_POLYS];
    //let out = d_energy_rev(&x, &mut dx, &mut polys2, &weights, 1.0);
    //for (i, val) in dx.iter().enumerate() {
    //    if i % 3 == 0 {
    //        println!("");
    //    }
    //    print!("{:10.3e} ", val);
    //}
    println!("");
}
