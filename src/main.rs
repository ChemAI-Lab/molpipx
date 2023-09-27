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

pub const N: usize = 50;
//pub const N: usize = 50_000;

#[allow(non_snake_case)]
fn main() {

    let use_grads = std::env::var("USE_GRADS").is_ok();
    println!("use_grads: {use_grads}");
    let ethanol_path = PathBuf::from("/h/344/drehwald/prog/msa_data/DDEthanol/Train_data_50000.xyz");
    assert!(ethanol_path.exists());
    let (mut inputs, b) = read_xyz(ethanol_path, N, use_grads);
    println!("read {N} lines");

    let m = if use_grads { N_XYZ*N+N } else { N };
    let n = N_POLYS;
    assert!(inputs.len() == N * N_XYZ);
    assert!(b.len() == m);
    println!("Allocating: {} GB for A", m * n * 4 / 1024 / 1024 / 1024);
    let mut A = Mat::zeros(m, n);

    // Primary - correct
    unsafe {
      for (i, molecule) in inputs.as_chunks_unchecked::<N_XYZ>().iter().enumerate() {
          let mut polys = [0.0; N_POLYS];
          f(&molecule, &mut polys);
          for j in 0..N_POLYS {
              A[(i, j)] = polys[j];
          }
          assert!(i < N);
      }
    }
    if use_grads {
        unsafe {
            assert!(inputs.len() / N_XYZ == N);
            for (i, molecule) in inputs.as_chunks_unchecked::<N_XYZ>().iter().enumerate() {
                for j in 0..N_POLYS {
                    let mut dmolecule = [0.0; N_XYZ];
                    let mut polys = [0.0; N_POLYS];
                    let mut dpolys = [0.0; N_POLYS];
                    dpolys[j] = 1.0;
                    d_f(&molecule, &mut dmolecule, &mut polys, &mut dpolys);
                    for k in 0..N_XYZ {
                        A[(N + i * N_XYZ + k, j)] = dmolecule[k];
                    }
                }
            }
        }
    }
    println!("Computed Polys and energy");

    //let regularization: f64 = 0.00001;
    //for i in 0..N_POLYS {
    //    A[(i, i)] += regularization;
    //}
    let svd = A.svd();
    println!("Computed SVD");
    let s_diag = svd.s_diagonal();
    let mut s_inv = Mat::zeros(n, m);
    for i in 0..Ord::min(m, n) {
        s_inv[(i, i)] = 1.0 / s_diag[(i, 0)];
    }
    let pseudoinv = svd.v() * &s_inv * svd.u().adjoint();
    println!("Computed Pseudoinverse");
    
    let mut b_vec = Mat::zeros(m, 1);
    for i in 0..m {
        b_vec[(i, 0)] = b[i];
    }
    let x_min = pseudoinv * b_vec.clone();
    println!("Computed x_min");

    let b_hat = A * x_min.clone();
    for i in 0..N_XYZ {
       println!("{} : {}", b_hat[(i, 0)], b[i]);
    }

    println!("");
}
