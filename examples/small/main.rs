#![allow(dead_code, unused)]
#![feature(generic_const_exprs)]
use autodiff::autodiff;
mod monomials_;
mod polynomials_;
use monomials_::*;
use polynomials_::*;

use faer_core::{mat, Mat, MatRef, MatMut, Parallelism};

use std::ops::Mul;
use pipenzyme::*;


// paper: The size of the matrix is (N + 3nN) × M
//
// M = linear coefficients, thus weights, thus = N_POLYS
//
// "n atoms"
// n(n − 1)/2 monomials, thus n = N_POINTS
//
// N energies and 3N components of the gradient vector obtained at N conﬁgurations, thus N = 

// compiling single poly fn
// and all monomial fns takes 24s
// and one monomial fn  takes  9s
//
// compiling all monomial fns
// and a single poly fn  takes 24s
// and 50       poly fns takes 37s

#[autodiff(d_energy_rev, Reverse, Active, Duplicated, Const)]
fn f_energy(inputs: &[f32; N_R], weights: &[f32; N_POLYS]) -> f32{
    let mut res = 0.0;
    f_energy_inplace(inputs, weights, &mut res);
    res
}

fn f_energy_inplace(inputs: &[f32; N_R], weights: &[f32; N_POLYS], res: &mut f32) {
    let mut distances = dist::<N_R>(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == weights.len());

    for i in 0..N_POLYS {
        *res += weights[i] * outs[i];
    }
}

#[autodiff(f_energy_inplace, Reverse, Const, Duplicated, Const, Duplicated)]
fn d_energy_inplace_rev(inputs: &[f32; N_R], d_inputs: &mut[f32; N_R], weights: &[f32; N_POLYS], res: &mut f32, res_t: & f32);

#[autodiff(d_energy_fwd, Forward, DuplicatedNoNeed)]
fn f_energy2(#[dup] inputs: &[f32; N_R], weights: &[f32; N_POLYS]) -> f32 {
    f_energy(inputs, weights)
}

// #[autodiff(d_energy_fwd, Forward, DuplicatedNoNeed, Const)]
// fn f_energy_rev(inputs: &[f32; N_R], &input_shaddow: &[f32; N_R], weights: &[f32; N_POLYS]) -> f32 {
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
    let mut dx_rev = [0.0; N_R];
    let mut dx_fwd = [0.0; N_R];
    let mut out = 0.0;
    //let out = d_energy_rev2(&x, &mut dx_rev, &weights, 1.0);

    let primal_elems = get_len(N_R);
    let width = 

    let mut large_mat = Mat::zeros();
   
    for i in 0..5 {
        let m = 4;
        let n = 3;
        let size = m.min(n);
        let mut dx = [0.0; N_R];

        d_energy_inplace_rev(&x, &mut dx, &weights, &mut out, &1.0);
        let ptr = dx.as_mut_ptr();
        let mat_mut: MatMut<'_, f32> = unsafe { MatMut::from_raw_parts(ptr, m, n, 1, 1) };
        let mat_const: Mat<f32> = mat_mut.to_owned();
        let mat_ref: MatRef<f32> = mat_const.as_ref();
        dbg!(&mat_ref);
      
        let (u,s,v) = get_svd(mat_ref);
        
        let mut s_expanded = Mat::zeros(m, n);
        for i in 0..size {
            s_expanded.write(i, i, s.read(i, 0));
        }
        let res = u.mul(s_expanded).mul(v.transpose());
        dbg!(&res);
        // comparing that res and mat_ref are the same
        let mut diff = 0.0;
        for i in 0..m {
          for j in 0..n {
            diff += (res.read(i, j) - mat_ref.read(i, j)).powi(2);
          }
        }
        println!("diff: {}", diff);
    }

    for (i, val) in dx_fwd.iter_mut().enumerate() {
        let mut dx_activity = [0.0; N_R];
        dx_activity[i] = 1.0;
        *val = d_energy_fwd(&x, &mut dx_activity, &weights);
    }
    d_energy_inplace_rev(&x, &mut dx_rev, &weights, &mut out, &1.0);
  
    assert!(dx_rev.len() == dx_fwd.len());
    for i in 0..dx_rev.len() {
      println!("rev: {:10.3e}, fwd: {:10.3e}", dx_rev[i], dx_fwd[i]);
    }
}
