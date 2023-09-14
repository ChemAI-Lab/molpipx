#![allow(dead_code, unused)]
#![feature(slice_as_chunks)]
use autodiff::autodiff;
mod monomials_ethanol;
mod polynomials_ethanol;
use monomials_ethanol::*;
use polynomials_ethanol::*;

use faer_core::*;
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

    for i in 0..N_ATOMS {
        for j in (i + 1)..N_ATOMS {
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
fn f_energy(#[dup] inputs: &[f32; N_R], poly_outs: &mut [f32; N_POLYS], weights: &[f32; N_POLYS]) -> f32 {
    let mut distances = dist(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == weights.len());
    poly_outs.copy_from_slice(&outs);

    let mut res = 0.0;
    for i in 0..N_POLYS {
        res += weights[i] * outs[i];
    }
    return res;
}

pub const N: usize = 50_000;

#[allow(non_snake_case)]
fn main() {

    let weights = [1.0; N_POLYS];

    let ethanol_path = PathBuf::from("/h/344/drehwald/prog/msa_data/DDEthanol/Train_data_50000.xyz");
    let (mut inputs, b) = read_xyz(ethanol_path, N, false);
    assert!(inputs.len() == N * 3 * 9);
    println!("read {N} lines");

    let mut A = Mat::zeros(N, N_POLYS);

    // Primary - correct
    unsafe {
      let mut i = 0;
      for input in inputs.as_chunks_unchecked::<N_R>() {
          let mut polys = [0.0; N_POLYS];
          let out = f_energy(&input, &mut polys, &weights);
          for j in 0..N_POLYS {
              A.write(i, j, polys[j]);
          }
          i += 1;
      }
    }
    println!("Computed Polys and energy");

    let mat_ref: MatRef<f32> = A.as_ref();
    let (u,mut s,v) = get_svd(mat_ref);
    println!("Computed SVD");

    let mut s_pinv = Mat::zeros(N_POLYS, N);
    for i in 0..N_POLYS.min(N) {
        let val = s.read(i,0);
        if val > 0.0 {
            s_pinv.write(i, i, 1.0 / val);
        }
    }
    println!("Computed S_pinv");

    assert!(v.ncols() == s_pinv.nrows());
    assert!(s_pinv.ncols() == u.ncols());
    let A_pinv = v * s_pinv * u.transpose();
    let mut b_vec = Mat::zeros(N, 1);
    for i in 0..N {
        b_vec.write(i, 0, b[i]);
    }
    println!("Computed Pseudoinverse");

    let x_min = A_pinv.clone() * b_vec.clone();
    println!("Computed x_min");

    let I = A_pinv * A.clone();
    for i in 0..I.nrows() {
        println!("{}", I.read(i, i));
    }
    println!("Computed I");

    let b_hat = A * x_min.clone();

    for i in 0..N_R {
       println!("{} : {}", b_hat.read(i, 0), b[i]);
    }



    // Jacobian - correct
    //let mut dx = [0.0; N_R];
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
