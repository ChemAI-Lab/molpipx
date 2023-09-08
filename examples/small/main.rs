#![allow(dead_code, unused)]
use autodiff::autodiff;
mod monomials_;
mod polynomials_;
use monomials_::*;
use polynomials_::*;

use faer_svd::{
    bidiag::bidiagonalize_in_place, bidiag_real_svd::compute_bidiag_real_svd, compute_svd,
    SvdParams,
};
use reborrow::IntoConst;
use dyn_stack::*;
use faer_core::{mat, Mat, MatRef, MatMut, Parallelism};
use core::ops::Mul;

use pipenzyme::*;

// compiling single poly fn
// and all monomial fns takes 24s
// and one monomial fn  takes  9s
//
// compiling all monomial fns
// and a single poly fn  takes 24s
// and 50       poly fns takes 37s

#[autodiff(d_energy_rev, Reverse, Active, Duplicated, Const)]
fn f_energy(inputs: &[f32; N_R], weights: &[f32; N_POLYS]) -> f32{
    let mut distances = dist::<N_R>(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == weights.len());

    let mut res = 0.0;
    for i in 0..N_POLYS {
        res += weights[i] * outs[i];
    }
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
// 
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

    // let x: [f32; N_R] = [// 8 * 8 + 2 = 66
    //     1.5534098, 4.9921384, 5.9013925, 3.1407506, 2.3746452, 2.3064625, 2.4243245, 0.9831121,
    //     4.9590926, 3.1897664, 1.7556409, 6.280233, 7.3188167, 3.9259133, 2.6457272, 3.8236654,
    //     3.7039938, 0.9613827, 6.3667183, 3.6145263, 3.2313368, 1.5631986, 3.747352, 4.6520596,
    //     3.8534281, 5.2598147, 5.3290095, 0.99386555, 4.2686257, 4.4923944, 4.554017, 5.7103868,
    //     4.369564, 5.5798626, 6.4054008, 0.96123296, 5.2826056, 4.9923925, 1.5310733, 3.7889235,
    //     4.372375, 3.1611478, 3.7787771, 0.98322767, 3.5199509, 3.8987951, 4.347726, 1.9725064,
    //     4.831806, 0.9698857, 3.5025046, 1.5626531, 3.1857383, 3.5315819, 4.3042774, 0.99871117,
    //     3.4016144, 4.806191, 4.8917413, 0.9615672, 5.450229, 2.9030426, 2.7246506, 4.482426,
    //     4.144017, 4.0145044,
    // ];

    let weights = [1.0; N_POLYS];

    // Primary - correct
    let energy = f_energy(&x, &weights);
    println!("Energy: {energy}");

    // Jacobian - correct
    let mut dx_rev = [0.0; N_R];
    let mut dx_fwd = [0.0; N_R];
    let mut out = 0.0;
    //let out = d_energy_rev2(&x, &mut dx_rev, &weights, 1.0);
   
    for i in 0..5 {
        let m = 4;
        let n = 3;

        d_energy_inplace_rev(&x, &mut dx_rev, &weights, &mut out, &1.0);
        // calculate svd(dx_rev) to invert it.
        // update x with it
        //x -= 0.005 * dx_rev;
        let ptr = dx_rev.as_mut_ptr();
        let mat_mut: MatMut<'_, f32> = unsafe { MatMut::from_raw_parts(ptr, m, n, 1, 1) };
        let mat_const: Mat<f32> = mat_mut.to_owned();
        let mat_ref: MatRef<f32> = mat_const.as_ref();
        dbg!(&mat_ref);
       
        
        let size = m.min(n);
        let mut s = Mat::zeros(size, 1);
        let mut s_expanded = Mat::zeros(m, n);
        let mut u = Mat::zeros(m, m);
        let mut v = Mat::zeros(n, n);
        let mut mem = GlobalMemBuffer::new(
            faer_svd::compute_svd_req::<f64>(
                m,
                n,
                faer_svd::ComputeVectors::Full,
                faer_svd::ComputeVectors::Full,
                Parallelism::None,
                SvdParams::default(),
            )
            .unwrap(),
        );
        let mut stack = DynStack::new(&mut mem);
        let mut mat_cpy = Mat::zeros(m, n);
        let mat_cpy = mat_cpy.clone_from(&mat_const);
        compute_svd(
                    mat_ref,
                    s.as_mut().col(0),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    f32::EPSILON,
                    f32::MIN_POSITIVE,
                    Parallelism::None,
                    stack.rb_mut(),
                    SvdParams::default(),
                );
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
    
    for (i, val) in dx_rev.iter().enumerate() {
        if i % 3 == 0 {
            println!("");
        }
        print!("{:10.3e} ", val);
    }
    println!("");

    for (i, val) in dx_fwd.iter_mut().enumerate() {
        let mut dx_activity = [0.0; N_R];
        dx_activity[i] = 1.0;
        *val = d_energy_fwd(&x, &mut dx_activity, &weights);
        //(*val, _) = d_energy_fwd(&x, &mut dx_activity, &weights);
    }
    
    for (i, val) in dx_fwd.iter().enumerate() {
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
