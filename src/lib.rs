#![feature(generic_const_exprs)]

use faer_svd::{compute_svd, SvdParams};
use dyn_stack::*;
use faer_core::{Mat, MatRef, Parallelism};

fn point_dist<const NR: usize>(positions: &[f32; NR], x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

pub const fn get_len(x: usize) -> usize {
    assert!(x % 3 == 0);
    let n = x / 3;
    (n * (n - 1)) / 2
}

pub fn dist<const NR: usize>(positions: &[f32; NR]) -> [f32; get_len(NR)] {
    assert!(NR % 3 == 0);
    let mut r = [0.0; get_len(NR)];
    let mut pos = 0;

    for i in 0..(NR / 3) {
        for j in (i + 1)..(NR / 3) {
            r[pos] = point_dist::<NR>(positions, 3 * i, 3 * j);
            pos += 1;
        }
    }
    return r;
}

pub fn morse(r: &mut [f32], l: f32) {
    for x in r {
        *x = f32::exp(-*x / l);
    }
    //r = r.iter().map(|x| f32::exp(-x / l)).collect();
}

pub fn get_svd(mat_ref: MatRef<f32>) -> (Mat<f32>, Mat<f32>, Mat<f32>) {
    let m = mat_ref.nrows();
    let n = mat_ref.ncols();
    let size = m.min(n);
    let mut s = Mat::zeros(size, 1);
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
    (u, s, v)
}

