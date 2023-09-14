#![feature(generic_const_exprs)]

use std::fs::File;
use std::io::Read;
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

// paper: The size of the matrix is (N + 3nN) × M
//
// M = linear coefficients, thus weights, thus = N_POLYS
//
// "n atoms"
// n(n − 1)/2 monomials, thus n = N_POINTS
//
// N energies and 3N components of the gradient vector obtained at N conﬁgurations, thus N = 

#[allow(non_snake_case)]
pub fn read_xyz(p: std::path::PathBuf, M: usize) -> (Mat<f32>, Vec<f32>){
    let mut f = File::open(p).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    let num_lines = s.lines().clone().count();
    let mut lines = s.lines();

    // First, we need to figure out how many rows we need.
    println!("num_lines: {}", num_lines);
    //println!("foo: {}", lines.next().unwrap());
    let num_atoms = lines.next().unwrap().trim_start().parse::<usize>().unwrap();
    lines.next().unwrap(); // skip energy
    let elems_in_line = lines.next().unwrap().split_whitespace().count();
    assert!(elems_in_line == 4 || elems_in_line == 7);
    let has_grads = elems_in_line == 7;
    assert!(num_lines % (num_atoms + 2) == 0);
    let N: usize = num_lines / (num_atoms + 2) ; // N \leq num_lines
    println!("num_lines: {}", num_lines);
    let mut num_cols = 3;
    let mut num_rows = N;
    if has_grads {
        num_rows += 3 * num_atoms * N;
    }
    let mut mat = Mat::zeros(num_rows, num_cols);
    let mut grads = vec![];
    if has_grads {
        grads = vec![0.0; 3 * num_atoms * N];
    }
    let mut energies = Vec::with_capacity(num_rows);

    // Phew, finally done. Now let's actually read the file.
    lines = s.lines();
    for i in 0..N {
        let n_atoms = lines.next().unwrap().trim_start().parse::<usize>().unwrap();
        assert!(n_atoms == num_atoms);

        let mut line = lines.next().unwrap().split_whitespace();
        let energy = line.next().unwrap().trim_start().parse::<f32>().unwrap();
        energies.push(energy);

        for _ in 0..num_atoms {
            let mut line = lines.next().unwrap().split_whitespace();
            line.next().unwrap(); // skip atom name
            
            for j in 0..3 {
                let val = line.next().unwrap().parse::<f32>().unwrap();
                mat.write(i, j, val);
            }

            if has_grads {
               for j in 0..3 {
                   let val = line.next().unwrap().parse::<f32>().unwrap();
                   grads.push(val);
               }
            }
        }
    }
    if has_grads {
        energies.append(&mut grads);
    }
    (mat, energies)
}
//   9
//  -154.895543944021     
//C       0.45998423     -0.13402741     -0.38149377      0.03725021      0.00255395     -0.00297219
//C      -1.08503757     -0.18668855     -0.70739706      0.01466442     -0.03849773     -0.00287445
//O       0.63320073      0.34857790      0.96004736     -0.03041850     -0.05188480      0.03728054
//H       0.66044727     -1.18971514     -0.31044642     -0.01889792      0.01630660      0.01234007
//H       1.12499407      0.39646951     -1.11250080      0.01168768      0.00503003     -0.01242474
//H      -1.61877352      0.80043915     -0.97438596     -0.02103924      0.01736866     -0.00762434
//H      -1.37298351     -0.85103728     -1.50934231     -0.00523914      0.01077300      0.01136568
//H      -1.66049066     -0.53890454      0.18361117     -0.01032167     -0.00046999      0.00458850
//H       0.26297997     -0.33017727      1.45919881      0.02233052      0.03882035     -0.03969332
