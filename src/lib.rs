#![feature(generic_const_exprs)]

use std::fs::File;
use std::io::Read;

fn point_dist<const NR: usize>(positions: &[f64; NR], x: usize, y: usize) -> f64 {
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

pub fn dist<const NR: usize>(positions: &[f64; NR]) -> [f64; get_len(NR)] {
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

pub fn morse(r: &mut [f64], l: f64) {
    for x in r {
        *x = f64::exp(-*x / l);
    }
    //r = r.iter().map(|x| f64::exp(-x / l)).collect();
}

// paper: The size of the matrix is (N + 3nN) × M
//
// M = linear coefficients, thus weights, thus = N_POLYS
//
// "n atoms"
// n(n − 1)/2 monomials, thus n = N_ATOMS

#[allow(non_snake_case)]
pub fn read_xyz(p: std::path::PathBuf, N: usize, use_grads: bool) -> (Vec<f64>, Vec<f64>){
    let mut f = File::open(p.clone()).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    let num_lines = s.lines().clone().count();
    let mut lines = s.lines();

    // First, we need to figure out how many rows we need.
    let num_atoms = lines.next().unwrap().trim_start().parse::<usize>().unwrap();
    lines.next().unwrap(); // skip energy
    let elems_in_line = lines.next().unwrap().split_whitespace().count();
    if use_grads {
        // Atom, xyz, grad_xyz
        assert!(elems_in_line == 7);
    } else {
        // Atom, xyz, (grad_xyz ignored)
        assert!(elems_in_line == 4 || elems_in_line == 7);
    }
    assert!(num_lines % (num_atoms + 2) == 0);
    let num_examples: usize = num_lines / (num_atoms + 2); // N \leq num_lines
    assert!(N <= num_examples, "The file: {}\n only contains {} entries, but you requested {}!", p.display()
            , num_examples, N);
    let num_cols = 3;
    let num_rows;
    if use_grads {
        num_rows = N + 3 * num_atoms * N;
    } else {
        num_rows = N;
    }
    let mut mat = Vec::with_capacity(num_rows * num_cols);
    let mut grads = vec![];
    if use_grads {
        grads = vec![0.0; 3 * num_atoms * N];
    }
    let mut energies = Vec::with_capacity(num_rows);

    // Phew, finally done. Now let's actually read the file.
    lines = s.lines();
    for _ in 0..N {
        let n_atoms = lines.next().unwrap().trim_start().parse::<usize>().unwrap();
        assert!(n_atoms == num_atoms);

        let mut line = lines.next().unwrap().split_whitespace();
        let energy = line.next().unwrap().trim_start().parse::<f64>().unwrap();
        energies.push(energy);

        for _ in 0..num_atoms {
            let mut line = lines.next().unwrap().split_whitespace();
            line.next().unwrap(); // skip atom name
            
            for _ in 0..3 {
                let val = line.next().unwrap().parse::<f64>().unwrap();
                mat.push(val);
            }

            if use_grads {
               for _ in 0..3 {
                   let val = line.next().unwrap().parse::<f64>().unwrap();
                   grads.push(val);
               }
            }
        }
    }
    if use_grads {
        energies.append(&mut grads);
    }
    (mat, energies)
}
