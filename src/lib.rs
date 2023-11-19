#![feature(generic_const_exprs)]

use std::fs::File;
use std::io::{Read, BufReader};

pub const fn get_len(x: usize) -> usize {
    assert!(x % 3 == 0);
    let n = x / 3;
    (n * (n - 1)) / 2
}

// paper: The size of the matrix is (N + 3nN) × M
//
// M = linear coefficients, thus weights, thus = N_POLYS
//
// "n atoms"
// n(n − 1)/2 monomials, thus n = N_ATOMS

#[allow(non_snake_case)]
pub fn write_xyz(p: std::path::PathBuf, A: Vec<f64>, b: Vec<f64>, N: usize) {
    let mut f = File::create(p.clone()).unwrap();
    for i in 0..N {
    }
}

#[allow(non_snake_case)]
pub fn read_xyz(p: std::path::PathBuf, N: usize, use_grads: bool) -> (Vec<f64>, Vec<f64>){
    let mut f = File::open(p.clone()).unwrap();
    let mut buf_reader = BufReader::new(f);
    let mut s = String::new();
    buf_reader.read_to_string(&mut s).unwrap();
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
    let n_elems = if use_grads { N * num_atoms * 6 } else { N * num_atoms * 3 };
    let mut mat = Vec::with_capacity(n_elems);
    let mut grads = vec![];
    if use_grads {
        grads.reserve(3 * num_atoms * N);
    }
    let mut energies = Vec::with_capacity(N * num_atoms);

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
