#![allow(non_snake_case)]
#![allow(dead_code)]
#![feature(autodiff)]

mod data;
pub use crate::data::molecule_A2B::{monomials_2_1_3::*, polynomials_2_1_3::*};

//mod methane;
//pub use crate::methane::polynomials_deg_3::*;
//pub use crate::methane::monomials_deg_3::*;

use std::io::Read;

#[cfg(pyo3)]
pub mod python;

// TODO: Here you see some example lines of an .xyz file that we are going to parse.
// Again, we have hardcoded a specific .xyz file, but you are free to adjust it with your own.
//
//
//    5
//   -40.48322077
//H     -0.1827861      0.5603340     -0.9332645     -0.0054580      0.0059540     -0.0036580
//H      0.3599180     -1.0070156     -0.1993307     -0.0039780      0.0039510      0.0040370
//H      0.7177714      0.5393594      0.6086020     -0.0050170     -0.0036640     -0.0058060
//H     -0.9549494     -0.0777535      0.5049601      0.0035980     -0.0002840     -0.0069600
//C      0.0050428     -0.0012536      0.0015986      0.0108530     -0.0059460      0.0123880

const ANG_TO_BOHR: f64 = 0.529177210929;
const BHOR_TO_ANG: f64 = ANG_TO_BOHR * 1.0e-1;


pub fn read_geometry_energy(filename: std::path::PathBuf) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>) {
    let file = std::fs::File::open(filename.clone()).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let mut atoms: Vec<String> = Vec::new();
    
    let mut s = String::new();
    reader.read_to_string(&mut s).unwrap();
    let num_lines = s.lines().clone().count();
    let mut lines = s.lines();

    // First, we need to figure out how many rows we need.
    let num_atoms = lines.next().unwrap().trim_start().parse::<usize>().unwrap();
    lines.next().unwrap(); // skip energy
    let elems_in_line = lines.next().unwrap().split_whitespace().count();
    let use_grads = elems_in_line == 7;
    if !use_grads {
        assert!(elems_in_line == 4);
    }
    assert!(num_lines % (num_atoms + 2) == 0);
    let num_examples: usize = num_lines / (num_atoms + 2); // N \leq num_lines
    let N = num_examples; // TODO: enable smaller values
    assert!(N <= num_examples, "The file: {}\n only contains {} entries, but you requested {}!", filename.display()
            , num_examples, N);
    let n_elems = if use_grads { N * num_atoms * 6 } else { N * num_atoms * 3 };
    let mut geometries = Vec::with_capacity(n_elems);
    let mut forces = vec![];
    if use_grads {
        forces.reserve(3 * num_atoms * N);
    }
    let mut energies: Vec<f64> = Vec::with_capacity(N * num_atoms);

    // Phew, finally done. Now let's actually read the file.
    lines = s.lines();
    for _ in 0..N {
        let n_atoms = lines.next().unwrap().trim_start().parse::<usize>().unwrap();
        assert!(n_atoms == num_atoms);

        let mut line = lines.next().unwrap().split_whitespace();
        let energy = line.next().unwrap().trim_start().parse::<f64>().unwrap();
        energies.push(energy);
        
        //        atom, x, y, z, dx, dy, dz = parts

        for _ in 0..num_atoms {
            let mut line = lines.next().unwrap().split_whitespace();

            let atom = line.next().unwrap().parse::<String>().unwrap();
            atoms.push(atom);
            
            for _ in 0..3 {
                let val = line.next().unwrap().parse::<f64>().unwrap();
                geometries.push(val);
            }

            if use_grads {
               for _ in 0..3 {
                   let val = line.next().unwrap().parse::<f64>().unwrap();
                   forces.push(val / ANG_TO_BOHR);
               }
            }
        }
    }
    //if use_grads {
    //    geometries.append(&mut forces);
    //}
    
    //if energy_normalization {
    //    let (e_min, e_max) = (energies.min(), energies.max());
    //    energies = (energies - e_min) / (e_max - e_min)
    //}
    (geometries, forces, energies, atoms)
    //    return jnp.array(geometries), jnp.array(forces), jnp.array(energies[:, np.newaxis]), np.array(atoms)
}

#[inline(always)]
fn point_dist(positions: &[f64; N_XYZ], x: usize, y: usize) -> f64 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

#[inline(always)]
pub fn dist(positions: &[f64; N_XYZ]) -> [f64; N_DISTANCES] {
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

pub fn morse(r: &mut [f64], l: f64) {
    for x in r {
        // x = e^(-x / l)
        *x = f64::exp(-*x / l);
    }
    //r = r.iter().map(|x| f64::exp(-x / l)).collect();
}

//#[autodiff(d_energy_rev, Reverse, Duplicated, Duplicated, Active)]
//fn f_energy(inputs: &[f64; N_R], weights: &[f64; N_POLYS]) -> f64{
//    let mut res = 0.0;
//    f_energy_inplace(inputs, weights, &mut res);
//    res
//}
//

#[no_mangle]
pub extern "C" fn c_d_energy(inputs: *const [f64; N_XYZ], d_inputs: *mut [f64; N_XYZ], weights: *const [f64; N_POLYS], res: *mut f64, d_res: *mut f64) {
    unsafe {
        let inputs = &*inputs;
        let d_inputs = &mut *d_inputs;
        let weights = &*weights;
        let res = &mut *res;
        let d_res = &mut *d_res;
        d_energy(inputs, d_inputs, weights, res, d_res);
    }
}

#[autodiff(d_energy, Reverse, Duplicated, Const, Duplicated)]
pub fn f_energy_inplace(inputs: &[f64; N_XYZ], weights: &[f64; N_POLYS], res: &mut f64) {
    let mut distances = dist(inputs);
    //let mut distances = dist::<N_XYZ>(inputs);
    morse(&mut distances, 1.0);
    let outs = f_polynomials(&distances);
    assert!(outs.len() == weights.len());

    for i in 0..N_POLYS {
        *res += weights[i] * outs[i];
    }
}

//#[test]
//fn test_energy() {
//    let inputs = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//    let weights = [1.0; N_POLYS];
//    let mut res = 0.0;
//    f_energy_inplace(&inputs, &weights, &mut res);
//    assert_eq!(res, 1.0 * N_POLYS as f64);
//}

#[test]
fn test_denergy() {
    let inputs = [0.0; N_XYZ];
    let mut d_inputs = [0.0; N_XYZ];
    let weights = [1.0; N_POLYS];
    let mut res = 0.0;
    let mut d_res = 0.0;
    d_energy(&inputs, &mut d_inputs, &weights, &mut res, &mut d_res);
    assert_eq!(res, 1.0 * N_POLYS as f64);
    assert_eq!(d_res, 0.0);
    assert_eq!(d_inputs, [1.0; N_XYZ]);
}

//
//#[autodiff(f_energy_inplace, Reverse, Const, Duplicated, Const, Duplicated)]
//fn d_energy_inplace_rev(inputs: &[f64; N_R], d_inputs: &mut[f64; N_R], weights: &[f64; N_POLYS], res: &mut f64, res_t: & f64);
//
//#[autodiff(d_energy_fwd, Forward, Dual, Const, Dual)]
//fn f_energy2(inputs: &[f64; N_R], weights: &[f64; N_POLYS]) -> f64 {
//    f_energy(inputs, weights)
//}

