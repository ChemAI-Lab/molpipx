#![feature(autodiff)]
#![feature(array_chunks)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]

use divan::{Bencher, black_box};

#[divan::bench]
fn add() -> i32 {
    black_box(1) + black_box(42)
}

use std::autodiff::autodiff;
use piprx::*;

use std::path::PathBuf;
use piprx::read_geometry_energy;

fn main() {
  divan::main();
}

#[autodiff(df_energy, Reverse, Duplicated, Const, Active)]
fn f_energy(res: &mut [f64; N_DISTANCES], w: &[f64; N_POLYS]) -> f64 {
    assert_ne!(*res, [0.0; N_DISTANCES]);
    morse(res, 1.0);
    let pip: [f64; N_POLYS] = f_polynomials(&res);
    let mut y = 0.0;
    for i in 0..N_POLYS {
        y += pip[i] * w[i];
    }
    assert_ne!(y, 0.0);
    y
}

#[divan::bench]
fn divan_inference(bencher: Bencher) {
    let mut path_to_xyz = PathBuf::new();
    path_to_xyz.push("/home/drehwald/prog/RMSA/Data/Methane/Methane.xyz");
    let (geometries, forces, energies, atoms) = read_geometry_energy(path_to_xyz);
    
    // split geometries and forces in half.
    let half = geometries.len() / 2;
    let (X_tr, X_val) = geometries.split_at(half);
    let half_e = energies.len() / 2;
    let (y_tr, y_val) = energies.split_at(half_e);
    let half_f = forces.len() / 2;
    let (F_tr, F_val) = forces.split_at(half_f);
   
    let mut res = 0.0;
    bencher.bench_local(move || {
        for x in X_tr.array_chunks::<N_XYZ>() {
            assert_ne!(*x, [0.0; N_XYZ]);
            let mut x = dist(&x);
            let w = [1.0; N_POLYS];
            res += f_energy(&mut x, &w);
        }
    });
}
#[divan::bench]
fn divan_training_verify(bencher: Bencher){
    let mut path_to_xyz = PathBuf::new();
    path_to_xyz.push("/home/drehwald/prog/RMSA/Data/Methane/Methane.xyz");
    let (geometries, forces, energies, atoms) = read_geometry_energy(path_to_xyz);
    
    // split geometries and forces in half.
    let half = geometries.len() / 2;
    let (X_tr, X_val) = geometries.split_at(half);
    let half_e = energies.len() / 2;
    let (y_tr, y_val) = energies.split_at(half_e);
    let half_f = forces.len() / 2;
    let (F_tr, F_val) = forces.split_at(half_f);
    
    let mut dx3 = [0.0; N_DISTANCES];
    bencher.bench_local(move || {
        for x in X_tr.array_chunks::<N_XYZ>() {
            assert_ne!(*x, [0.0; N_XYZ]);
            //let x = [1.0, 12.0, 3.0, 44.0, 5.0, 6.0, 7.3, 8.0, 9.0];
            let mut x = dist(&x);
            let w = [1.0; N_POLYS];
            
            df_energy(&mut x, &mut dx3, &w, 1.0);
        }
    });
}

#[divan::bench]
fn divan_training(bencher: Bencher){
    let mut path_to_xyz = PathBuf::new();
    path_to_xyz.push("/home/drehwald/prog/RMSA/Data/Methane/Methane.xyz");
    let (geometries, forces, energies, atoms) = read_geometry_energy(path_to_xyz);
    
    // split geometries and forces in half.
    let half = geometries.len() / 2;
    let (X_tr, X_val) = geometries.split_at(half);
    let half_e = energies.len() / 2;
    let (y_tr, y_val) = energies.split_at(half_e);
    let half_f = forces.len() / 2;
    let (F_tr, F_val) = forces.split_at(half_f);
    
    let mut dx3 = [0.0; N_DISTANCES];
    bencher.bench_local(move || {
        for x in X_tr.array_chunks::<N_XYZ>() {
            assert_ne!(*x, [0.0; N_XYZ]);
            //let x = [1.0, 12.0, 3.0, 44.0, 5.0, 6.0, 7.3, 8.0, 9.0];
            let mut x = dist(&x);
            let w = [1.0; N_POLYS];
            
            df_energy(&mut x, &mut dx3, &w, 1.0);
            assert_ne!(dx3, [0.0; N_DISTANCES]);
        }
    });
}
