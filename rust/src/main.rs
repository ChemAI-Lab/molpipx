use piprx::*;
use divan::Bencher;
use piprx::{N_XYZ, N_POLYS};

//fn main() {
//    let inputs = [0.0; N_XYZ];
//    let mut d_inputs = [0.0; N_XYZ];
//    let weights = [1.0; N_POLYS];
//    let mut res = 0.0;
//    let mut d_res = 0.0;
//    d_energy(&inputs, &mut d_inputs, &weights, &mut res, &mut d_res);
//    assert_eq!(res, 1.0 * N_POLYS as f64);
//    assert_eq!(d_res, 0.0);
//    assert_eq!(d_inputs, [1.0; N_XYZ]);
//}
fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench]
fn grad(bencher: Bencher) {
    let inputs = [0.0; N_XYZ];
    let mut d_inputs = [0.0; N_XYZ];
    let weights = [1.0; N_POLYS];
    let mut res = 0.0;
    let mut d_res = 0.0;

    bencher.bench_local(move || {
        d_energy(&inputs, &mut d_inputs, &weights, &mut res, &mut d_res);
    });
}

#[divan::bench]
fn primal(bencher: Bencher) {
    let inputs = [0.0; N_XYZ];
    let weights = [1.0; N_POLYS];
    let mut res = 0.0;

    bencher.bench_local(move || {
        f_energy_inplace(&inputs, &weights, &mut res);
    });
}
