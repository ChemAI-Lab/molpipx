use crate::{N_XYZ, N_DISTANCES};
use pyo3::prelude::*;
use super::*;

//use pyo3::prelude::*;
//#[cfg(pyo3)]
#[pymodule]
fn piprx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_f_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_df_energy, m)?)?;
    m.add_function(wrap_pyfunction!(get_N_XYZ, m)?)?;
    m.add_function(wrap_pyfunction!(get_N_DISTANCES, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_geometry_energy, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn py_read_geometry_energy(filename: std::path::PathBuf) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>) {
    read_geometry_energy(filename)
}


#[pyfunction]
fn py_df_energy(inputs: [f64; N_XYZ], weights: [f64; N_POLYS], mut res: f64) -> ([f64; N_XYZ], f64) {
    let dres: &mut f64 = &mut 0.0;
    let dinputs: &mut [f64; N_XYZ] = &mut [0.0; N_XYZ];
    d_energy(&inputs, dinputs, &weights, &mut res, dres);
    (*dinputs, *dres)
}

#[pyfunction]
fn py_f_energy(inputs: Vec<f64>, weights: Vec<f64>) -> f64 {
    let mut res = 0.0;
    assert!(inputs.len() == N_XYZ);
    assert!(weights.len() == N_POLYS);
    let mut inputs = inputs[0..N_XYZ].try_into().unwrap();
    let mut weights = weights[0..N_POLYS].try_into().unwrap();
    f_energy_inplace(&inputs, &weights, &mut res);
    res
}
#[pyfunction]
fn get_N_XYZ() -> usize {
    N_XYZ
}
#[pyfunction]
fn get_N_DISTANCES() -> usize {
    N_DISTANCES
}
