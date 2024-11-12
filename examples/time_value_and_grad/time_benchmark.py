import os
import pandas as pd
import argparse
import time
import timeit

import jax
import jax.numpy as jnp
import jax.random as jrnd

from molpipx import PIPlayer as PIP
from molpipx import EnergyPIP
from molpipx import detect_molecule, get_functions, get_energy_and_forces


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def compute_gradients(molecule_type: str, poly_degree: int, batch_size: int):
    mol_dict, mol_sym = detect_molecule(molecule_type)
    print(molecule_type, mol_dict, mol_sym)
    na = get_number_of_atoms(mol_dict)

    f_mono, f_poly = get_functions(molecule_type, poly_degree)
    x = jnp.ones(int((na**2 - na)/2))
    d_mono = f_mono(x).shape[-1]

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    # random geometries to test the energy and forces
    geoms = jrnd.normal(key, (batch_size, na, 3))
    geom0 = geoms[:1]

    # load PIP-Flax model
    model_pip = PIP(f_mono, f_poly)
    model_energy = EnergyPIP(f_mono, f_poly)
    params = model_energy.init(key, geom0)
    params_pip = model_pip.init(key, geom0)
    z_pip0 = model_pip.apply(params_pip, geom0)
    d_pip = z_pip0.shape[-1]

    # timeit after jitting
    def my_function():
        e, f = get_energy_and_forces(
            model_energy.apply, geoms, params)
        # e = e.block_until_ready()
        # f = f.block_until_ready()
        return {'e': e, 'f': f}

    # pre-jitting
    n = 1
    timing_result = timeit.timeit(
        lambda: jax.block_until_ready(jax.jit(my_function)()),
        number=n
    )
    warmup_time = timing_result / n
    print("Time warm up: ", timing_result)

    # timeit after jitting
    n = 1
    timing_result = timeit.timeit(
        lambda: jax.block_until_ready(jax.jit(my_function)()),
        number=n
    )
    exec_time = timing_result / n
    print("Time jit exec: ", timing_result)

    r = {'mol': molecule_type,
         'pip': d_pip,
         'mono': d_mono,
         'poly degree': poly_degree,
         'bs': batch_size,
         'time wup': warmup_time,
         'time val and grad': exec_time,
         }
    return r


def list_of_strings(arg):
    return arg.split(',')


def main():
    parser = argparse.ArgumentParser(description="pipjax gradient test")
    parser.add_argument("--bs", type=int, default=1,)

    args = parser.parse_args()
    bs = args.bs  # 1000

    mols_ = ['A3', 'A2B', 'ABC',
             'A4', 'A3B', 'A2B2', 'A2BC', 'ABCD',
             'A5', 'A4B', 'A3B2', 'A3BC', 'A2B2C', 'A2BCD', 'ABCDE']
    poly_ = [3, 4, 5, 6, 7, 8]  # , 7

    path = '/u/rvargas/PIPMSA_jax/examples/value_and_grad_test'
    file = f'value_and_gradient_time_jax_jit.csv'

    def get_pd_frame():
        if os.path.isfile(f"{path}/{file}"):
            results = pd.read_csv(
                f"{path}/{file}")
        else:
            results = pd.DataFrame()
        return results

    for mi in mols_:
        for p in poly_[:1]:
            # for p in range(8, 9):
            ri = compute_gradients(mi, p, bs)
            print(ri)

            results = get_pd_frame()
            results = pd.concat(
                [results, pd.DataFrame(ri, index=[0])], ignore_index=True)
            results.to_csv(
                f"{path}/{file}", index=False)


if __name__ == "__main__":
    main()
