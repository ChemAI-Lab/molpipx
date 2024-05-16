import argparse
from ml_collections import config_dict

import jax
import jax.numpy as jnp

from pipx import EnergyPIP, PIPlayer as PIP
from pipx import training, training_w_gradients, get_forces
from pipx import split_train_and_test_data, split_train_and_test_data_w_forces
from pipx import flax_params
from pipx import mse_loss
from pipx import get_functions, detect_molecule

from load_data_methane import read_geometry_energy


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def train_and_evaluate(config: config_dict.ConfigDict,
                       workdir: str):

    n_tr = config.ntr
    n_val = config.nval
    molecule_type = config.molecule
    poly_degree = config.poly_degree
    # get monomials and polynomials functions
    mol_dict, mol_sym = detect_molecule(molecule_type)
    na = get_number_of_atoms(mol_dict)
    f_mono, f_poly = get_functions(molecule_type, poly_degree)

    # load all CH4 data
    X_all, _, y_all, atoms = read_geometry_energy()

    # split training and validation data
    i0 = 0
    rng = jax.random.PRNGKey(i0)
    _, key = jax.random.split(rng)
    (X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
        X_all, y_all, n_tr, key, n_val)

    # load PIP-Flax model
    model_pip = PIP(f_mono, f_poly)
    model_energy = EnergyPIP(f_mono, f_poly)
    params = model_energy.init(key, X_tr[:1])

    # training
    w = training(model_pip, X_tr, y_tr)  # jnp.array
    params = flax_params(w, params)  # transform to Pytree

    # prediction
    y_pred_tr = model_energy.apply(params, X_tr)
    y_pred_val = model_energy.apply(params, X_val)
    print(y_pred_tr.shape, y_pred_val.shape)

    loss_tr = mse_loss(y_pred_tr, y_tr)
    loss_val = mse_loss(y_pred_val, y_val)

    print(n_tr, loss_tr, loss_val)
    results = {'X_tr': X_tr, 'y_tr': y_tr, 'y_pred_tr': y_pred_tr,
               'X_val': X_val, 'y_val': y_val, 'y_pred_val': y_pred_val}

    results_file = f'{workdir}/results_N_{n_tr}_deg_{poly_degree}.npy'

    jnp.save(results_file, results, allow_pickle=True)


def train_and_evaluate_w_gradients(config: config_dict.ConfigDict,
                                   workdir: str):

    n_tr = config.ntr
    n_val = config.nval
    molecule_type = config.molecule
    poly_degree = config.poly_degree
    # get monomials and polynomials functions
    mol_dict, mol_sym = detect_molecule(molecule_type)
    na = get_number_of_atoms(mol_dict)
    f_mono, f_poly = get_functions(molecule_type, poly_degree)

    # load all CH4 data
    X_all, F_all, y_all, atoms = read_geometry_energy()

    # split training and validation data
    i0 = 0
    rng = jax.random.PRNGKey(i0)
    _, key = jax.random.split(rng)
    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
        X_all, F_all, y_all, n_tr, key, n_val)

    # load PIP-Flax model
    model_pip = PIP(f_mono, f_poly)
    model_energy = EnergyPIP(f_mono, f_poly)
    params = model_energy.init(key, X_tr[:1])

    # training
    w = training_w_gradients(model_pip, X_tr, F_tr, y_tr)
    params = flax_params(w, params)

    # prediction Energy
    y_pred_tr = model_energy.apply(params, X_tr)
    y_pred_val = model_energy.apply(params, X_val)
    print(y_pred_tr.shape, y_pred_val.shape)

    # prediction Forces
    F_pred_tr = get_forces(model_energy.apply, X_tr, params)
    F_pred_val = get_forces(model_energy.apply, X_val, params)

    loss_energy_tr = mse_loss(y_pred_tr, y_tr)
    loss_energy_val = mse_loss(y_pred_val, y_val)

    # loss_force_tr = mse_loss(F_pred_tr, F_tr)
    # loss_force_val = mse_loss(F_pred_val, F_val)

    loss_force_tr = jnp.linalg.norm(F_pred_tr - F_tr)
    loss_force_val = jnp.linalg.norm(F_pred_val - F_val)
    print(n_tr, loss_energy_tr, loss_energy_val, loss_force_tr, loss_force_val)

    results = {'X_tr': X_tr, 'F_tr': F_tr, 'y_tr': y_tr, 'y_pred_tr': y_pred_tr, 'F_pred_tr': F_pred_tr,
               'X_val': X_val, 'y_val': y_val, 'y_pred_val': y_pred_val, 'y_pred_val': y_pred_val, 'F_pred_val': F_pred_val}

    results_file = f'{workdir}/results_N_{n_tr}_deg_{poly_degree}_wgrad.npy'
    jnp.save(results_file, results, allow_pickle=True)
