import argparse
from ml_collections import config_dict
import numpy as onp
import pandas as pd

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax
from optax.tree_utils import tree_l2_norm as l2_norm
from flax import linen as nn

from molpipx import split_train_and_test_data_w_forces
from molpipx import LayerPIPAniso, EnergyPIPAniso, get_mask
from molpipx import flax_params, mse_loss
from molpipx import get_f_mask, lambda_random_init
from molpipx import get_functions, detect_molecule, get_pip_grad

from load_data_methane import read_geometry_energy

jax.config.update("jax_debug_nans", True)


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def train_and_evaluate(config: config_dict.ConfigDict,
                       workdir: str,):

    opt_tol = config.opt_tol
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
    atoms = atoms[0]

    # split training and validation data
    i0 = 0
    rng = jax.random.PRNGKey(i0)
    _, key = jax.random.split(rng)
    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
        X_all, F_all, y_all, n_tr, key, n_val)
    n, na, _ = X_tr.shape
    data = ((X_tr, F_tr, y_tr), (X_val, F_val, y_val))

    mask, unique_pairs = get_mask(atoms)
    n_pairs = mask.shape[0]

    f_mask = get_f_mask(mask)

    # define models
    model_pip = LayerPIPAniso(f_mono, f_poly, f_mask, n_pairs)
    params_pip = model_pip.init(key, X_tr[:1])
    # random initial parameters
    params_pip = lambda_random_init(params_pip, key)
    print('initial random parameters: ', params_pip)

    model_energy = EnergyPIPAniso(f_mono, f_poly, f_mask, n_pairs)
    params_energy = model_energy.init(key, X_tr[:1])

    @jax.jit
    def validation_loss(params_pip, data_, params_energy):
        (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = data_
        # params_pip, params_energy = params_

        # params_pip = flax_params_aniso(l, params_pip)
        @jax.jit
        def inner_loss(params_pip):
            Pip_tr = model_pip.apply(params_pip, X_tr)
            n_pip = Pip_tr.shape[-1]
            Gpip_tr = get_pip_grad(model_pip.apply, X_tr, params_pip)
            # (bs*n_atoms*3,number of pip)
            Gpip_tr = Gpip_tr.reshape(n*na*3, n_pip)
            Pip_tr_w_grad_full = jax.lax.concatenate(
                (Pip_tr, Gpip_tr), dimension=0)

            y_tr_w_grad_full = jax.lax.concatenate(
                (y_tr, F_tr.reshape(n*na*3, 1)), dimension=0)

            # Optimization
            results_w_grad = jnp.linalg.lstsq(
                Pip_tr_w_grad_full, y_tr_w_grad_full)
            theta = results_w_grad[0]
            return theta

        w_opt = inner_loss(params_pip)
        # print(w_opt)
        params_energy = flax_params(w_opt, params_energy)
        y_val_pred = model_energy.apply(params_energy, X_val)

        loss_val = mse_loss(y_val_pred, y_val)
        loss_tr = mse_loss(model_energy.apply(params_energy, X_tr), y_tr)
        return loss_val, (params_energy, loss_tr)

    params_all = (params_pip, params_energy)

    # optax loop
    lr = config.learning_rate
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_pip)

    grad_fn = jax.jit(jax.value_and_grad(
        validation_loss, argnums=(0,), has_aux=True))

    @ jax.jit
    def train_step(li, optimizer_state):
        (loss, (params_e, loss_tr)), grads = grad_fn(li, data, params_energy)
        updates, opt_state = optimizer.update(
            grads[0], optimizer_state, li)
        return optax.apply_updates(li, updates), opt_state, loss, loss_tr, params_e

    l_params = params_pip  # l_init
    l_ = []
    df = pd.DataFrame()
    for epoch in range(1, config.num_epochs + 1):
        l0_params = l_params
        l_params, opt_state, loss_val_i, loss_tr_i, params_e = train_step(
            l_params, opt_state)
        l0_epoch = l_params['params']['VmapJitPIPAniso_0']['lambda']
        l_.append(l0_epoch)

        # print
        print('epoch:% 3d, train_loss: %.6f, val_loss: %.6f'
              % (
                  epoch,
                  loss_tr_i,
                  loss_val_i,
              ), nn.softplus(l0_epoch))

        # save to file
        l_names = ['l_' + str(i)
                   for i in range(len(onp.array(l0_epoch)))]
        r_epoch = {'epoch': epoch,
                   'tr_loss': loss_tr_i,
                   'val_loss': loss_val_i,
                   **dict(zip(l_names, onp.array(l0_epoch)))
                   }
        df = pd.concat(
            [df, pd.DataFrame(r_epoch, index=[0])], ignore_index=True)
        df.to_csv(
            f"{workdir}/training_w_grad_trajectory_and_l_params.csv", index=False)

        if l2_norm(jtu.tree_map(lambda x, y: x - y, l_params, l0_params)) < opt_tol:
            break

    return params_e
