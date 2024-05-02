from typing import Tuple, List, Callable, Any
import operator

from jaxopt import linear_solve
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import lax
from optax.tree_utils import tree_l2_norm as l2_norm
from jax import tree_util as jtu

from pipjax import split_train_and_test_data_w_forces
from pipjax import LayerPIPAniso, EnergyPIPAniso, get_mask
from pipjax import flax_params_aniso, flax_params
from pipjax import mse_loss, softplus_inverse
from pipjax import get_functions, get_pip_grad
from pipjax import detect_molecule

from jaxtyping import (
    Array,
    Float,
    PyTree,
    install_import_hook,
)

import flax
from flax import linen as nn
import jaxopt
import optax

from jax import config
config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def lambada_random_init(params_pip: Any, key: Any):
    w_l = params_pip['params']['VmapJitPIPAniso_0']['lambda']

    _, key = jrnd.split(key)
    w_rnd = jrnd.uniform(key, shape=(w_l.shape), minval=0.3, maxval=2.1)
    params_pip['params']['VmapJitPIPAniso_0']['lambda'] = w_rnd
    return params_pip


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def _train_adam(molecule_type: str, poly_degree: int, data: Tuple, atoms: List, optimizer_info: PyTree):

    opt_tol = optimizer_info['tol']
    n_epochs = optimizer_info['Maxiters']
    lr = optimizer_info['learning_rate']
    if 'i0' in optimizer_info.keys():
        i0 = optimizer_info['i0']
    else:
        i0 = 0

    rng = jax.random.PRNGKey(i0)
    _, key = jax.random.split(rng)

    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = data
    n, na, _ = X_tr.shape

    molecule_type = molecule_type
    f_mono, f_poly = get_functions(molecule_type, poly_degree)

    mask, unique_pairs = get_mask(atoms)
    n_pairs = mask.shape[0]

    # initialize models
    model_pip = LayerPIPAniso(f_mono, f_poly, n_pairs)
    params_pip = model_pip.init(key, X_tr[:1], mask)
    params_pip = lambada_random_init(params_pip, key)

    x0_pip = model_pip.apply(params_pip, X_tr[:1], mask)
    n_pip = x0_pip.shape[1]

    model_energy = EnergyPIPAniso(f_mono, f_poly, n_pairs)
    params_energy = model_energy.init(key, X_tr[:1], mask)

    def f_pip(params, x): return model_pip.apply(params, x, mask)
    def f_pip_energy(params, x): return model_energy.apply(params, x, mask)

    @jax.jit
    def validation_loss(params_pip, data_, params_energy):
        (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = data_
        # params_pip, params_energy = params_

        # params_pip = flax_params_aniso(l, params_pip)

        @jax.jit
        def inner_loss(params_pip):
            Pip_tr = f_pip(params_pip, X_tr)
            Gpip_tr = get_pip_grad(f_pip, X_tr, params_pip)
            # (bs*n_atoms*3,number of pip)
            Gpip_tr = Gpip_tr.reshape(n*na*3, n_pip)
            Pip_tr_w_grad_full = lax.concatenate(
                (Pip_tr, Gpip_tr), dimension=0)

            y_tr_w_grad_full = lax.concatenate(
                (y_tr, F_tr.reshape(n*na*3, 1)), dimension=0)

            # Optimization
            results_w_grad = jnp.linalg.lstsq(
                Pip_tr_w_grad_full, y_tr_w_grad_full)
            theta = results_w_grad[0]
            return theta

        w_opt = inner_loss(params_pip)
        # print(w_opt)
        params_energy = flax_params(w_opt, params_energy)
        y_val_pred = f_pip_energy(params_energy, X_val)

        loss_val = mse_loss(y_val_pred, y_val)
        return loss_val, (w_opt, mse_loss(f_pip_energy(params_energy, X_tr), y_tr))

    params_all = (params_pip, params_energy)

    # optax loop
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_pip)

    grad_fn = jax.jit(jax.value_and_grad(
        validation_loss, argnums=(0,), has_aux=True))

    @jax.jit
    def train_step(li, optimizer_state):
        (loss, (_, loss_tr)), grads = grad_fn(li, data, params_energy)
        updates, opt_state = optimizer.update(
            grads[0], optimizer_state, li)
        return optax.apply_updates(li, updates), opt_state, loss, loss_tr

    l_params = params_pip  # l_init
    l_ = []
    for i in range(n_epochs):
        l0_params = l_params
        l_params, opt_state, loss_val_i, loss_tr_i = train_step(
            l_params, opt_state)
        li = l_params['params']['VmapJitPIPAniso_0']['lambda']
        l_.append(li)
        print(i, loss_val_i, li, nn.softplus(li))
        if l2_norm(jtu.tree_map(lambda x, y: x + y, l_params, l0_params)) < opt_tol:
            break
    l_opt_trajectory = jnp.array(l_)

    return l_opt_trajectory[-1], l_opt_trajectory


def plotting(molecule_type, poly_degree, atoms, data):

    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = data
    n, na, _ = X_tr.shape

    f_mono, f_poly = get_functions(molecule_type, poly_degree)

    mask, unique_pairs = get_mask(atoms)
    n_pairs = mask.shape[0]

    # initialize models
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    model_pip = LayerPIPAniso(f_mono, f_poly, n_pairs)
    params_pip = model_pip.init(key, X_tr[:1], mask)
    x0_pip = model_pip.apply(params_pip, X_tr[:1], mask)
    n_pip = x0_pip.shape[1]

    model_energy = EnergyPIPAniso(f_mono, f_poly, n_pairs)
    params_energy = model_energy.init(key, X_tr[:1], mask)

    params_all = (params_pip, params_energy)

    def f_pip(params, x): return model_pip.apply(params, x, mask)
    def f_pip_energy(params, x): return model_energy.apply(params, x, mask)

    @jax.jit
    def validation_loss(l_pip, data_, params_):
        (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = data_
        params_pip, params_energy = params_

        params_pip = flax_params_aniso(l_pip, params_pip)

        @jax.jit
        def inner_loss(params_pip):
            Pip_tr = f_pip(params_pip, X_tr)
            Gpip_tr = get_pip_grad(f_pip, X_tr, params_pip)
            # (bs*n_atoms*3,number of pip)
            Gpip_tr = Gpip_tr.reshape(n*na*3, n_pip)
            Pip_tr_w_grad_full = lax.concatenate(
                (Pip_tr, Gpip_tr), dimension=0)

            y_tr_w_grad_full = lax.concatenate(
                (y_tr, F_tr.reshape(n*na*3, 1)), dimension=0)

            # Optimization
            results_w_grad = jnp.linalg.lstsq(
                Pip_tr_w_grad_full, y_tr_w_grad_full)
            theta = results_w_grad[0]
            return theta

        w_opt = inner_loss(params_pip)
        # print(w_opt)
        params_energy = flax_params(w_opt, params_energy)
        y_val_pred = f_pip_energy(params_energy, X_val)

        loss_val = mse_loss(y_val_pred, y_val)
        return loss_val, (w_opt, mse_loss(f_pip_energy(params_energy, X_tr), y_tr))

    # plotting
    l_grid = jnp.linspace(0.05, 2., 50)
    l_grid_sftplus = nn.softplus(l_grid)

    l0, l1 = jnp.meshgrid(l_grid, l_grid)
    l0_sftplus, l1_sftplus = jnp.meshgrid(l_grid_sftplus, l_grid_sftplus)
    l01 = jnp.column_stack((l0.flatten(), l1.flatten()))
    loss_val_grid, (_, loss_tr_grid) = jax.vmap(
        validation_loss, in_axes=(0, None, None))(l01, data, params_all)

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm, ticker

    fig, ax = plt.subplots()
    cs = ax.contourf(l0_sftplus, l1_sftplus, jnp.reshape(
        loss_val_grid, l0.shape), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'${\cal L}_{\text{val}}$', fontsize=22)

    opt_info = {'tol': 1e-4, 'Maxiters': 1000, 'learning_rate': 2e-2}
    for i in range(10):
        opt_info.update({'i0': i})
        l_opt, l_opt_trj = _train_adam(molecule_type, poly_degree,
                                       data, atoms,  opt_info)
        l_opt_sftplus = nn.softplus(l_opt_trj)
        ax.scatter(l_opt_sftplus[:, 0], l_opt_sftplus[:, 1],
                   marker='*', color='k', s=5)

    # plt.text(0.875, 1.85, 'Adam', fontsize=16)
    plt.text(0.875, 1.85, 'd)', fontsize=18)
    plt.xlim(0.8, 2.0)
    plt.ylim(0.8, 2.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\lambda_{\text{HH}}$', fontsize=20)
    plt.ylabel(r'$\lambda_{\text{CH}}$', fontsize=20)
    plt.tight_layout()

    path = '/scratch/r/ravh011/ravh011/pipjax_test'
    fig_file = f'loss_val_wforces_grid_Adam_n{n}.png'
    plt.savefig(f'{path}/{fig_file}', dpi=600)

    # return l_opt[-1], l_opt


def main():

    from load_data_methane import read_geometry_energy as load_data

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    n_tr = 250
    n_val = 250

    X_all, F_all, y_all, atoms_all = load_data()
    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
        X_all, F_all, y_all, n_tr, key, n_val)

    atoms = atoms_all[0]
    data = (X_tr, F_tr, y_tr), (X_val, F_val, y_val)

    molecule_type = 'A4B'
    poly_degree = 3
    opt_info = {'tol': 1e-4, 'Maxiters': 500, 'learning_rate': 2e-2}

    # l_opt, l_opt_trajectory = _train_adam(
    # molecule_type, poly_degree, data, atoms,  opt_info)

    plotting(molecule_type, poly_degree, atoms, data)  # clean


if __name__ == '__main__':
    main()
