import jax
import jax.numpy as jnp
from pip_utils import *
from training_utils import *

from training_utils import get_pip_grad, get_forces, split_train_and_test_data_w_forces
from polynomials_ddethanol_p2 import f_polynomials as f_poly
from monomials_ddethanol_p2 import f_monomials as f_mono
from load_data_ddethanol import read_geometry_energy as load_data


def main():
    geometries, forces, energies, atoms = load_data()

    xyz = geometries[0]

    # models in flax
    # PIP
    model_pip = PIP(f_mono, f_poly,)
    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)

    params_pip = model_pip.init(key, xyz[jnp.newaxis])
    print(params_pip)

    # energy
    model_energy = PIPLinear(f_mono, f_poly,)
    params = model_energy.init(key, lax.expand_dims(xyz, dimensions=(0,)))

    # data
    n_tr = 1000
    _, key = jax.random.split(key)
    (X_tr, G_tr, y_tr), (X_tst, G_tst, y_tst) = split_train_and_test_data_w_forces(geometries, forces, energies,
                                                                                   n_tr, key)
    print(X_tr.shape, G_tr.shape, y_tr.shape)

    # training with only energy
    Pip_tr = model_pip.apply(params_pip, X_tr)
    results = jnp.linalg.lstsq(Pip_tr, y_tr)
    theta = results[0]
    params = model_energy.init(key, lax.expand_dims(xyz, dimensions=(0,)))
    params['params']['Dense_0']['kernel'] = theta
    # print(params)

    # print(X_tr.shape, G_tr.shape)
    # X_temp = X_tr[:2]
    # G_temp = G_tr[:2]
    # y_temp = y_tr[:2]
    # print(X_temp.shape)
    # print(G_temp)
    # print(G_temp.reshape(2, 27))
    # print(model_energy.apply(params, X_temp).shape)

    # y_temp_pred = model_energy.apply(params, X_temp)
    # print(y_temp - y_temp_pred)
    # G_temp_pred = get_forces(model_energy.apply, X_temp, params)
    # print(G_temp_pred.shape)
    # print(G_temp - G_temp_pred)

    # NEW
    GPIP_tr = get_pip_grad(model_pip.apply, X_tr, params_pip)
    GPIP_new_tr = GPIP_tr.reshape(n_tr*9*3, 208)
    Pip_tr_new = lax.concatenate((Pip_tr, GPIP_new_tr), dimension=0)
    print(Pip_tr.shape, GPIP_tr.shape, Pip_tr_new.shape)
    y_tr_new = lax.concatenate((y_tr, G_tr.reshape(n_tr*9*3, 1)), dimension=0)
    print(Pip_tr_new.shape, y_tr_new.shape)
    results_w_grad = jnp.linalg.lstsq(Pip_tr_new, y_tr_new)
    params_w_grad = model_energy.init(
        key, lax.expand_dims(xyz, dimensions=(0,)))
    params_w_grad['params']['Dense_0']['kernel'] = results_w_grad[0]
    # print(results_w_grad)
    # print(jax.tree_util.tree_map(lambda x, y: x-y, params, params_w_gad))
    print(results_w_grad[0] - results[0])

    y_pred = model_energy.apply(params, X_tr)
    y_pred_w_grad = model_energy.apply(params_w_grad, X_tr)

    import matplotlib
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.scatter(y_pred, y_tr)
    plt.scatter(y_pred_w_grad, y_tr)
    plt.show()

    # # print(Pip_tr.shape)
    # key = jax.random.PRNGKey(0)
    # _, key = jax.random.split(key)
    # x = jax.random.normal(key, (2, 3, 4))
    # print(x)
    # print(lax.reshape(x, new_sizes=(2*3, 4)))

    # to do
    # check the gradient error jan/13/2024


if __name__ == '__main__':
    main()
