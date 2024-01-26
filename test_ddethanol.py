import jax
import jax.numpy as jnp
from pip_utils import *
from training_utils import *

from training_utils import get_pip_grad, get_forces, split_train_and_test_data_w_forces
from polynomials_ddethanol_p2 import f_polynomials as f_poly
from monomials_ddethanol_p2 import f_monomials as f_mono
from load_data_ddethanol import read_geometry_energy as load_data


def training(n_tr: int = 2000):
    geometries, forces, energies, atoms = load_data()

    xyz = geometries[0]

    # models in flax
    # PIP
    model_pip = PIP(f_mono, f_poly,)
    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)

    params_pip = model_pip.init(key, xyz[jnp.newaxis])
    # print(params_pip)

    # energy
    model_energy = PIPLinear(f_mono, f_poly,)
    params = model_energy.init(key, lax.expand_dims(xyz, dimensions=(0,)))

    # data
    # n_tr = 2000
    _, key = jax.random.split(key)
    (X_tr, G_tr, y_tr), (X_tst, G_tst, y_tst) = split_train_and_test_data_w_forces(geometries, forces, energies,
                                                                                   n_tr, key)
    # print(X_tr.shape, G_tr.shape, y_tr.shape)

    # training with only energy
    Pip_tr = model_pip.apply(params_pip, X_tr)
    results = jnp.linalg.lstsq(Pip_tr, y_tr)
    theta = results[0]
    params = model_energy.init(key, lax.expand_dims(xyz, dimensions=(0,)))
    params['params']['Dense_0']['kernel'] = theta
    # print(params)

    # NEW
    GPIP_tr = get_pip_grad(model_pip.apply, X_tr, params_pip)
    GPIP_new_tr = GPIP_tr.reshape(n_tr*9*3, 208)
    Pip_tr_new = lax.concatenate((Pip_tr, GPIP_new_tr), dimension=0)
    # print(Pip_tr.shape, GPIP_tr.shape, Pip_tr_new.shape)
    # check the sign of the forces
    y_tr_new = lax.concatenate((y_tr, G_tr.reshape(n_tr*9*3, 1)), dimension=0)
    # print(Pip_tr_new.shape, y_tr_new.shape)
    results_w_grad = jnp.linalg.lstsq(Pip_tr_new, y_tr_new)
    params_w_grad = model_energy.init(
        key, lax.expand_dims(xyz, dimensions=(0,)))
    params_w_grad['params']['Dense_0']['kernel'] = results_w_grad[0]

    y_pred = model_energy.apply(params, X_tr)
    y_pred_w_grad = model_energy.apply(params_w_grad, X_tr)

    y_pred_tst = model_energy.apply(params, X_tst)
    y_pred_tst_w_grad = model_energy.apply(params_w_grad, X_tst)

    return (y_pred, y_pred_w_grad, y_tr), (y_pred_tst, y_pred_tst_w_grad, y_tst)

    # print(jnp.linalg.norm(y_pred-y_tr), jnp.linalg.norm(y_pred_w_grad-y_tr))
    # print(jnp.linalg.norm(y_pred_tst-y_tst),
    #       jnp.linalg.norm(y_pred_tst_w_grad-y_tst))

    # import matplotlib
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.scatter(y_pred, y_tr, label='Energy')
    # ax1.scatter(y_pred_w_grad, y_tr, label='Energy + Forces')
    # ax1.plot(y_tr, y_tr, color='k')

    # ax2.scatter(y_pred_tst, y_tst, label='Energy')
    # ax2.scatter(y_pred_tst_w_grad, y_tst, label='Energy + Forces')
    # ax2.plot(y_tst, y_tst, color='k')

    # plt.legend()
    # plt.show()


def main():
    ni_ = jnp.arange(100, 2000, 100, dtype=jnp.int32)
    for ni in ni_:
        (y_pred, y_pred_w_grad, y_tr), (y_pred_tst,
                                        y_pred_tst_w_grad, y_tst) = training(ni)

        print(ni, ' : ',  jnp.linalg.norm(y_pred-y_tr), jnp.linalg.norm(y_pred_w_grad-y_tr), ' : ',
              jnp.linalg.norm(y_pred_tst-y_tst), jnp.linalg.norm(y_pred_tst_w_grad-y_tst))


if __name__ == '__main__':
    main()
