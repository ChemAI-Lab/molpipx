import argparse

import jax
import jax.numpy as jnp

import pipjax
from pipjax import training, training_w_gradients
from pipjax import split_train_and_test_data
from pipjax import PIPlayer as PIP
from pipjax import EnergyPIP
from pipjax import flax_params
from pipjax import mse_loss, mae_loss

from load_data_methane import read_geometry_energy


def get_pip_functions(deg: int):
    if deg == 3:
        from monomials_deg_3 import f_monomials
        from polynomials_deg_3 import f_polynomials

        def f_mono(x): return f_monomials(x)
        def f_poly(x): return f_polynomials(x)
    elif deg == 4:
        from monomials_deg_4 import f_monomials
        from polynomials_deg_4 import f_polynomials

        def f_mono(x): return f_monomials(x)
        def f_poly(x): return f_polynomials(x)
    elif deg == 5:
        from monomials_deg_5 import f_monomials
        from polynomials_deg_5 import f_polynomials

        def f_mono(x): return f_monomials(x)
        def f_poly(x): return f_polynomials(x)
    elif deg == 6:
        from monomials_deg_6 import f_monomials
        from polynomials_deg_6 import f_polynomials

        def f_mono(x): return f_monomials(x)
        def f_poly(x): return f_polynomials(x)
    else:
        raise ValueError(f'PIP degree must be between 3 to 6!')

    return f_mono, f_poly


def _training(n_tr: int, deg: int, n_val: int):

    # get monomials and polynomials functions
    f_mono, f_poly = get_pip_functions(deg)

    # load all CH4 data
    X_all, _, y_all, atoms = read_geometry_energy()

    # split training and validation data
    i0 = 0
    rng = jax.random.PRNGKey(i0)
    _, key = jax.random.split(rng)
    (X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
        X_all, y_all, n_tr, key, n_val)
    print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)

    # load PIP-Flax moddel
    model_pip = PIP(f_mono, f_poly)
    model_energy = EnergyPIP(f_mono, f_poly)
    params = model_energy.init(key, X_tr[:1])

    # training
    w = training(model_pip, X_tr, y_tr)
    params = flax_params(w, params)

    # prediction
    y_pred_tr = model_energy.apply(params, X_tr)
    y_pred_val = model_energy.apply(params, X_val)
    print(y_pred_tr.shape, y_pred_val.shape)

    loss_tr = mse_loss(y_pred_tr, y_tr)
    loss_val = mse_loss(y_pred_val, y_val)

    print(n_tr, loss_tr, loss_val)

    import matplotlib
    import matplotlib.pyplot as plt

    plt.scatter(y_pred_tr, y_tr)
    plt.scatter(y_pred_val, y_val)
    plt.show()
    return 0


def main():

    parser = argparse.ArgumentParser(
        description='Training PIP for Methane')
    parser.add_argument('--N', type=int,
                        default=100, help='number of training data')
    parser.add_argument('--Nval', type=int,
                        default=2500, help='number of validation data')
    parser.add_argument('--deg', type=int, default=3,
                        help='pip degree')
    args = parser.parse_args()

    N = int(args.N)
    Nval = int(args.Nval)
    deg = int(args.deg)

    _training(N, deg, Nval)


if __name__ == '__main__':
    main()
