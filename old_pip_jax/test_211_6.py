import jax
import jax.numpy as jnp
from pip_utils import *
from training_utils import *

from training_utils import get_pip_grad, split_train_and_test_data
from polynomials_2_1_1_6 import f_polynomials as f_poly
from monomials_2_1_1_6 import f_monomials as f_mono
from load_data_CH2O import load_i_data as load_data


def main():

    xyz = jnp.array([[-0.93420327,   0.00025591,  -0.58483022],
                     [0.93420368,   0.00005784,  -0.58483017],
                     [0.00000019,   0.00012556,  -0.00066180],
                     [0.00000014,   0.00006061,   1.21105410]])
    # xyz = lax.expand_dims(xyz,dimensions=(0,))
    print('xyz', xyz)

    # PIP model in flax
    model_pip = PIP(f_mono, f_poly,)
    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)

    params_i = model_pip.init(key, xyz[jnp.newaxis])
    print(params_i)
    print('---')

    geometries, energies, atoms = load_data()
    print(geometries.shape)
    print(energies.shape)
    # assert 0
    print('---')

    # training
    _, key = jax.random.split(key)
    def f_pip(params, x): return model_pip.apply(params, x)

    (Xtr, ytr), (Xtst, ytst) = split_train_and_test_data(
        geometries, energies, 2000, key)

    # optimization
    Piptr = model_pip.apply(params_i, Xtr)
    results = jnp.linalg.lstsq(Piptr, ytr)
    theta = results[0]
    print(theta[:10])

    # y_pred_tr = Piptr@theta
    # y_pred_tst = Piptst@theta

    model_energy = PIPLinear(f_mono, f_poly,)
    params = model_energy.init(key, lax.expand_dims(xyz, dimensions=(0,)))
    params['params']['Dense_0']['kernel'] = theta
    print(params)
    # assert 0

    y_pred_tr_flx = model_energy.apply(params, Xtr)
    y_pred_tst_flx = model_energy.apply(params, Xtst)

    g_tr = get_pip_grad(
        model_pip.apply, geometries[:2], params_i)  # tensor
    print(g_tr.shape)

    jnp.linalg.norm(g_tr)
    # print('---')


if __name__ == '__main__':
    main()
