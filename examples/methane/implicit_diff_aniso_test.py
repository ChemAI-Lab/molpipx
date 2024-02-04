from jaxopt import linear_solve
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import lax

from pipjax import split_train_and_test_data
from pipjax import LayerPIPAniso, EnergyPIPAniso, get_mask
from pipjax import flax_params_aniso, flax_params
from pipjax import training
from pipjax import mse_loss


from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import flax
from flax import linen as nn

from monomials_deg_3 import f_monomials as f_mono
from polynomials_deg_3 import f_polynomials as f_poly

from load_data_methane import read_geometry_energy as load_data

rng = jrnd.PRNGKey(0)
_, key = jrnd.split(rng)

X_all, _, y_all, atoms_all = load_data()
(X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
    X_all, y_all, 100, key, 100)

mask = get_mask(atoms_all[0])
n_pairs = mask.shape[0]

# initialize models
model_pip = LayerPIPAniso(f_mono, f_poly, n_pairs)
params_pip = model_pip.init(key, X_all[:1], mask)
print(params_pip)

pips_pred = model_pip.apply(params_pip, X_all[:2], mask)
print(pips_pred.shape)

model = EnergyPIPAniso(f_mono, f_poly, n_pairs)
params_energy = model.init(key, X_all[:1], mask)

print(params_energy)


l_init = jrnd.uniform(key, shape=(2, 1), minval=1., maxval=2.)
print(nn.softplus(l_init))
data = ((X_tr, y_tr), (X_val, y_val))
models = (model_pip, model)
params = (params_pip, params_energy, mask)

# v,g = jax.value_and_grad(validation_loss,argnums=(0,))(l_init, data,params,models)
# print(v)
# print(g)


def validation_loss(l, data, params, models):
    (X_tr, y_tr), (X_val, y_val) = data
    params_pip, params_energy, mask = params
    model_pip, model_energy = models

    def inner_training(l, X_tr, y_tr, params_pip):
        params_pip = flax_params_aniso(l, params_pip)
        Pip_tr = model_pip.apply(params_pip, X_tr, mask)
        results = jnp.linalg.lstsq(Pip_tr, y_tr)
        w = results[0]
        return w

    w = inner_training(l, X_tr, y_tr, params_pip)
    params_energy = flax_params(w, params_energy)
    y_val_pred = model_energy.apply(params_energy, X_val, mask)
    loss = mse_loss(y_val_pred, y_val)
    return loss


# def inner_training(l, X_tr, y_tr, params_pip):
#     params_pip = flax_params_aniso(l, params_pip)
#     Pip_tr = model_pip.apply(params_pip, X_tr, mask)
#     results = jnp.linalg.lstsq(Pip_tr, y_tr)
#     w = results[0]
#     return jnp.sum(w)

#     # def matvec_A(x):
#     #     print(x.shape)
#     #     return Pip_tr @ x
#     # results = linear_solve.solve_lu(matvec_A, y_tr.ravel())
#     # # w = results[0]
#     # return results
# print(jax.value_and_grad(loss, argnums=(0,))
#       (l_init, X_tr, y_tr, params_pip))
# print(inner_training(l_init, X_tr, y_tr, params_pip))


l_init = jrnd.uniform(key, shape=(2, 1), minval=1., maxval=2.)
print(nn.softplus(l_init))
data = ((X_tr, y_tr), (X_val, y_val))
models = (model_pip, model)
params = (params_pip, params_energy, mask)


v, g = jax.value_and_grad(validation_loss, argnums=(0,))(
    l_init, data, params, models)
print(v)
print(g)
