import os
from absl import logging
import ml_collections
import numpy as onp
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jrnd

from pipjax import PIPNN
from pipjax import mse_loss, mae_loss
from pipjax import get_functions, detect_molecule, split_train_and_test_data_w_forces
from pipjax import get_energy_and_forces

from jaxtyping import (
    Array,
    Float,
    PyTree,
    install_import_hook,
)

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import chex
import optax
from optax import ema

from ml_collections import config_dict
from train import *


@jax.jit
def apply_model_training(state, geometries, force_true, e_true, l0):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        e_pred, f_pred = state.apply_fn({'params': params}, geometries)
        e_pred = jnp.expand_dims(e_pred, axis=1)
        loss_e = mse_loss(e_pred, e_true)
        loss_f = jnp.linalg.norm(f_pred - force_true)

        return l0*loss_e + loss_f, (loss_e, loss_f)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_e, loss_f)), grads = grad_fn(state.params)
    return grads, (loss, (loss_e, loss_f))


@jax.jit
def apply_model(state, geometries, force_true, e_true):
    """Computes gradients, loss and accuracy for a single batch."""
    e_pred, f_pred = state.apply_fn({'params': state.params}, geometries)
    loss_e = mse_loss(e_pred, e_true)
    loss_f = jnp.linalg.norm(f_pred - force_true)
    return loss_e, loss_f


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng, l0):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['x'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['x']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_e_loss = []
    epoch_f_loss = []

    for perm in perms:
        batch_x = train_ds['x'][perm, ...]
        batch_f = train_ds['f'][perm, ...]
        batch_y = train_ds['e'][perm, ...]
        grads, (loss, (loss_e, loss_f)) = apply_model_training(
            state, batch_x, batch_f, batch_y, l0)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_e_loss.append(loss_e)
        epoch_f_loss.append(loss_f)
    train_loss = jnp.mean(jnp.array(epoch_loss))
    train_e_loss = jnp.mean(jnp.array(epoch_e_loss))
    train_f_loss = jnp.mean(jnp.array(epoch_f_loss))
    return state, (train_loss, train_e_loss, train_f_loss)


def get_datasets(n_tr, key, n_val=1000):

    from load_data_methane import read_geometry_energy as load_data

    X_all, F_all, y_all, atoms_all = load_data()
    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
        X_all, F_all, y_all, n_tr, key, n_val)

    train_ds = {'x': X_tr, 'f': F_tr, 'e': y_tr}
    val_ds = {'x': X_val, 'f': F_val, 'e': y_val}
    # test_ds = {'x': X_tst, 'e': y_test}

    return train_ds, val_ds


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    molecule_type = config.molecule
    poly_degree = config.poly_degree
    n_layers = config.n_layers
    n_neurons = config.n_neurons
    features = ()
    for i in range(n_layers):
        features = features + (n_neurons,)

    mol_dict, mol_sym = detect_molecule(molecule_type)
    na = get_number_of_atoms(mol_dict)
    f_mono, f_poly = get_functions(molecule_type, poly_degree)

    pipnn = PIPNN(f_mono, f_poly, features,)
    params = pipnn.init(rng, jnp.ones([1, na, 3]))['params']
    tx = optax.adam(config.learning_rate)

    @jax.jit
    def f_w_grad(params, geoms): return get_energy_and_forces(
        pipnn.apply, geoms, params)

    return train_state.TrainState.create(apply_fn=f_w_grad, params=params, tx=tx)


@chex.dataclass
class loss_values:
    tr_loss: chex.ArrayDevice
    tr_e_loss: chex.ArrayDevice
    tr_f_loss: chex.ArrayDevice


def train_and_evaluate(
    config: config_dict.ConfigDict,
    workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    rng = jax.random.key(0)

    n_tr = config.ntr
    n_val = config.nval
    n_tst = config.ntst
    l0 = config.l0

    workdir_parms = workdir + f'_l0_{l0:.4f}'
    if not os.path.exists(workdir_parms):
        os.makedirs(workdir_parms)

        # read training data
    train_ds, val_ds = get_datasets(n_tr, rng, n_val)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    loss_trj_ema = ema(decay=0.99)
    loss_trj_ema_state = loss_trj_ema.init(loss_values(tr_loss=jnp.array(0.),
                                                       tr_e_loss=jnp.array(0.),
                                                       tr_f_loss=jnp.array(0.)))

    df = pd.DataFrame()
    rng, init_rng = jax.random.split(init_rng)
    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)

        state, (train_loss, train_e_loss, train_f_loss) = train_epoch(
            state, train_ds, config.batch_size, input_rng, l0
        )
        val_e_loss, val_f_loss = apply_model(
            state, val_ds['x'], val_ds['f'], val_ds['e'])

        epoch_values = loss_values(
            tr_loss=train_loss,
            tr_e_loss=train_e_loss,
            tr_f_loss=train_f_loss,)
        # values ema
        epoch_values_ema, loss_trj_ema_state = loss_trj_ema.update(
            epoch_values, loss_trj_ema_state)

        print('epoch:% 3d, (ema)tr_loss: %.4f, (ema)train_e_loss: %.6f, (ema)tr_f_loss: %.4f, val_e_loss: %.6f, val_f_loss: %.6f'
              % (
                  epoch,
                  epoch_values_ema.tr_loss,
                  epoch_values_ema.tr_e_loss,
                  epoch_values_ema.tr_f_loss,
                  val_e_loss, val_f_loss,
              ))
        r_epoch = {'epoch': epoch,
                   'tr_loss': train_loss,
                   'ema_tr_loss': epoch_values_ema.tr_loss,
                   'ema_tr_e_loss': epoch_values_ema.tr_e_loss,
                   'ema_tr_f_loss': epoch_values_ema.tr_f_loss,
                   'val_e_loss': val_e_loss,
                   'val_f_loss': val_f_loss,
                   }
        df = pd.concat(
            [df, pd.DataFrame(r_epoch, index=[0])], ignore_index=True)
        df.to_csv(
            f"{workdir}/training_trajectory_ema_wgrad_l0_{l0:.4f}.csv", index=False)

        # save models
        checkpoints.save_checkpoint(
            ckpt_dir=workdir_parms, target=state.params, step=epoch, keep=1, overwrite=True)

    return state
