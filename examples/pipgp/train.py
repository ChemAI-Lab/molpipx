import sys
sys.path.append('/scratch/r/ravh011/asmaj/PIPMSA_jax')
import jax
import jax.numpy as jnp
import optax as ox
import gpjax as gpx
from gpjax import Dataset
jax.config.update("jax_enable_x64", True)
from molpipx import get_functions, detect_molecule
from molpipx import split_train_and_test_data, split_train_and_test_data_w_forces
from molpipx import PIPlayerGP
from molpipx import PIPLayerKernel
from molpipx import get_forces_gp

import os
import pandas as pd
import numpy as np


from ml_collections import config_dict
from load_data_methane import read_geometry_energy as load_data




def get_datasets(n_tr, key, n_tst):

    X_all, _, y_all, _ = load_data()
    n_tst = y_all.shape[0] - n_tr    
    (X_tr, y_tr), (X_tst, y_tst) = split_train_and_test_data(
        X_all, y_all, n_tr, key, n_tst)
    
    train_ds = Dataset(X = X_tr.reshape(X_tr.shape[0], X_tr.shape[1]*X_tr.shape[2]).astype(jnp.float64), y = y_tr.astype(jnp.float64))
    X_tst = X_tst.reshape(X_tst.shape[0], X_tst.shape[1]*X_tst.shape[2])
    tst_ds = {'x': X_tst.astype(jnp.float64), 'e': y_tst.astype(jnp.float64)}

    return train_ds, tst_ds

def get_datasets_w_forces(n_tr, key, n_tst):

    X_all, F_all, y_all, _ = load_data()
    (X_tr, F_tr, y_tr), (X_tst, F_tst, y_tst) = split_train_and_test_data_w_forces(
        X_all, F_all, y_all, n_tr, key, n_tst)

    train_ds = Dataset(X = X_tr.reshape(X_tr.shape[0], X_tr.shape[1]*X_tr.shape[2]).astype(jnp.float64), y = y_tr.astype(jnp.float64))
    X_tst = X_tst.reshape(X_tst.shape[0], X_tst.shape[1]*X_tst.shape[2])
    tst_ds = {'x': X_tst.astype(jnp.float64), 'e': y_tst.astype(jnp.float64)}
    force_ds = {'f_tr': F_tr, 'f_tst': F_tst}

    return train_ds, tst_ds, force_ds


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def train_and_evaluate(config: config_dict.ConfigDict, workdir: str):  
    
    """
    Train and evaluate a Gaussian Process model for molecular potential energy surface prediction.

    Args:
        config: Configuration dictionary containing settings for the model and training.
        workdir: Working directory where results will be saved.

    Returns:
        tuple: A tuple containing training history, predictive mean, and predictive standard deviation.
    """
    
    molecule_type = config.molecule
    poly_degree = config.poly_degree
    num_iter = config.num_iter
    n_tr = config.ntr
    n_tst = config.ntst
    kernel_type = config.kernel_type
    trainable_l= config.trainable_l

    mol_dict, mol_sym = detect_molecule(molecule_type)
    na = get_number_of_atoms(mol_dict)
    f_mono, f_poly = get_functions(molecule_type, poly_degree)


    key = jax.random.key(123)
    X_train, X_test, forces_ds = get_datasets_w_forces(n_tr, key, n_tst)

    key, sub_key = jax.random.split(key)
    x0 = jax.random.normal(sub_key, (na, 3))
    # Reshape x0 into 3d array
    x0 = jnp.reshape(x0,(1,na,3))
    

    # pipgp_layer 
    pipgp_layer = PIPlayerGP(f_mono, f_poly, trainable_l= trainable_l)
    params = pipgp_layer.init(key, x0)
    output = pipgp_layer.apply(params, x0)
    n_poly = output.shape[1]
    

    kernel_init = {
        "RBF": gpx.kernels.RBF,
        "Matern12": gpx.kernels.Matern12,
        "Matern32": gpx.kernels.Matern32,
        "Matern52": gpx.kernels.Matern52
    }

    # Initialize the base kernel using the appropriate function based on kernel_type
    if kernel_type not in kernel_init:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    base_kernel = kernel_init[kernel_type](active_dims=list(range(n_poly)),
                                                     lengthscale=jnp.ones((n_poly,)),)

    
    kernel = PIPLayerKernel(network = pipgp_layer,
                            base_kernel = base_kernel,
                            key = key,
                            dummy_x = x0)
    
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function = meanf, kernel = kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints = X_train.n)
    posterior = prior * likelihood


    init_value = config.init_value
    peak_value = config.peak_value
    warmup_steps = config.warmup_steps
    decay_steps = config.decay_steps
    end_value = config.end_value

    
    schedule = ox.warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps,
                                               decay_steps, end_value)
    optimiser = ox.chain(
        ox.clip(1.0),
        ox.adamw(learning_rate = schedule),
    )
    # Start training process
    opt_posterior, history = gpx.fit(
        model = posterior,
        objective = jax.jit(gpx.objectives.ConjugateMLL(negative=True)),
        train_data = X_train,
        optim = optimiser,
        num_iters = num_iter,
        key = key,
        safe = False
    )

    # Energy prediction
    latent_dist = opt_posterior(X_test['x'], train_data = X_train)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    # Compute forces with trained model
    forces, _ = get_forces_gp(gp_model=opt_posterior, train_data=X_train, x=X_test['x'])

    # Save training history (loss values) to CSV file
    Results_path = os.path.join(workdir, 'Results/PIPGP')
    os.makedirs(Results_path, exist_ok=True)
    history_df = pd.DataFrame({'loss':np.array(history)})
    history_file = os.path.join(Results_path, f'his_{kernel_type}_{n_tr}.csv')
    history_df.to_csv(history_file, index=False)

  
    return history, predictive_mean, predictive_std, forces
