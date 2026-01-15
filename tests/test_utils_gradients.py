import pytest
import jax
import jax.numpy as jnp
import gpjax as gpx
import flax.linen as nn
from gpjax import Dataset

from molpipx import (
    EnergyPIP, 
    PIPlayer as PIP,
    PIPlayerGP,
    training_w_gradients, 
    get_forces,
    get_energy_and_forces,
    get_pip_grad,
    get_forces_gp,
    flax_params, 
    split_train_and_test_data_w_forces,
    PIPLayerKernel
)
from molpipx.load_pip import get_functions
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

class BatchWrapper(nn.Module):
    """
    Wraps a model to ensure it handles unbatched inputs by temporarily adding a batch dimension.
    Required because GPJax kernels often pass single data points (Atoms, 3)
    whereas PIPlayerGP strictly expects (Batch, Atoms, 3).
    """
    model: nn.Module

    @nn.compact
    def __call__(self, x):
        is_unbatched = x.ndim == 2
        if is_unbatched:
            # Add fake batch dim: (Atoms, 3) -> (1, Atoms, 3)
            x = jnp.expand_dims(x, 0)
        
        # Call base model
        out = self.model(x)
        
        if is_unbatched:
            # Remove fake batch dim: (1, Features) -> (Features)
            out = jnp.squeeze(out, 0)
            
        return out

@pytest.fixture
def pip_functions():
    try:
        # Using degree 2 for faster testing
        return get_functions("A4B", 3)
    except Exception as e:
        pytest.skip(f"Skipping: {e}")

@pytest.fixture
def trained_linear_model(dataset_context, pip_functions):
    X_all, y_all, F_all, n_total = dataset_context
    f_mono, f_poly = pip_functions
    key = jax.random.PRNGKey(42)

    if n_total < 2:
        pytest.skip("Not enough data.")

    n_train = n_total - 1
    (X_tr, F_tr, y_tr), _ = split_train_and_test_data_w_forces(
        X_all, F_all, y_all, N=n_train, key=key, Nval=1)

    model_pip = PIP(f_mono, f_poly)
    model_energy = EnergyPIP(f_mono, f_poly)
    params = model_energy.init(key, X_tr[:1])

    y_tr = y_tr.reshape(-1, 1)
    
    # Train
    w = training_w_gradients(model_pip, X_tr, F_tr, y_tr)
    params = flax_params(w, params)
    
    return model_energy, model_pip, params, X_tr

def test_get_forces(trained_linear_model):
    model_energy, _, params, X_tr = trained_linear_model
    X_sample = X_tr[:5]
    
    with jax.disable_jit():
        F_pred = get_forces(model_energy.apply, X_sample, params)
        
    batch, atoms, dim = X_sample.shape
    assert F_pred.reshape(batch, -1).shape == (batch, atoms * dim)
    assert not jnp.isnan(F_pred).any()

def test_get_energy_and_forces(trained_linear_model):
    model_energy, _, params, X_tr = trained_linear_model
    X_sample = X_tr[:5]
    
    with jax.disable_jit():
        E_pred, F_pred_joint = get_energy_and_forces(model_energy.apply, X_sample, params)
        F_pred_separate = get_forces(model_energy.apply, X_sample, params)
        
    batch, atoms, dim = X_sample.shape
    assert E_pred.shape[0] == batch
    assert jnp.allclose(F_pred_joint.reshape(batch, -1), F_pred_separate.reshape(batch, -1), atol=1e-5)

def test_get_pip_grad(trained_linear_model):
    _, model_pip, _, X_tr = trained_linear_model
    key = jax.random.PRNGKey(99)
    X_sample = X_tr[:5]
    
    params_pip = model_pip.init(key, X_sample[:1])
    
    with jax.disable_jit():
        grads_pip = get_pip_grad(model_pip.apply, X_sample, params_pip)
        
    batch, atoms, dim = X_sample.shape
    pip_vec = model_pip.apply(params_pip, X_sample[:1])
    n_poly = pip_vec.shape[-1]
    
    assert grads_pip.shape[1] == atoms * dim
    assert grads_pip.shape[2] == n_poly

def test_get_forces_gp(dataset_context, pip_functions):
    """
    Verifies force computation for the Gaussian Process model.
    """
    X_all, y_all, _, n_total = dataset_context
    f_mono, f_poly = pip_functions
    key = jax.random.PRNGKey(100)

    if n_total < 5:
        pytest.skip("Not enough data")

    # Prepare Data 
    X_sample = X_all[:5].astype(jnp.float64)
    y_sample = y_all[:5].reshape(-1, 1).astype(jnp.float64)

    # Flatten X for GPJax Dataset (N, Atoms*3)
    X_flat = X_sample.reshape(X_sample.shape[0], -1)
    train_ds = Dataset(X=X_flat, y=y_sample)

    # Setup PIP Layer with Wrapper
    base_pip = PIPlayerGP(f_mono, f_poly)
    # Wrap the model so it can handle rank-2 inputs from GPJax
    model_pip = BatchWrapper(model=base_pip)
    
    params_pip = model_pip.init(key, X_sample[:1])
    
    # Check output shape
    out = model_pip.apply(params_pip, X_sample[:1])
    n_poly = out.shape[1]

    # Setup Kernel
    # Use Rank 3 tensor (Batch=1, Atoms, 3) for dummy_x to satisfy PIPlayerGP init
    dummy_x_batch = X_sample[:1] # Shape (1, 5, 3)
    
    base_kernel = gpx.kernels.RBF(active_dims=list(range(n_poly)), lengthscale=jnp.ones(n_poly))
    
    kernel = PIPLayerKernel(
        network=model_pip, 
        base_kernel=base_kernel, 
        key=key, 
        dummy_x=dummy_x_batch 
    )

    # Create Posterior
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=train_ds.n)
    posterior = prior * likelihood

    # Run Force Calculation
    # X_sample passed here is (5, 5, 3) - get_forces_gp handles expanding dims internally
    with jax.disable_jit():
        forces, (mu, std) = get_forces_gp(gp_model=posterior, train_data=train_ds, x=X_sample)
    
    batch, atoms, dim = X_sample.shape
    
    assert forces.reshape(batch, -1).shape == (batch, atoms * dim)
    assert mu.shape == (batch,)
    assert not jnp.isnan(forces).any()