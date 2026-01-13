import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
# Assuming the functions are exposed under molpipx
from molpipx import training, training_w_gradients


class MockPIP(nn.Module):
    """
    A simple identity-like PIP model for testing linear solvers (a trivial linear system y = X*theta).
    """
    @nn.compact
    def __call__(self, x):
        # Flatten the last two dimensions (Atoms, 3)
        # Input: (Batch, Atoms, 3) -> Output: (Batch, Atoms*3)
        return x.reshape(x.shape[0], -1)


@pytest.fixture
def linear_data():
    """
    Generates synthetic data where Energy is a perfect linear combination of coordinates.
    Model: E = Dot(Flattened_Coords, Theta)
    Forces: Gradient(E) w.r.t Coords = Theta (reshaped)
    """
    n_batch = 10
    n_atoms = 3
    dims = 3
    n_features = n_atoms * dims
    
    key = jax.random.PRNGKey(42)
    
    # Generate Random Geometries (X)
    X_tr = jax.random.normal(key, (n_batch, n_atoms, dims))
    
    # Define Ground Truth Coefficients (Theta): Shape (N_features, 1)
    theta_true = jnp.linspace(0.1, 1.0, n_features).reshape(-1, 1)
    
    # Compute Energies (y = X_flat . theta)
    X_flat = X_tr.reshape(n_batch, -1)
    y_tr = jnp.dot(X_flat, theta_true)
    
    # Compute Forces (-Grad(E)
    # The training: PIP * theta = E  AND  Grad(PIP) * theta = F (in this test PIP(x) = x, Grad(PIP) is Identity)
    F_tr_flat = jnp.tile(theta_true.T, (n_batch, 1))
    F_tr = F_tr_flat.reshape(n_batch, n_atoms, dims)
    
    return X_tr, y_tr, F_tr, theta_true, MockPIP()



def test_training(linear_data):
    """
    Tests the 'training' function (Energy only).
    Verifies it can recover theta from y = X*theta using Least Squares.
    """
    X_tr, y_tr, _, theta_true, model = linear_data
    
    # Run training
    theta_pred = training(model, X_tr, y_tr)
    
    # Check shape
    assert theta_pred.shape == theta_true.shape
    
    # Check values (allow small numerical error)
    assert jnp.allclose(theta_pred, theta_true, atol=1e-5)

def test_training_w_gradients(linear_data):
    """
    Tests 'training_w_gradients' (Energy + Forces).
    Verifies it can recover theta using both Energy and Gradient information.
    """
    X_tr, y_tr, F_tr, theta_true, model = linear_data
    
    # Run training with gradients
    theta_pred = training_w_gradients(model, X_tr, F_tr, y_tr)
    
    # Check shape
    assert theta_pred.shape == theta_true.shape
    
    # Check values
    assert jnp.allclose(theta_pred, theta_true, atol=1e-5)

def test_training_w_gradients_noise_resilience(linear_data):
    """
    Tests that providing gradients helps (or at least works) even if we add small noise.
    """
    X_tr, y_tr, F_tr, theta_true, model = linear_data
    
    # Add tiny noise to Energies to make it slightly harder
    key = jax.random.PRNGKey(99)
    y_noisy = y_tr + 0.001 * jax.random.normal(key, y_tr.shape)
    
    # Train
    theta_pred = training_w_gradients(model, X_tr, F_tr, y_noisy)
    
    # Check values
    assert jnp.allclose(theta_pred, theta_true, atol=1e-2)