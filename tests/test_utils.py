import pytest
import jax.numpy as jnp
import numpy as np
import jax
from molpipx.utils import (
    all_distances, 
    softplus_inverse, 
    morse_variables, 
    mse_loss, 
    mae_loss,
    flax_params,
    flax_params_aniso
)


def test_all_distances():
    """Test distance calculation on a 3-4-5 Right Triangle."""

    coords = jnp.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0]
    ])
    
    expected = jnp.array([3.0, 4.0, 5.0])
    result = all_distances(coords)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_all_distances_shape():
    """Test that N atoms produce N*(N-1)/2 distances."""
    N = 5
    # Random coordinates
    coords = jnp.zeros((N, 3))
    expected_pairs = (N * (N - 1)) // 2  # 5*4/2 = 10
    
    result = all_distances(coords)
    assert result.shape == (expected_pairs,)

def test_softplus_inverse():
    """Test that softplus_inverse is the mathematical inverse of softplus."""
    # softplus(x) = log(1 + exp(x))
    # input to softplus_inverse must be positive (domain of softplus output)
    y = jnp.array([1.0, 2.0, 10.0])
    
    x = softplus_inverse(y)
    y_reconstructed = jax.nn.softplus(x)
    np.testing.assert_allclose(y_reconstructed, y, rtol=1e-5)

@pytest.mark.parametrize("l_val, expected",[
    (0.0, 1.0),            # Limit case: exp(0) = 1
    (2.0, jnp.exp(-2.0))   # Calculation case: exp(-2*1)
])
def test_morse_variables(l_val, expected):
    """
    Test morse variables calculation.
    Case 1: lambda=0 -> Result should be 1.0
    Case 2: lambda=2 -> Result should be exp(-2*dist)
    """
    coords = jnp.array([[0.,0.,0.], [1.,0.,0.]])
    l_param = jnp.array(l_val)
    
    # exp(-0 * r) = exp(0) = 1
    result = morse_variables(coords, l_param)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_losses():
    """Test MSE and MAE with known integers."""
    pred = jnp.array([2.0, 4.0])
    target = jnp.array([2.0, 6.0])
    
    # Errors: [0, -2]
    # Squared: [0, 4] -> Mean: 2.0
    # Abs: [0, 2] -> Mean: 1.0
    
    mse = mse_loss(pred, target)
    mae = mae_loss(pred, target)
    
    assert mse == 2.0
    assert mae == 1.0


def test_flax_params_update():
    """Test that flax_params updates the correct key in the dict."""
    # Setup a dummy parameter dictionary matching the expected structure
    initial_params = {
        'params': {
            'Dense_0': {
                'kernel': jnp.zeros((2, 2)) # Original is zeros
            }
        }
    }
    
    # Create flat input array to update with
    # 4 elements are passed, expecting them to be reshaped to (2,2)
    new_weights = jnp.array([1.0, 2.0, 3.0, 4.0])
    
    updated_params = flax_params(new_weights, initial_params)
    result_kernel = updated_params['params']['Dense_0']['kernel']
    expected_kernel = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    np.testing.assert_array_equal(result_kernel, expected_kernel)

def test_flax_params_aniso_update():
    """Test that flax_params_aniso updates the correct key."""
    initial_params = {
        'params': {
            'VmapJitPIPAniso_0': {
                'lambda': jnp.zeros((3,))
            }
        }
    }
    
    new_lambda = jnp.array([0.1, 0.2, 0.3])
    
    updated_params = flax_params_aniso(new_lambda, initial_params)
    
    result_lambda = updated_params['params']['VmapJitPIPAniso_0']['lambda']
    np.testing.assert_array_equal(result_lambda, new_lambda)