import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from molpipx.pipnn_flax import MLP, PIPNN


def mock_f_mono(x):
    """Identity mock for monomials."""
    return x

def mock_f_poly(x):
    """Identity mock for polynomials. Returns the input as the PIP vector."""
    return x

@pytest.fixture
def batch_geometry():
    """
    Batch of 5 geometries, each with 3 atoms in 3D space.
    """
    return jnp.ones((5, 3, 3))

@pytest.fixture
def batch_features():
    """
    Batch of dummy feature vectors for MLP testing.
    Shape: (5, 10)
    """
    return jnp.ones((5, 10))


def test_mlp(batch_features):
    """
    Tests that the MLP initializes its layers based on the 'features' tuple
    and produces a scalar output per batch item.
    """
    # Define MLP with 2 hidden layers (sizes 4 and 2)
    model = MLP(features=(4, 2), act_fun=nn.relu)
    
    key = jax.random.PRNGKey(0)
    
    params = model.init(key, batch_features) # Initialize
    
    # Check params structure ('layers_0', 'layers_1' (for the hidden layers) and 'last_layer')
    assert 'params' in params
    assert 'layers_0' in params['params']
    assert 'layers_1' in params['params']
    assert 'last_layer' in params['params']
    
    output = model.apply(params, batch_features) # Forward
    
    assert output.shape == (batch_features.shape[0], 1) # Output shape: (Batch, 1)
    assert not jnp.isnan(output).any()


def test_pipnn(batch_geometry):
    """
    Tests the PIPNN model.
    Verifies that it correctly integrates the PIPlayer (via mocks) and the MLP.
    """

    # f_poly returns the morse variables directly (size = n_pairs)
    # For 3 atoms, n_pairs = 3. So input to MLP is size 3.
    model = PIPNN(
        f_mono=mock_f_mono, 
        f_poly=mock_f_poly, 
        features=(8, 4), 
        l=1.5
    )
    
    key = jax.random.PRNGKey(1)
    
    params = model.init(key, batch_geometry) # Initialize (Input (Batch, Atoms, 3))
    
    # Check params structure (PIPlayer usually doesn't have params unless 'l' is trainable, )
    assert 'params' in params
    assert 'last_layer' in params['params']

    
   
    output = model.apply(params, batch_geometry)  # Forward 
    
    assert output.shape == (batch_geometry.shape[0], 1) # Output shape: (Batch, 1)
    assert not jnp.isnan(output).any()

def test_pipnn_activation(batch_geometry):
    """Test PIPNN with a specific activation function."""
    model = PIPNN(
        f_mono=mock_f_mono,
        f_poly=mock_f_poly,
        features=(4,),
        act_fun=nn.elu
    )
    
    key = jax.random.PRNGKey(2)
    params = model.init(key, batch_geometry)
    output = model.apply(params, batch_geometry)
    
    assert output.shape == (batch_geometry.shape[0], 1)