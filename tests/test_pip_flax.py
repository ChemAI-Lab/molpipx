import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

from molpipx.pip_flax import PIP, PIPlayer, EnergyPIP, PIPlayerGP

def mock_f_mono(x):
    """Identity for testing."""
    return x

def mock_f_poly(x):
    """Identity for testing."""
    return x

@pytest.fixture
def single_geometry():
    # Single geometry: 3 atoms, 3D
    # 3 atoms -> 3 unique pairs
    return jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

@pytest.fixture
def batch_geometry():
    # Batch of 2 geometries, 3 atoms each
    return jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], # Geom 1
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]  # Geom 2 (shifted z)
    ])


def test_pip_forward(single_geometry):
    """Test PIP layer with a single geometry input."""
    model = PIP(f_mono=mock_f_mono, f_poly=mock_f_poly, trainable_l=False)
    key = jax.random.PRNGKey(0)
    
    params = model.init(key, single_geometry) # Initialize (Input batch 1)
    output = model.apply(params, single_geometry) # Forward (Input batch 1)
    
    # 3 atoms = 3 distances. f_poly returns input ==> output should be vector of size 3.
    assert output.shape == (3,)
    
    # Verify values are finite
    assert jnp.all(jnp.isfinite(output))
    
    # Should allow no params when trainable_l is False
    assert len(params) == 0

def test_pip_trainable_l(single_geometry):
    """Test PIP layer with learnable length scale parameter."""
    model = PIP(f_mono=mock_f_mono, f_poly=mock_f_poly, trainable_l=True)
    key = jax.random.PRNGKey(0)
    
    params = model.init(key, single_geometry) # Initialize (Input batch 1)
    
    # Check that 'lambda' parameter is created
    assert 'params' in params
    assert 'lambda' in params['params']
    assert params['params']['lambda'].shape == (1,)
    
    output = model.apply(params, single_geometry) # Forward (Input batch 1)
    assert output.shape == (3,) # Output should be vector of size 3


def test_piplayer_forward(batch_geometry):
    """Test PIPlayer with a batch of inputs."""
    model = PIPlayer(f_mono=mock_f_mono, f_poly=mock_f_poly)
    key = jax.random.PRNGKey(0)
    
    params = model.init(key, batch_geometry) # Initialize (Input batch 2)
    output = model.apply(params, batch_geometry) # Forward (Input batch 2)
    
    assert output.shape == (2, 3) # Output shape (Batch size 2, 3 pips per geometry)


def test_energy_pip_forward(batch_geometry):
    """Test EnergyPIP output shape (scalar energy per batch item)."""
    model = EnergyPIP(f_mono=mock_f_mono, f_poly=mock_f_poly)
    key = jax.random.PRNGKey(0)
    
    params = model.init(key, batch_geometry) # Initialize (Input batch 2)
    
    assert 'params' in params # Check params to ensure whether Dense layer is initialized

    output = model.apply(params, batch_geometry) # Forward (Input batch 2)
    
    assert output.shape == (2, 1)   # Output shape (Batch, 1)


def test_piplayer_gp(single_geometry):
    """Test handling of 1D flattened input array (Na*3,)."""
    model = PIPlayerGP(f_mono=mock_f_mono, f_poly=mock_f_poly)
    key = jax.random.PRNGKey(0)
    
    # Flatten input to simulate GPJax single point input logic
    flat_input = single_geometry.flatten() # Shape (9,)
    
    params = model.init(key, flat_input) # Initialize (Flatted input batch 1)
    output = model.apply(params, flat_input) # Forward (Flatted input batch 1)
    
    # Internally reshapes to (1, Na, 3), applies PIPlayer -> (1, P)
    assert output.shape == (1, 3)

def test_piplayer_gp_standard_batch(batch_geometry):
    """Test PIPlayerGP passing through standard batch input."""
    model = PIPlayerGP(f_mono=mock_f_mono, f_poly=mock_f_poly)
    key = jax.random.PRNGKey(0)
    
    params = model.init(key, batch_geometry)  # Initialize (Input batch 2)
    output = model.apply(params, batch_geometry) # Forward (Input batch 2)
    
    assert output.shape == (2, 3) # (Batch, Number of PIPs)