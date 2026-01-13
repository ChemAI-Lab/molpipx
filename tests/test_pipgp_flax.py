import pytest
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from flax import nnx
import gpjax as gpx
from dataclasses import dataclass, field
from typing import Any
from molpipx import PIPLayerKernel


class MockNetwork(nn.Module):
    """
    A simple network to mimic PIPlayer behavior for testing the Kernel wrapper.
    It projects inputs to a latent space.
    """
    output_dim: int = 2

    @nn.compact
    def __call__(self, x):
        # Expecting x to be (Batch, Atoms, 3) or (1, Atoms, 3)
        # Flatten input: (Batch, Atoms*3)
        x_flat = x.reshape(x.shape[0], -1)
        # Simple projection
        out = nn.Dense(self.output_dim)(x_flat)
        # Remove batch dim if size 1 to act like a vector for the base kernel
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out


@pytest.fixture
def mock_setup():
    """Sets up a kernel instance with a mock network."""
    key = jax.random.PRNGKey(42)
    
    # Dummy data: 1 molecule, 2 atoms, 3D coordinates => (1, 2, 3)
    dummy_x = jnp.array([[[0., 0., 0.], [1., 1., 1.]]])
    # Base Kernel (RBF)
    base_kernel = gpx.kernels.RBF()
    # Network
    network = MockNetwork(output_dim=2)
    return key, dummy_x, base_kernel, network


def test_PIPLayerKernel(mock_setup):
    """Test that the kernel initializes network parameters correctly."""
    key, dummy_x, base_kernel, network = mock_setup
    
    kernel = PIPLayerKernel(
        base_kernel=base_kernel,
        network=network,
        dummy_x=dummy_x,
        key=key
    )
    
    # Check that nn_params were generated
    assert kernel.nn_params is not None
    # MockNetwork uses a Dense layer, so 'params' should exist
    assert 'params' in kernel.nn_params
    
def test_kernel_init_validation(mock_setup):
    """Test that missing arguments raise errors."""
    key, dummy_x, base_kernel, network = mock_setup
    
    # Missing network
    with pytest.raises(ValueError, match="network must be specified"):
        PIPLayerKernel(
            base_kernel=base_kernel,
            dummy_x=dummy_x
        )


def test_PIPLayerKernel_forward(mock_setup):
    """
    Test the __call__ method, specifically the 1D -> 3D reshaping logic for 'x'.
    """
    key, dummy_x, base_kernel, network = mock_setup
    
    kernel = PIPLayerKernel(
        base_kernel=base_kernel,
        network=network,
        dummy_x=dummy_x,
        key=key
    )
    

    # Input x: 1D array (flattened geometry) -> Needs reshaping by kernel
    x_1d = jnp.zeros(6) 
    
    # y: 3D array (already shaped) -> Kernel passes as is (based on provided code)
    # Note: If the actual implementation doesn't reshape 'y', we must pass it shaped.
    y_3d = jnp.ones((1, 2, 3)) 
    
    # Call kernel
    result = kernel(x_1d, y_3d)
    
    # Result should be a scalar (0-dim array) representing distance/kernel value
    assert result.shape == () # GPJax typically returns a scalar () for single pair evaluation
    

    # Check initialized params
    params = kernel.nn_params
    
    # Run network 
    x_reshaped = x_1d.reshape(1, 2, 3)
    xt = network.apply(params, x_reshaped)
    yt = network.apply(params, y_3d)
    
    # Run base kernel 
    expected = base_kernel(xt, yt)
    
    assert jnp.allclose(result, expected)

def test_PIPLayerKernel_identical_inputs(mock_setup):
    """Test kernel with identical inputs (should be max correlation)."""
    key, dummy_x, base_kernel, network = mock_setup
    
    kernel = PIPLayerKernel(
        base_kernel=base_kernel,
        network=network,
        dummy_x=dummy_x,
        key=key
    )
    
    # Pass dummy_x as both arguments
    result = kernel(dummy_x, dummy_x)
    
    # RBF kernel of identical vectors is 1.0 (variance)
    assert jnp.allclose(result, jnp.array([1.0])) # MockNetwork is deterministic, so outputs will be identical.