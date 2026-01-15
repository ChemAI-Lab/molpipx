import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.core
from molpipx.pip_anisotropic_flax import get_mask, get_f_mask, PIPAniso, LayerPIPAniso, EnergyPIPAniso, lambda_random_init


def test_get_mask():
    """
    Verifies that get_mask correctly identifies unique pairs and constructs
    the binary mask matrix.
    """
    # Example: Water-like topology (H, H, O)
    atom_types = ['H', 'H', 'O']
    mask, unique_pairs = get_mask(atom_types)
    
    # Pairs generated logic:
    # 0,1: H-H -> HH
    # 0,2: H-O -> HO
    # 1,2: H-O -> HO
    #
    # Sorted Unique Pairs: order depends on implementation, but likely encounter order [HH, HO]
    
    assert len(unique_pairs) == 2
    assert 'HH' in unique_pairs
    assert 'HO' in unique_pairs
    
    # Total pairs = 3 (HH, HO, HO)
    # Mask shape should be (n_unique_pairs, n_total_pairs) -> (2, 3)
    assert mask.shape == (2, 3)
    
    # Verify Mask Values (One-hot-like encoding per unique pair type)
    # Check HH row (should select only the first pair)
    # Check HO row (should select the 2nd and 3rd pairs)
    
    # We find the index of HH and HO in unique_pairs to be robust
    idx_hh = unique_pairs.index('HH')
    idx_ho = unique_pairs.index('HO')
    
    # The pairs list internal to function was [HH, HO, HO]
    # Row for HH should be [1, 0, 0]
    assert mask[idx_hh, 0] == 1.
    assert mask[idx_hh, 1] == 0.
    assert mask[idx_hh, 2] == 0.
    
    # Row for HO should be [0, 1, 1]
    assert mask[idx_ho, 0] == 0.
    assert mask[idx_ho, 1] == 1.
    assert mask[idx_ho, 2] == 1.


def test_get_f_mask():
    """
    Verifies that the JIT-compiled function returned by get_f_mask correctly 
    computes the weighted distance vector.
    """
    # Create a simple mask for 2 unique pairs, 3 total distances
    # Pair 0 covers dist 0. Pair 1 covers dist 1 and 2.
    mask = jnp.array([
        [1., 0., 0.],  # Unique Pair A
        [0., 1., 1.]   # Unique Pair B
    ])
    
    f_mask = get_f_mask(mask)
    
    # l (length scales) for unique pairs: [l_A, l_B]
    l = jnp.array([2.0, 0.5]) 
    
    # d (distances): [d0, d1, d2]
    d = jnp.array([10.0, 20.0, 30.0])
    
    # Calculate f_mask:
    # z = vmap((l*mask) * d) over unique pairs dimension
    # i=0: (2.0 * [1,0,0]) * [10,20,30] 
    # i=1: (0.5 * [0,1,1]) * [10,20,30]
    # sum(z, axis=0) = [20, 10, 15]
    
    result = f_mask(l, d)
    
    expected = jnp.array([20.0, 10.0, 15.0])
    
    assert jnp.allclose(result, expected)


def test_lambda_random_init():
    """
    Tests that lambda_random_init correctly randomizes the 'lambda' parameter
    within the specific bounds [0.3, 2.5].
    """
    # We construct a fake params dictionary that matches the structure 
    # expected by the function: params['params']['VmapJitPIPAniso_0']['lambda']
    

    # The logic of the function is tested here by providing the expected structure.
    
    key = jax.random.PRNGKey(42)
    
    # Create dummy params
    # Shape (2,) assuming 2 unique pairs
    initial_lambda = jnp.array([1.0, 1.0]) 
    
    params = {
        'params': {
            'VmapJitPIPAniso_0': {
                'lambda': initial_lambda
            }
        }
    }
    
    # Apply randomization
    new_params = lambda_random_init(params, key)
    new_lambda = new_params['params']['VmapJitPIPAniso_0']['lambda']
    
    
    # Values should have changed
    assert not jnp.allclose(new_lambda, initial_lambda)
    
    # Values should be within bounds [0.3, 2.5]
    assert jnp.all(new_lambda >= 0.3)
    assert jnp.all(new_lambda <= 2.5)


def test_layer_pip_aniso_forward():
    """
    Tests LayerPIPAniso on a batch of geometries.
    """

    def mock_f_mono(x): return x  # Identity for testing
    def mock_f_poly(x): return x  # Identity for testing


    atom_types = ['H', 'H', 'O']
    mask, unique_pairs = get_mask(atom_types)
    f_mask = get_f_mask(mask)
    n_pairs = len(unique_pairs)

    model = LayerPIPAniso(
        f_mono=mock_f_mono,
        f_poly=mock_f_poly,
        f_mask=f_mask,
        n_pairs=n_pairs
    )

    # Batch of 5 molecules, 3 atoms each, 3D coords
    key = jax.random.PRNGKey(1)
    batch_input = jax.random.normal(key, (5, 3, 3)) # Shape (5, 3, 3)
    params = model.init(key, batch_input) # Initialize
    output = model.apply(params, batch_input)  # Forward
    
    # Batch size 5. Output dim 3 (identity poly).
    assert output.shape == (5, 3)
    assert jnp.all(jnp.isfinite(output))


def test_energy_pip_aniso_forward():
    """
    Tests EnergyPIPAniso to ensure it reduces output to a scalar energy value.
    """
    def mock_f_mono(x): return x  # Identity for testing
    def mock_f_poly(x): return x  # Identity for testing
    
    atom_types = ['H', 'H', 'O']
    mask, unique_pairs = get_mask(atom_types)
    f_mask = get_f_mask(mask)
    n_pairs = len(unique_pairs)
    
    model = EnergyPIPAniso(
        f_mono=mock_f_mono,
        f_poly=mock_f_poly,
        f_mask=f_mask,
        n_pairs=n_pairs
    )
    
    key = jax.random.PRNGKey(1)
    inputs = jax.random.normal(key, (5, 3, 3)) # Batch 5
    
    params = model.init(key, inputs) # Initialize
    energy = model.apply(params, inputs) # Forward
    
    # Expected shape: (Batch, 1)
    assert energy.shape == (5, 1)
