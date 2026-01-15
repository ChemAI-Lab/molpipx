import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platform_name", "cpu")

def test_methane_coords_loading(methane_coords):
    """
    Tests that the methane_coords fixture (via molpipx.data) loads correctly.
    """
    # Verify it returns a JAX array
    assert hasattr(methane_coords, 'shape'), "Fixture should return a JAX-compatible array"

    # Verify Dimensions: (Batch, Atoms, 3)
    assert methane_coords.ndim == 3, f"Expected 3 dimensions, got {methane_coords.ndim}"
    assert methane_coords.shape[1] == 5, f"Expected 5 atoms (CH4), got {methane_coords.shape[1]}"
    assert methane_coords.shape[2] == 3, f"Expected 3 coords (XYZ), got {methane_coords.shape[2]}"

    # Verify Batch Size
    assert methane_coords.shape[0] > 0, "Batch dimension should be > 0"
    
    # Verify Specific Values
    first_atom_first_frame = methane_coords[0, 0] # the first atom is selected to ensure we are comparing shape (3,) vs (3,)
    
    # Expected values for the first Hydrogen atom
    expected_first_atom = jnp.array([-0.1827861, 0.5603340, -0.9332645])
    
    # Used 1e-5 to accommodate potential float32 precision differences
    assert jnp.allclose(first_atom_first_frame, expected_first_atom, atol=1e-5), \
        f"First atom coordinates incorrect.\nGot: {first_atom_first_frame}\nExp: {expected_first_atom}"

    # Verify Data Integrity
    assert not jnp.isnan(methane_coords).any(), "Parsed coordinates contain NaNs"