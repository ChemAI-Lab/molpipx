import pytest
import jax.numpy as jnp
from molpipx.data import load_methane

@pytest.fixture
def methane_data():
    """
    Returns real Methane data using the molpipx package loader.
    
    Adapts the package output (geoms, forces, energies, atoms) 
    to the legacy test format (coords, energies, forces).
    """
    # 1. Load from package
    # Returns: (Batch, Atoms, 3), (Batch, Atoms, 3), (Batch, 1), (Batch, Atoms)
    geoms, forces, energies, atoms = load_methane(energy_normalization=False)
    
    # 2. Shape Adaptation
    # Your old parser returned energies as shape (Batch,), 
    # but the new loader returns (Batch, 1). We squeeze it to avoid shape errors.
    energies = jnp.squeeze(energies)
    
    # 3. Order Adaptation
    # Old fixtures expect: (coords, energies, forces)
    # New loader provides: (geoms, forces, energies, atoms)
    return geoms, energies, forces

@pytest.fixture
def methane_coords(methane_data):
    """Returns just the coordinates array."""
    return methane_data[0]

@pytest.fixture
def methane_forces(methane_data):
    """
    Returns just the forces array.
    """
    # Note: I corrected the index from [1] (Energies) to [2] (Forces) 
    # based on the tuple order returned above.
    return methane_data[2]

@pytest.fixture
def dataset_context(methane_data):
    """
    Helper fixture to unpack methane_data and calculate n_total once.
    Returns: (geo, en, forces, n_total)
    """
    # Unpack the tuple from methane_data
    geo, en, forces = methane_data
    n_total = geo.shape[0]
    return geo, en, forces, n_total