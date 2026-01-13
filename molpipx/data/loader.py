import os
import re
import numpy as np
import jax.numpy as jnp

# Constants
ANG_TO_BOHR = 0.529177210929

def load_xyz(filename: str, num_atoms: int = 5, energy_normalization: bool = False):
    """Loads  XYZ dataset.
    
    Args:
        filename (str): Path to the .xyz file on your system.
        num_atoms (int): Number of atoms per configuration.
        energy_normalization (bool): Whether to normalize energies.

    Returns:
        Tuple: (geoms, forces, energies, atoms)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find data file at: {filename}")
        
    return _read_geometry_energy(filename, num_atoms, energy_normalization)


def load_methane(energy_normalization: bool = False):
    """Loads Methane data (5 atoms)."""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'Methane', 'Methane.xyz')
    
    return load_xyz(path, num_atoms=5, energy_normalization=energy_normalization)

def load_ethanol(energy_normalization: bool = False):
    """Loads Ethanol data (9 atoms)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'Ethanol', 'Ethanol.xyz')
    
    return load_xyz(path, num_atoms=9, energy_normalization=energy_normalization)

def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _read_geometry_energy(filename: str, num_atoms: int, energy_normalization: bool):
    """Parses the specific XYZ format."""
    with open(filename, 'r') as file:
        data = file.read()

    separator = rf"\s\s{num_atoms}\n"
    blocks = re.split(separator, data.strip())

    geometries = []
    energies = []
    atoms = []

    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 2: continue

        if _is_float(lines[0]) and not _is_int(lines[0]):
            energy = float(lines[0])
        elif _is_int(lines[0]):
            energy = float(lines[1])
        else:
            continue

        coordinates = []
        atoms_i = []
        
        # Parse Atom Lines
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 7: # Expecting x, y, z, dx, dy, dz
                atom_type = parts[0]
                vals = [float(v) for v in parts[1:7]]
                coordinates.append(vals)
                atoms_i.append(atom_type)
        
        if len(coordinates) > 0:
            geometries.append(coordinates)
            energies.append(energy)
            atoms.append(atoms_i)

    energies = np.array(energies)
    data_block = np.array(geometries) # Shape: (N, Atoms, 6)
    
    geoms = data_block[:, :, 0:3]
    forces = data_block[:, :, 3:] / ANG_TO_BOHR
    
    if energy_normalization:
        e_min, e_max = np.min(energies), np.max(energies)
        energies = (energies - e_min) / (e_max - e_min)

    return (jnp.array(geoms), jnp.array(forces), 
            jnp.array(energies[:, np.newaxis]), np.array(atoms))