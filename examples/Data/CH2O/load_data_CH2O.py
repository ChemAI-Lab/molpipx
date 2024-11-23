import os
import re
import numpy as onp


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def get_file_name(name:str):
    if name.lower() == 'min':
        return '01_min.xyz'
    elif name.lower() == 'cis':
        return '02_cis.xyz'
    elif name.lower() == 'trans':
        return '03_trans.xyz'
    
def read_geometry_energy(filename:str,num_atoms:int =4):
    with open(filename, 'r') as file:
        data = file.read()

    # Split the data into blocks using a blank line as a separator
    blocks = re.split(r'\s\s4\n', data.strip())

    # Initialize lists to store geometries and energies
    geometries = []
    energies = []
    atoms = []

    # Process each block
    for block in blocks:
        lines = block.split('\n')

        # Extract energy from the first line of each block
        if is_float(lines[0]) and not is_int(lines[0]):
            energy = float(lines[0])
        elif is_int(lines[0]):
            energy = float(lines[1])        

        # Extract atom coordinates
        coordinates = []
        atoms_i = []
        for line in lines[1:]:
            parts = line.split()            
            if len(parts) == 4:
                atom, x, y, z = parts
                coordinates.append([float(x), float(y), float(z)])
                atoms_i.append((atom))

        # Append data to lists
        geometries.append(list(coordinates))
        energies.append(energy)
        atoms.append(list(atoms_i))
    geometries = onp.array(geometries)
    energies = onp.array(energies)[:,None]
    return geometries, energies, atoms

def load_i_data(name:str='min'):
    _path = './Data/CH2O/regions_datasets_only_energies'
    file_geoms = get_file_name(name)
    full_file_geoms = os.path.join(_path,file_geoms)
    
    return read_geometry_energy(full_file_geoms)

def main():
    for si in ['cis','trans','min']:
        geometries, energies, atoms = load_i_data(si)


if __name__ == '__main__':
    main()