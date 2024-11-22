import re
import numpy as np

ang_to_bohr = 0.529177210929
bhor_to_ang = ang_to_bohr * 1.0e-1

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
  
def read_geometry_energy(filename:str='Train_data_50000.xyz',
                         num_atoms:int = 9,
                         energy_normalization:bool = True):    
    
    with open(filename, 'r') as file:
        data = file.read()

    # Split the data into blocks using a blank line as a separator
    blocks = re.split(r'\s\s9\n', data.strip())

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
            if len(parts) == 4 + 3:
                atom, x, y, z, dx, dy, dz = parts
                coordinates.append([float(x), float(y), float(z), float(dx), float(dy), float(dz)])
                atoms_i.append((atom))

        # Append data to lists
        geometries.append(list(coordinates))
        energies.append(energy)
        atoms.append(list(atoms_i))

    energies = np.array(energies)
    geometries_and_forces = np.array(geometries)   
    geometries = geometries_and_forces[:,:,0:3]
    forces = geometries_and_forces[:,:,3:]#*ang_to_bohr
    if energy_normalization:
        e_min,e_max = np.min(energies), np.max(energies)
        energies = (energies - e_min) / (e_max - e_min)
    return geometries, forces, energies, np.array(atoms)

def main():
    
    geoms, forces, energies, and_atoms = read_geometry_energy()
    print(geoms.shape, forces.shape, energies.shape, and_atoms.shape)      
    print(geoms[1])
    print(forces[1])
    print(energies[1])      
    
    import matplotlib.pyplot as plt
    
    plt.hist(energies,bins=100)
    plt.show()
    


if __name__ == '__main__':
    main()  