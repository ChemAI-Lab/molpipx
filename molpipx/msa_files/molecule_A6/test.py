import molpipx
from molpipx import msa_file_generator
from molpipx.utils import all_distances
from molpipx import detect_molecule, get_functions

import jax
import jax.random as jrnd


def generate_msa_file_generator(n_A: int = 6, poly_degree: int = 3):

    filename = f'MOL_{n_A}_{poly_degree}'
    path = '/home/ravh011/molpipx/molpipx/msa_files/molecule_A6'
    label = f'_MOL_{n_A}_{poly_degree}'
    for p in range(3, 7):  # max degree generated was 6
        filename = f'MOL_{n_A}_{p}'
        label = f'_MOL_{n_A}_{p}'
        msa_file_generator(filename=filename, path=path, label=label)


def test_mono_files(poly_degree: int = 3):
    n_a = 6
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    # random geometries to test the energy and forces
    geoms = jrnd.normal(key, (1, n_a, 3))
    distance = all_distances(geoms[0])
    print(f'Distances: {distance}')
    print(distance.shape)

    f_mono, f_poly = get_functions('A6', poly_degree)
    

if __name__ == "__main__":
    # generate_msa_file_generator()
    test_mono_files(poly_degree=3)
