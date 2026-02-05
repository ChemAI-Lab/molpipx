import molpipx
from molpipx import msa_file_generator


def test_msa_file_generator(n_A: int = 6, poly_degree: int = 3):

    filename = f'MOL_{n_A}_{poly_degree}'
    path = '/home/ravh011/molpipx/molpipx/msa_files/molecule_A6'
    label = f'_MOL_{n_A}_{poly_degree}'
    msa_file_generator(filename=filename, path=path, label=label)


if __name__ == "__main__":
    test_msa_file_generator()
