import re
import os
import sys
from typing import Any, Callable, List, Dict, Tuple

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# from pipjax.pip_generator import msa_file_generator
from pipjax.pip_generator import msa_file_generator
msa_path = 'pipjax/msa_files'


def parse_molecule(molecule: str):
    # This pattern matches elements followed optionally by a number
    # Elements are expected in the order A, B, C, D
    pattern = re.compile(r"(A|B|C|D|E)(\d*)")
    parts = pattern.findall(molecule)
    # Initialize dictionary with expected elements, defaulting to 0
    result = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for element, count in parts:
        if count == '':
            count = 1
        else:
            count = int(count)
        result[element] = result.get(element, 0) + count
    # Check if the found elements are in the correct sequence and complete
    expected_elements = ['A', 'B', 'C', 'D', 'E']
    found_elements = [part[0] for part in parts]
    if found_elements != expected_elements[:len(found_elements)]:
        return "Error: Elements are out of order or missing"

    mol_numbers = ""
    for i, k in enumerate(result):
        if result[k] != 0:
            mol_numbers += str(result[k])
            mol_numbers += "_"
        elif result[k] == 0:
            break
    if mol_numbers[-1] == "_":
        mol_numbers = mol_numbers[:-1]
    return result, mol_numbers


def detect_molecule(input_data):
    if isinstance(input_data, str):
        # Single molecule string
        return parse_molecule(input_data)
    elif isinstance(input_data, list):
        # List of molecule strings
        return [parse_molecule(molecule) for molecule in input_data]
    else:
        return "Invalid input type"


def get_functions(molecule_type: str, degree: str) -> tuple:
    mol_dict, mol_sym = detect_molecule(molecule_type)
    # Assumes this script is in the main_directory
    base_directory = Path(__file__).parent
    functions = {}

    # Construct file names
    poly_module_name = f"polynomials_MOL_{mol_sym}_{degree}.py"
    mono_module_name = f"monomials_MOL_{mol_sym}_{degree}.py"

    # Construct full paths to the modules
    poly_path = base_directory / 'msa_files' / \
        f'molecule_{molecule_type}' / poly_module_name
    # mono_path = base_directory / molecule_type / mono_module_name
    mono_path = base_directory / 'msa_files' / \
        f'molecule_{molecule_type}' / mono_module_name
    # Helper function to load module

    def load_module(path):
        module_name = path.stem  # Name of the module from the file name
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            return None

    # Load polynomial module and function
    poly_module = load_module(poly_path)
    if poly_module and hasattr(poly_module, 'f_polynomials'):
        functions['poly'] = getattr(poly_module, 'f_polynomials')

    # Load monomial module and function
    mono_module = load_module(mono_path)
    if mono_module and hasattr(mono_module, 'f_monomials'):
        functions['mono'] = getattr(mono_module, 'f_monomials')

    # return functions
    return functions['mono'], functions['poly']


def list_of_strings(arg):
    return arg.split(',')


def main():
    parser = argparse.ArgumentParser(description="run msa files")
    # parser.add_argument("--mol", type=str, required=True)
    parser.add_argument("--mol-list", type=list_of_strings, required=True)
    # parser.add_argument("--poly", type=int, required=False)
    parser.add_argument("--poly-list", type=list_of_strings, required=True)

    args = parser.parse_args()
    # molecule = args.mol
    mols = args.mol_list
    poly_degrees = args.poly_list

    # print(molecule)
    # mol, mol_sym = detect_molecule(molecule)
    # print(molecule, mol_sym)
    import jax
    import jax.numpy as jnp
    for moli in mols:
        moli_dict, mol_sym = detect_molecule(moli)
        print(moli)
        print(moli_dict)
        na = 0
        for i, k in enumerate(moli_dict):
            na += moli_dict[k]
        d = (na**2 - na)/2
        for p in poly_degrees:
            print(f'degree {p}')
            poly_degree = p
            # msa_path = 'pipjax/msa_files'
            msa_path = os.getcwd()
            msa_path = os.path.join(msa_path, 'msa_files')
            filename = 'MOL_' + mol_sym + f'_{poly_degree}'
            msa_path = os.path.join(msa_path,  'molecule_' + moli)
            print(filename)
            print(msa_path)
            print(os.getcwd())

            # parents_path = f'pipjax.msa_files.molecule_{molecule}'
            # msa_file_generator(filename, msa_path, mol_sym, parents_path)

            f_mono, f_poly = get_functions(moli, p)

            x = jnp.ones(int(d))
            x_mono = f_mono(x)
            x_poly = f_poly(x)
            print(x)
            print(x_mono.shape, x_mono)
            print(x_poly.shape, x_poly)
            print("-------------------------")


if __name__ == "__main__":
    main()


'''

# from pipjax.msa_files.molecule_A2B.monomials_MOL_2_1_3 import f_monomials as f_monos

'''
