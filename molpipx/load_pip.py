import re
import os
from typing import Any, List, Dict, Tuple

import argparse
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp


from molpipx.pip_generator import msa_file_generator
msa_path = 'molpipx/msa_files'


def parse_molecule(molecule: str):
    """Parses a molecule string into its constituent atom counts and symmetry label.

    Analyzes a string (e.g., 'A2B') to count atoms of types A, B, C, D, E.
    It verifies the order and returns a dictionary of counts along with a
    formatted symmetry string (e.g., '2_1').

    Args:
        molecule (str): A string representing the molecule (e.g., "A2B").

    Returns:
        tuple: A dictionary of atom counts (e.g., {'A': 2, 'B': 1, ...}) and a
        string representing the molecule's numerical symmetry (e.g., "2_1").
    """

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
    """Detects and parses molecule information from a string or list of strings.

    Serves as a wrapper around ``parse_molecule``. 

    Args:
        input_data (str or list): A single molecule string (e.g., "A2B") or a list
        of molecule strings.

    Returns:
        tuple or list: If input is a string, returns the result of ``parse_molecule``.
        If input is a list, returns a list of such results.
    """
    if isinstance(input_data, str):
        # Single molecule string
        return parse_molecule(input_data)
    elif isinstance(input_data, list):
        # List of molecule strings
        return [parse_molecule(molecule) for molecule in input_data]
    else:
        return "Invalid input type"


def get_functions(molecule_type: str, degree: str) -> tuple:
    """Retrieves the dynamically loaded monomial and polynomial functions for a given molecule.

    Loads the corresponding Python modules from the `msa_files` directory based on the
    molecule type and polynomial degree. It specifically looks for `f_monomials` and
    `f_polynomials` within those modules.

    Args:
        molecule_type (str): The type of the molecule (e.g., "A2B").
        degree (str): The degree of the polynomial expansion.

    Returns:
        tuple: A tuple containing two callables: ``(f_monomials, f_polynomials)``.
    """
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
    """Parses a comma-separated string into a list of strings.

    Used primarily as a helper for command-line argument parsing to convert
    inputs like "A2B,A3" into ``['A2B', 'A3']``.

    Args:
        arg (str): A comma-separated string.

    Returns:
        list: A list of substrings split by comma.
    """
    return arg.split(',')


def main():
    parser = argparse.ArgumentParser(description="run msa files")
    parser.add_argument("--mol-list", type=list_of_strings, required=True)
    parser.add_argument("--poly-list", type=list_of_strings, required=True)
    args = parser.parse_args()

    mols = args.mol_list
    poly_degrees = args.poly_list


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
            msa_path = os.getcwd()
            msa_path = os.path.join(msa_path, 'msa_files')
            filename = 'MOL_' + mol_sym + f'_{poly_degree}'
            msa_path = os.path.join(msa_path,  'molecule_' + moli)
            print(filename)
            print(msa_path)
            print(os.getcwd())


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

