import os
import numpy as np
import argparse
import math

# ----------------------------------------------------------------------------------------
#   READ MONOMIALS


def f_monomial_flag_0(x):
    """Parses the indices for a monomial term from the input array.

    Helper function that processes a line from the MSA file to determine the
    indices involved in a monomial term.

    Args:
        x (np.ndarray): Array of indices derived from the input file line.

    Returns:
        int or list: Returns an integer index if a single element is active,
        a list of indices if multiple are active, or -1 if none.
    """
    if np.sum(x) == 1:
        j = np.where(x == 1)[0]
        j = j[0].tolist()
    elif np.sum(x) > 0:
        j = x.tolist()
    else:
        j = -1
    return j


def create_f_monomials(file_mono: str, file_label: str):
    """Generates a Python file containing JAX-compiled monomial functions.

    Reads a ``.MONO`` file provided by the MSA software and writes a corresponding
    Python script (``monomials_{label}.py``). 
    Args:
        file_mono (str): Path to the source ``.MONO`` file.
        file_label (str): Label used to suffix the output filename.
    """

    f_out_monomials = 'monomials_{}.py'.format(file_label)
    f_out_monomials = os.path.join(_path, f_out_monomials)

    f_out = open(f_out_monomials, 'w+')
    f_out.write('import jax \n')
    f_out.write('import jax.numpy as jnp \n')
    f_out.write('from jax import jit\n')
    f_out.write('\n')
    f_out.write('# File created from {} \n'.format(file_mono))
    f_out.write('\n')

    f = open(file_mono, 'r')
    Lines = f.readlines()
    n_mono = len(Lines)
    index = 0

    for l in Lines:
        if l[0] == '0':
            z = np.array(l.split(), dtype=np.int32)
            if np.sum(z) == 1:
                index += 1

    zeros, ones = (Lines[:index], Lines[index:])
    offset = len(zeros)
    N_DISTANCES = index  # offset - 1
    f_out.write('# N_DISTANCES == N_ATOMS * (N_ATOMS - 1) / 2;\n')
    f_out.write('N_DISTANCES = {}\n'.format(N_DISTANCES))
    N_ATOMS = round(+0.5 + math.sqrt(1 + 4*2*N_DISTANCES)*0.5)
    f_out.write('N_ATOMS = {}\n'.format(N_ATOMS))
    f_out.write('N_XYZ = N_ATOMS * 3\n\n')
    f_out.close()

#     ----------------------------
#     MAIN
    f_out = open(f_out_monomials, 'a+')
    f_out.write('# Total number of monomials = {} \n'.format(n_mono))
    f_out.write('\n')
    f_out.write('@jit\n')
    f_out.write('def f_monomials(r): \n')
    f_out.write('    assert(r.shape == (N_DISTANCES,))\n')
    f_out.write('\n')
    f_out.write('    mono = jnp.zeros({}) \n'.format(n_mono))
    f_out.write('\n')

    for i, l in enumerate(Lines):
        z = l.split()
        z = np.array(z, dtype=int)
        x = z[1:]

#         FLAG = 0
        if z[0] == 0:
            j = f_monomial_flag_0(x)
            if isinstance(j, int):
                if j > -1:
                    f_out.write('    mono_{} = jnp.take(r,{}) \n'.format(i, j))
                else:
                    f_out.write('    mono_{} = 1. \n'.format(i))
            elif isinstance(j, list):
                f_out.write(
                    '    mono_{} = jnp.prod(jnp.power(r,jnp.array({},dtype=jnp.int32))) \n'.format(i, j))

#         FLAG = 1
        elif int(z[0]) == 1:
            a, b = x[0], x[1]
            f_out.write('    mono_{} = mono_{} * mono_{} \n'.format(i, a, b))

    f_out.write('\n')
    f_out.write('#    stack all monomials \n')
    f_out.write('    mono = jnp.stack([')
    for i, _ in enumerate(Lines[:]):
        f_out.write('    mono_{},'.format(i))
        if i % 5 == 0 and i > 0:
            f_out.write(' \n'.format(i))

    f_out.write('    ]) \n')
    f_out.write('\n')
    f_out.write('    return mono \n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.close()
    f.close()
# ----------------------------------------------------------------------------------------
#   READ POLYNOMIALS


def create_f_polynomials(file_poly: str, file_label: str, parents_path: str = ''):
    """Generates a Python file containing JAX-compiled polynomial functions.

    Reads a ``.POLY`` file provided by the MSA software and writes a corresponding
    Python script (``polynomials_{label}.py``).

    Args:
        file_poly (str): Path to the source ``.POLY`` file.
        file_label (str): Label used to suffix the output filename.
        parents_path (str): Dotted path to the parent package for import statements.
    """
    f_out_polynomials = 'polynomials_{}.py'.format(file_label)
    f_out_polynomials = os.path.join(_path, f_out_polynomials)

    f_out = open(f_out_polynomials, 'w+')
    f_out.write('import jax \n')
    f_out.write('import jax.numpy as jnp \n')
    f_out.write('from jax import jit\n')
    f_out.write('\n')
    f_out.write(
        'from {}.monomials_{} import f_monomials as f_monos \n'.format(parents_path, file_label))
    f_out.write('\n')
    f_out.write('\n')
    f_out.write('# File created from {} \n'.format(file_poly))
    f_out.write('\n')
    f_out.write('\n')

    f = open(file_poly, 'r')
    Lines = f.readlines()
    n_poly = len(Lines)
    f_out.write('N_POLYS = {}\n\n'.format(n_poly))
    f_out.close()

#     TEST
#     MAIN
    f_out = open(f_out_polynomials, 'a+')
    f_out.write('# Total number of monomials = {} \n'.format(n_poly))
    f_out.write('\n')
    f_out.write('@jit\n')
    f_out.write('def f_polynomials(r): \n')
    f_out.write('\n')
    f_out.write('    mono = f_monos(r.ravel()) \n'.format(n_poly))
    f_out.write('\n')
    f_out.write('    poly = jnp.zeros({}) \n'.format(n_poly))
    f_out.write('\n')

    for i, l in enumerate(Lines):
        z = l.split()
        z = np.array(z, dtype=int)
        x = z[1:]


#         FLAG = 2
        if z[0] == 2:
            str_ = '    poly_{} = '.format(i)
            for j, xi in enumerate(x):
                str_ += 'jnp.take(mono,{})'.format(xi)
                if j < x.shape[0] - 1:
                    str_ += ' + '

            f_out.write('{} \n'.format(str_))


#         FLAG = 3
        elif z[0] == 3:
            str_ = '    poly_{} = '.format(i)
            for j, xi in enumerate(x):
                str_ += 'poly_{}'.format(xi)

                if j == 0:
                    str_ += ' * '
                elif j < x.shape[0]-1:
                    str_ += ' - '

            f_out.write('{} \n'.format(str_))

#     -----------------------
    f_out.write('\n')
    f_out.write('#    stack all polynomials \n')
    f_out.write('    poly = jnp.stack([')
    for i, _ in enumerate(Lines[:]):
        f_out.write('    poly_{},'.format(i))
        if i % 5 == 0 and i > 0:
            f_out.write(' \n'.format(i))

    f_out.write('    ]) \n')
#     -----------------------
    f_out.write('\n')
    f_out.write('    return poly \n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.close()
    f.close()


# ----------------------------------------------------------------------------------------
def msa_file_generator(filename: str, path: str = './', label: str = None, parents_path: str = ''):
    """Orchestrates the generation of both monomial and polynomial JAX files from MSA input.

    This is the main entry point for generating the python code. It checks for the existence
    of ``.MONO`` and ``.POLY`` files based on the filename and calls the creation functions.

    Args:
        filename (str): The base name of the file (e.g., 'MOL_1_3_4').
        path (str): Directory containing the input files and where output will be saved.
        label (str, optional): Custom label for the output files. Defaults to ``filename``.
    """
    global _path
    _path = path
    f_head = os.path.join(_path, filename)
    if label is not None:
        f_label = filename
    else:
        f_label = label

    file_mono = '{}.MONO'.format(f_head)
    file_poly = '{}.POLY'.format(f_head)

    if not os.path.isfile(file_mono):
        print(file_mono)
        print('File {} does not exist!'.format(file_mono))
        assert 0
    if not os.path.isfile(file_poly):
        print('File {} does not exist!'.format(file_poly))
        assert 0

#     construct monomials
    create_f_monomials(file_mono, f_label)

#     construct polynomials
    create_f_polynomials(file_poly, f_label, parents_path)


def main():

    parser = argparse.ArgumentParser(
        description='MSA monomials and polynomials to JAX')
    parser.add_argument('--file', type=str,
                        default='MOL_1_3_4', help='head of the file')
    parser.add_argument('--path', type=str, default='.',
                        help='path to the file')
    parser.add_argument('--label', type=str, default='',
                        help='label for the file')
    args = parser.parse_args()

    global _path
    _path = args.path
    f_head = os.path.join(_path, args.file)
    f_label = args.label

    file_mono = '{}.MONO'.format(f_head)
    file_poly = '{}.POLY'.format(f_head)

    if not os.path.isfile(file_mono):
        print('File {} does not exist!'.format(file_mono))
        assert 0
    if not os.path.isfile(file_poly):
        print('File {} does not exist!'.format(file_poly))
        assert 0

#     construct monomials
    create_f_monomials(file_mono, f_label)

#     construct polynomials
    create_f_polynomials(file_poly, f_label)


if __name__ == "__main__":

    main()
