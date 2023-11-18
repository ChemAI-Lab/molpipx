import timeit
import jax 
import jax.numpy as jnp 
import numpy as np
from jax import grad, jit, jacfwd, value_and_grad, hessian
from polynomials_ethanol import f_polynomials, N_POLYS
from monomials_ethanol import N_XYZ, N_ATOMS

def read_xyz(p, N, use_grads):
    # python version of the rust code below
    with open(p, 'r') as f:
        # First, we need to figure out how many rows we need.
        num_atoms = int(f.readline().strip())
        f.readline() # skip f_energy
        elems_in_line = len(f.readline().split())
        if use_grads:
            # Atom, xyz, grad_xyz
            assert elems_in_line == 7
        else:
            # Atom, xyz, (grad_xyz ignored)
            assert elems_in_line == 4 or elems_in_line == 7
        num_lines = len(f.readlines()) + 3 # for previous readline()
        print("num_lines: ", num_lines)
        print("num_atoms: ", num_atoms)
        assert num_lines % (num_atoms + 2) == 0
        num_examples = num_lines / (num_atoms + 2) # N \leq num_lines
        assert N <= num_examples, "The file: {}\n only contains {} entries, but you requested {}!".format(p, num_examples, N)
        n_elems =  N * num_atoms * 6 if use_grads else N * num_atoms * 3 
        mat = np.zeros(n_elems)
        energies = np.zeros((N, 1))
        # continue 
        # Phew, finally done. Now let's actually read the file.
        f.seek(0)
        for i in range(N):
            n_atoms = int(f.readline().strip())
            assert n_atoms == num_atoms
            energy = float(f.readline().strip())
            energies[i] = energy
            for j in range(num_atoms):
                line = f.readline().split()[1:]
                line = [float(x) for x in line]
                pos = i * 3 * num_atoms + 3 * j
                mat[pos:pos + 3] = line[0:3]
        if use_grads:
            energies = np.concatenate((energies, grads), axis=1)
        return (mat, energies, n_atoms)

@jit
def _fb(xi):
  n_atoms = xi.shape[0]
  #print("_fb, n_atoms: ", n_atoms)
  z = xi[:, None] - xi[None, :]  # compute all difference
  i0 = jnp.triu_indices(
    n_atoms, 1
  )  #     select upper diagonal (LEXIC ORDER)
  diff = z[i0]
  r = jnp.linalg.norm(diff, axis=1)  # compute the bond length
  return r 

@jit
def f(x, l):
    z = _fb(x)
    z_morse = jnp.exp(-l*z)
    z = f_polynomials(z_morse)
    return z

@jit
def f_energy(xyz,w,l):
    z = _fb(xyz)
    z_morse = jnp.exp(-l*z)
    z = f_polynomials(z_morse)
    return jnp.vdot(w,z)

N = 50

#x = jnp.array([[0.00000000,  0.00000000,  14.00307401],
#               [0.00000000,  -1.98144397, 1.00782504],
#               [-1.71598081, 0.99072198,  1.00782504],
#               [1.71598081,  0.99072198,  1.00782504]])

#print(y)
#print(jacfwd(f_energy,argnums=(0))(x,w,1.))
#print()
#print(hessian(f_energy,argnums=(0))(x,w,1.))

def main():
    use_grads = False
    ethanol_path = "/h/344/drehwald/prog/msa_data/DDEthanol/Train_data_50000.xyz"
    (mat, b, n_atoms) = read_xyz(ethanol_path, N, use_grads)
    mat = mat.reshape((N * n_atoms, 3))
    m = N 
    if use_grads:
        m += N_XYZ*N
    n = N_POLYS
    print("m: ", m)
    print("n: ", n)
    print("len(mat): ", mat.shape)
    print("len(b): ", b.shape)
    print("N_XYZ: ", N_XYZ)
    print("N_POLYS: ", N_POLYS)
    assert len(b) == m
    print("energy: ", f_energy(mat, jnp.ones(N_POLYS), 1.))
    print("Allocating: {} GB for A", m * n * 4 / 1024 / 1024 / 1024)
    A = np.zeros((m, n));
    for i in range(N):
        if i % 10 == 0:
            print("i: ", i)
        molecule = mat[i*N_ATOMS:(i+1)*N_ATOMS, :]
        polys = f(molecule, 1.0)
        if (i == 0):
            print("molecule: {:?}", molecule)
        for j in range(N_POLYS):
            A[(i, j)] = polys[j]
        assert i < N
    print("Computed Polys and energy")

    pseudoinv = np.linalg.pinv(A)
    print("pseudoinv: ", pseudoinv.shape)
    x_min = np.dot(pseudoinv, b)
    b_hat = np.dot(A, x_min)
    for i in range(N_XYZ):
        print("b[{}]: {}, b_hat[{}]: {}".format(i, b[i], i, b_hat[i]))



main()
