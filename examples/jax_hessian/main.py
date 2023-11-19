import timeit
import jax 
import jax.numpy as jnp 
import numpy as np
from jax import grad, jit, jacfwd, jacrev, value_and_grad, hessian
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
        n_elems = N * num_atoms * 3
        xyz = np.zeros(n_elems)
        energies = np.zeros(N)
        if use_grads:
            grads = np.zeros(n_elems)
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
                xyz[pos:pos + 3] = line[0:3]
                if use_grads:
                    grads[pos:pos + 3] = line[3:6]
        if use_grads:
            print("energies: ", energies.shape)
            print("grads: ", grads.shape)
            energies = np.concatenate((energies, grads))
        return (xyz, energies, n_atoms)

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
    assert(x.shape[0] == N_ATOMS)
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

# RODRIGO FACTOR
# How patient are you?
N = 50
use_grads = True
ethanol_path = "/h/344/drehwald/prog/msa_data/DDEthanol/Train_data_50000.xyz"

def main():
    (xyz, b, n_atoms) = read_xyz(ethanol_path, N, use_grads)
    # if use_grads, than b will first have N energies, then N * N_XYZ gradients
    # the gradients are in the x, y, z, x, y, z, ... order 
    # thus for the file above we have b[50:53] = [0.03725021, 0.00255395, -0.00297219]
    xyz = xyz.reshape((N * n_atoms, 3))
    m = N 
    n = N_POLYS
    print("m: ", m)
    print("n: ", n)
    print("xyz: ", xyz.shape)
    print("b: ", b.shape)
    print("N_XYZ: ", N_XYZ)
    print("N_POLYS: ", N_POLYS)
    print("Allocating: {} GB for A", m * n * 4 / 1024 / 1024 / 1024)
    A = np.zeros((m, n));
    if use_grads:
        dA = np.zeros((N * N_XYZ, n))
    for i in range(N):
        if i % 10 == 0:
            print(i,", ", end='')
        molecule = xyz[i*N_ATOMS:(i+1)*N_ATOMS, :]
        polys = f(molecule, 1.0)
        A[i, :] = polys
        if use_grads:
            molecule = xyz[i*N_ATOMS:(i+1)*N_ATOMS, :]
            tmp = jacrev(f, argnums=(0))(molecule, 1.0)
            assert(not np.allclose(tmp, 0.0))
            tmp = tmp.reshape((N_POLYS, N_XYZ)).transpose()
            dA[i*N_XYZ:(i+1)*N_XYZ, :] = tmp
    if use_grads:
        A = np.concatenate((A, dA))
    print("\nComputed Polys and energy")


    A2 = A[0:N, :]
    b2 = b[0:N]
    x2_min = np.linalg.lstsq(A2, b2, rcond=None)[0]
    b2_hat = np.dot(A2, x2_min)

    A3 = A[N:, :]
    b3 = b[N:]
    x3_min = np.linalg.lstsq(A3, b3, rcond=None)[0]
    b3_hat = np.dot(A3, x3_min)

    print("Computed gradients")

    print("A: ", A.shape)
    print("b: ", b.shape)
    # lstdq ret consists of a tuple of (x_min, residuals, rank, s)
    x_min = np.linalg.lstsq(A, b, rcond=None)[0]
    b_hat = np.dot(A, x_min)
    for i in range(N):
        print("b: ", b[i], " b_hat: ", b_hat[i], " b2_hat: ", b2_hat[i], " b3_hat: ", b3_hat[i])
    for i in range(N_POLYS):
        print("x_min: ", x_min[i], " x2_min: ", x2_min[i], " x3_min: ", x3_min[i])

if __name__ == "__main__":
    main()
