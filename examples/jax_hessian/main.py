import timeit
import jax 
import jax.numpy as jnp 
from jax import grad, jit, jacfwd, value_and_grad, hessian
from poly import f_polynomials

def _fb(xi):
  n_atoms = xi.shape[0]
  z = xi[:, None] - xi[None, :]  # compute all difference
  i0 = jnp.triu_indices(
    n_atoms, 1
  )  #     select upper diagonal (LEXIC ORDER)
  diff = z[i0]
  r = jnp.linalg.norm(diff, axis=1)  # compute the bond length
  return r 

def f_energy(xyz,w,l):
    z = _fb(xyz)
    z_morse = jnp.exp(-l*z)
    z = f_polynomials(z_morse)
    return jnp.vdot(w,z)

x = jnp.array([[0.00000000,  0.00000000,  14.00307401],
               [0.00000000,  -1.98144397, 1.00782504],
               [-1.71598081, 0.99072198,  1.00782504],
               [1.71598081,  0.99072198,  1.00782504]])

w = jnp.ones(51)
y = f_energy(x,w,1.)
print(y)
print(jacfwd(f_energy,argnums=(0))(x,w,1.))
print()
print(hessian(f_energy,argnums=(0))(x,w,1.))
