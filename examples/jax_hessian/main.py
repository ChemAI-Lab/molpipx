import timeit
import jax 
import jax.numpy as jnp 
from jax import grad, jit, jacfwd, value_and_grad, hessian

def f_monomials(r): 

    mono = jnp.zeros(33) 

    mono_0 = 1. 
    mono_1 = jnp.take(r,5) 
    mono_2 = jnp.take(r,4) 
    mono_3 = jnp.take(r,3) 
    mono_4 = jnp.take(r,2) 
    mono_5 = jnp.take(r,1) 
    mono_6 = jnp.take(r,0) 
    mono_7 = mono_1 * mono_2 
    mono_8 = mono_1 * mono_3 
    mono_9 = mono_2 * mono_3 
    mono_10 = mono_3 * mono_4 
    mono_11 = mono_2 * mono_5 
    mono_12 = mono_1 * mono_6 
    mono_13 = mono_4 * mono_5 
    mono_14 = mono_4 * mono_6 
    mono_15 = mono_5 * mono_6 
    mono_16 = mono_1 * mono_9 
    mono_17 = mono_1 * mono_10 
    mono_18 = mono_2 * mono_10 
    mono_19 = mono_1 * mono_11 
    mono_20 = mono_3 * mono_11 
    mono_21 = mono_2 * mono_12 
    mono_22 = mono_3 * mono_12 
    mono_23 = mono_2 * mono_13 
    mono_24 = mono_3 * mono_13 
    mono_25 = mono_1 * mono_14 
    mono_26 = mono_3 * mono_14 
    mono_27 = mono_1 * mono_15 
    mono_28 = mono_2 * mono_15 
    mono_29 = mono_4 * mono_15 
    mono_30 = mono_2 * mono_24 
    mono_31 = mono_1 * mono_26 
    mono_32 = mono_1 * mono_28 

#    stack all monomials 
    mono = jnp.stack([    mono_0,    mono_1,    mono_2,    mono_3,    mono_4,    mono_5, 
    mono_6,    mono_7,    mono_8,    mono_9,    mono_10, 
    mono_11,    mono_12,    mono_13,    mono_14,    mono_15, 
    mono_16,    mono_17,    mono_18,    mono_19,    mono_20, 
    mono_21,    mono_22,    mono_23,    mono_24,    mono_25, 
    mono_26,    mono_27,    mono_28,    mono_29,    mono_30, 
    mono_31,    mono_32,    ]) 

    return mono

def f_polynomials(r): 

    mono = f_monomials(r) 

    poly = jnp.zeros(51) 

    poly_0 = jnp.take(mono,0) 
    poly_1 = jnp.take(mono,1) + jnp.take(mono,2) + jnp.take(mono,3) 
    poly_2 = jnp.take(mono,4) + jnp.take(mono,5) + jnp.take(mono,6) 
    poly_3 = jnp.take(mono,7) + jnp.take(mono,8) + jnp.take(mono,9) 
    poly_4 = jnp.take(mono,10) + jnp.take(mono,11) + jnp.take(mono,12) 
    poly_5 = poly_1 * poly_2 - poly_4 
    poly_6 = jnp.take(mono,13) + jnp.take(mono,14) + jnp.take(mono,15) 
    poly_7 = poly_1 * poly_1 - poly_3 - poly_3 
    poly_8 = poly_2 * poly_2 - poly_6 - poly_6 
    poly_9 = jnp.take(mono,16) 
    poly_10 = jnp.take(mono,17) + jnp.take(mono,18) + jnp.take(mono,19) + jnp.take(mono,20) + jnp.take(mono,21) + jnp.take(mono,22) 
    poly_11 = poly_2 * poly_3 - poly_10 
    poly_12 = jnp.take(mono,23) + jnp.take(mono,24) + jnp.take(mono,25) + jnp.take(mono,26) + jnp.take(mono,27) + jnp.take(mono,28) 
    poly_13 = poly_1 * poly_6 - poly_12 
    poly_14 = jnp.take(mono,29) 
    poly_15 = poly_1 * poly_3 - poly_9 - poly_9 - poly_9 
    poly_16 = poly_1 * poly_4 - poly_10 
    poly_17 = poly_2 * poly_7 - poly_16 
    poly_18 = poly_2 * poly_4 - poly_12 
    poly_19 = poly_1 * poly_8 - poly_18 
    poly_20 = poly_2 * poly_6 - poly_14 - poly_14 - poly_14 
    poly_21 = poly_1 * poly_7 - poly_15 
    poly_22 = poly_2 * poly_8 - poly_20 
    poly_23 = poly_9 * poly_2 
    poly_24 = jnp.take(mono,30) + jnp.take(mono,31) + jnp.take(mono,32) 
    poly_25 = poly_3 * poly_6 - poly_24 
    poly_26 = poly_14 * poly_1 
    poly_27 = poly_9 * poly_1 
    poly_28 = poly_3 * poly_4 - poly_23 
    poly_29 = poly_1 * poly_10 - poly_23 - poly_28 - poly_23 
    poly_30 = poly_1 * poly_11 - poly_23 
    poly_31 = poly_1 * poly_12 - poly_25 - poly_24 - poly_24 
    poly_32 = poly_1 * poly_13 - poly_25 
    poly_33 = poly_4 * poly_5 - poly_25 - poly_31 
    poly_34 = poly_2 * poly_11 - poly_25 
    poly_35 = poly_4 * poly_6 - poly_26 
    poly_36 = poly_2 * poly_12 - poly_26 - poly_35 - poly_26 
    poly_37 = poly_2 * poly_13 - poly_26 
    poly_38 = poly_14 * poly_2 
    poly_39 = poly_3 * poly_3 - poly_27 - poly_27 
    poly_40 = poly_3 * poly_7 - poly_27 
    poly_41 = poly_1 * poly_16 - poly_28 
    poly_42 = poly_2 * poly_21 - poly_41 
    poly_43 = poly_1 * poly_18 - poly_33 
    poly_44 = poly_7 * poly_8 - poly_43 
    poly_45 = poly_6 * poly_6 - poly_38 - poly_38 
    poly_46 = poly_2 * poly_18 - poly_35 
    poly_47 = poly_1 * poly_22 - poly_46 
    poly_48 = poly_6 * poly_8 - poly_38 
    poly_49 = poly_1 * poly_21 - poly_40 
    poly_50 = poly_2 * poly_22 - poly_48 

#    stack all polynomials 
    poly = jnp.stack([    poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5, 
    poly_6,    poly_7,    poly_8,    poly_9,    poly_10, 
    poly_11,    poly_12,    poly_13,    poly_14,    poly_15, 
    poly_16,    poly_17,    poly_18,    poly_19,    poly_20, 
    poly_21,    poly_22,    poly_23,    poly_24,    poly_25, 
    poly_26,    poly_27,    poly_28,    poly_29,    poly_30, 
    poly_31,    poly_32,    poly_33,    poly_34,    poly_35, 
    poly_36,    poly_37,    poly_38,    poly_39,    poly_40, 
    poly_41,    poly_42,    poly_43,    poly_44,    poly_45, 
    poly_46,    poly_47,    poly_48,    poly_49,    poly_50, 
    ]) 

    return poly

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
#test
print(jacfwd(f_energy,argnums=(0))(x,w,1.))
print()
print(hessian(f_energy,argnums=(0))(x,w,1.))


