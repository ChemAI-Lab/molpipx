import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B.monomials_MOL_2_1_6 import f_monomials as f_monos


# File created from /gpfs/fs1/home/r/ravh011/ravh011/PIPMSA_jax/pipjax/msa_files/molecule_A2B/MOL_2_1_6.POLY


N_POLYS = 50

# Total number of monomials = 50


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(50)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2)
    poly_2 = jnp.take(mono, 3)
    poly_3 = jnp.take(mono, 4)
    poly_4 = poly_2 * poly_1
    poly_5 = poly_1 * poly_1 - poly_3 - poly_3
    poly_6 = poly_2 * poly_2
    poly_7 = poly_2 * poly_3
    poly_8 = poly_3 * poly_1
    poly_9 = poly_2 * poly_5
    poly_10 = poly_2 * poly_4
    poly_11 = poly_1 * poly_5 - poly_8
    poly_12 = poly_2 * poly_6
    poly_13 = poly_2 * poly_8
    poly_14 = poly_2 * poly_7
    poly_15 = poly_3 * poly_3
    poly_16 = poly_3 * poly_5
    poly_17 = poly_2 * poly_11
    poly_18 = poly_2 * poly_9
    poly_19 = poly_2 * poly_10
    poly_20 = poly_1 * poly_11 - poly_16
    poly_21 = poly_2 * poly_12
    poly_22 = poly_2 * poly_15
    poly_23 = poly_2 * poly_16
    poly_24 = poly_2 * poly_13
    poly_25 = poly_2 * poly_14
    poly_26 = poly_3 * poly_8
    poly_27 = poly_3 * poly_11
    poly_28 = poly_2 * poly_20
    poly_29 = poly_2 * poly_17
    poly_30 = poly_2 * poly_18
    poly_31 = poly_2 * poly_19
    poly_32 = poly_1 * poly_20 - poly_27
    poly_33 = poly_2 * poly_21
    poly_34 = poly_2 * poly_26
    poly_35 = poly_2 * poly_27
    poly_36 = poly_2 * poly_22
    poly_37 = poly_2 * poly_23
    poly_38 = poly_2 * poly_24
    poly_39 = poly_2 * poly_25
    poly_40 = poly_3 * poly_15
    poly_41 = poly_3 * poly_16
    poly_42 = poly_3 * poly_20
    poly_43 = poly_2 * poly_32
    poly_44 = poly_2 * poly_28
    poly_45 = poly_2 * poly_29
    poly_46 = poly_2 * poly_30
    poly_47 = poly_2 * poly_31
    poly_48 = poly_1 * poly_32 - poly_42
    poly_49 = poly_2 * poly_33

#    stack all polynomials
    poly = jnp.stack([poly_0,    poly_1,    poly_2,    poly_3,    poly_4,    poly_5,
                      poly_6,    poly_7,    poly_8,    poly_9,    poly_10,
                      poly_11,    poly_12,    poly_13,    poly_14,    poly_15,
                      poly_16,    poly_17,    poly_18,    poly_19,    poly_20,
                      poly_21,    poly_22,    poly_23,    poly_24,    poly_25,
                      poly_26,    poly_27,    poly_28,    poly_29,    poly_30,
                      poly_31,    poly_32,    poly_33,    poly_34,    poly_35,
                      poly_36,    poly_37,    poly_38,    poly_39,    poly_40,
                      poly_41,    poly_42,    poly_43,    poly_44,    poly_45,
                      poly_46,    poly_47,    poly_48,    poly_49,])

    return poly
