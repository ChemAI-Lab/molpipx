import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2.monomials_MOL_2_2_4 import f_monomials as f_monos

# File created from ./MOL_2_2_4.POLY

N_POLYS = 75

# Total number of monomials = 75


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(75)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = jnp.take(mono, 2) + jnp.take(mono, 3) + \
        jnp.take(mono, 4) + jnp.take(mono, 5)
    poly_3 = jnp.take(mono, 6)
    poly_4 = poly_1 * poly_2
    poly_5 = jnp.take(mono, 7) + jnp.take(mono, 8)
    poly_6 = jnp.take(mono, 9) + jnp.take(mono, 10)
    poly_7 = jnp.take(mono, 11) + jnp.take(mono, 12)
    poly_8 = poly_1 * poly_3
    poly_9 = poly_3 * poly_2
    poly_10 = poly_1 * poly_1
    poly_11 = poly_2 * poly_2 - poly_7 - poly_6 - poly_5 - poly_7 - poly_6 - poly_5
    poly_12 = poly_3 * poly_3
    poly_13 = poly_1 * poly_5
    poly_14 = poly_1 * poly_6
    poly_15 = poly_1 * poly_7
    poly_16 = jnp.take(mono, 13) + jnp.take(mono, 14) + \
        jnp.take(mono, 15) + jnp.take(mono, 16)
    poly_17 = poly_1 * poly_9
    poly_18 = poly_3 * poly_5
    poly_19 = poly_3 * poly_6
    poly_20 = poly_3 * poly_7
    poly_21 = poly_1 * poly_4
    poly_22 = poly_1 * poly_11
    poly_23 = poly_2 * poly_5 - poly_16
    poly_24 = poly_2 * poly_6 - poly_16
    poly_25 = poly_2 * poly_7 - poly_16
    poly_26 = poly_1 * poly_8
    poly_27 = poly_3 * poly_11
    poly_28 = poly_1 * poly_12
    poly_29 = poly_3 * poly_9
    poly_30 = poly_1 * poly_10
    poly_31 = poly_2 * poly_11 - poly_25 - poly_24 - poly_23
    poly_32 = poly_3 * poly_12
    poly_33 = poly_1 * poly_16
    poly_34 = jnp.take(mono, 17)
    poly_35 = poly_1 * poly_18
    poly_36 = poly_1 * poly_19
    poly_37 = poly_1 * poly_20
    poly_38 = poly_3 * poly_16
    poly_39 = poly_1 * poly_13
    poly_40 = poly_1 * poly_14
    poly_41 = poly_1 * poly_15
    poly_42 = poly_1 * poly_23
    poly_43 = poly_1 * poly_24
    poly_44 = poly_5 * poly_6
    poly_45 = poly_1 * poly_25
    poly_46 = poly_5 * poly_7
    poly_47 = poly_6 * poly_7
    poly_48 = poly_1 * poly_17
    poly_49 = poly_1 * poly_27
    poly_50 = poly_3 * poly_23
    poly_51 = poly_3 * poly_24
    poly_52 = poly_3 * poly_25
    poly_53 = poly_1 * poly_29
    poly_54 = poly_3 * poly_18
    poly_55 = poly_3 * poly_19
    poly_56 = poly_3 * poly_20
    poly_57 = poly_1 * poly_21
    poly_58 = poly_1 * poly_22
    poly_59 = poly_5 * poly_5 - poly_34 - poly_34
    poly_60 = poly_6 * poly_6 - poly_34 - poly_34
    poly_61 = poly_7 * poly_7 - poly_34 - poly_34
    poly_62 = poly_1 * poly_31
    poly_63 = poly_5 * poly_11 - poly_47
    poly_64 = poly_6 * poly_11 - poly_46
    poly_65 = poly_7 * poly_11 - poly_44
    poly_66 = poly_1 * poly_26
    poly_67 = poly_3 * poly_31
    poly_68 = poly_1 * poly_28
    poly_69 = poly_3 * poly_27
    poly_70 = poly_1 * poly_32
    poly_71 = poly_3 * poly_29
    poly_72 = poly_1 * poly_30
    poly_73 = poly_2 * poly_31 - poly_65 - poly_64 - poly_63
    poly_74 = poly_3 * poly_32

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
                      poly_46,    poly_47,    poly_48,    poly_49,    poly_50,
                      poly_51,    poly_52,    poly_53,    poly_54,    poly_55,
                      poly_56,    poly_57,    poly_58,    poly_59,    poly_60,
                      poly_61,    poly_62,    poly_63,    poly_64,    poly_65,
                      poly_66,    poly_67,    poly_68,    poly_69,    poly_70,
                      poly_71,    poly_72,    poly_73,    poly_74,])

    return poly
