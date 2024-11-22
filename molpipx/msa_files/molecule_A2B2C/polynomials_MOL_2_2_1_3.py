import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2B2C.monomials_MOL_2_2_1_3 import f_monomials as f_monos

# File created from ./MOL_2_2_1_3.POLY

N_POLYS = 102

# Total number of monomials = 102


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(102)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1) + jnp.take(mono, 2)
    poly_2 = jnp.take(mono, 3)
    poly_3 = jnp.take(mono, 4) + jnp.take(mono, 5)
    poly_4 = jnp.take(mono, 6) + jnp.take(mono, 7) + \
        jnp.take(mono, 8) + jnp.take(mono, 9)
    poly_5 = jnp.take(mono, 10)
    poly_6 = jnp.take(mono, 11)
    poly_7 = poly_2 * poly_1
    poly_8 = poly_1 * poly_3
    poly_9 = poly_2 * poly_3
    poly_10 = jnp.take(mono, 12)
    poly_11 = jnp.take(mono, 13) + jnp.take(mono, 14) + \
        jnp.take(mono, 15) + jnp.take(mono, 16)
    poly_12 = poly_1 * poly_4 - poly_11
    poly_13 = poly_2 * poly_4
    poly_14 = jnp.take(mono, 17) + jnp.take(mono, 18) + \
        jnp.take(mono, 19) + jnp.take(mono, 20)
    poly_15 = jnp.take(mono, 21) + jnp.take(mono, 22)
    poly_16 = jnp.take(mono, 23) + jnp.take(mono, 24)
    poly_17 = poly_3 * poly_4 - poly_14
    poly_18 = jnp.take(mono, 25) + jnp.take(mono, 26)
    poly_19 = poly_5 * poly_1
    poly_20 = poly_2 * poly_5
    poly_21 = poly_5 * poly_3
    poly_22 = poly_5 * poly_4
    poly_23 = poly_1 * poly_1 - poly_6 - poly_6
    poly_24 = poly_2 * poly_2
    poly_25 = poly_3 * poly_3 - poly_10 - poly_10
    poly_26 = poly_4 * poly_4 - poly_18 - poly_16 - \
        poly_15 - poly_18 - poly_16 - poly_15
    poly_27 = poly_5 * poly_5
    poly_28 = poly_2 * poly_6
    poly_29 = poly_6 * poly_3
    poly_30 = poly_2 * poly_8
    poly_31 = poly_10 * poly_1
    poly_32 = poly_2 * poly_10
    poly_33 = poly_6 * poly_4
    poly_34 = poly_2 * poly_11
    poly_35 = poly_2 * poly_12
    poly_36 = jnp.take(mono, 27) + jnp.take(mono, 28) + \
        jnp.take(mono, 29) + jnp.take(mono, 30)
    poly_37 = poly_1 * poly_14 - poly_36
    poly_38 = poly_2 * poly_14
    poly_39 = poly_1 * poly_15
    poly_40 = poly_2 * poly_15
    poly_41 = jnp.take(mono, 31) + jnp.take(mono, 32)
    poly_42 = poly_1 * poly_16 - poly_41
    poly_43 = poly_2 * poly_16
    poly_44 = poly_3 * poly_11 - poly_36
    poly_45 = poly_1 * poly_17 - poly_44
    poly_46 = poly_2 * poly_17
    poly_47 = poly_10 * poly_4
    poly_48 = poly_3 * poly_15
    poly_49 = poly_3 * poly_16
    poly_50 = poly_1 * poly_18
    poly_51 = poly_2 * poly_18
    poly_52 = jnp.take(mono, 33) + jnp.take(mono, 34)
    poly_53 = jnp.take(mono, 35) + jnp.take(mono, 36) + \
        jnp.take(mono, 37) + jnp.take(mono, 38)
    poly_54 = poly_3 * poly_18 - poly_52
    poly_55 = poly_5 * poly_6
    poly_56 = poly_2 * poly_19
    poly_57 = poly_5 * poly_8
    poly_58 = poly_2 * poly_21
    poly_59 = poly_5 * poly_10
    poly_60 = poly_5 * poly_11
    poly_61 = poly_5 * poly_12
    poly_62 = poly_2 * poly_22
    poly_63 = poly_5 * poly_14
    poly_64 = poly_5 * poly_15
    poly_65 = poly_5 * poly_16
    poly_66 = poly_5 * poly_17
    poly_67 = poly_5 * poly_18
    poly_68 = poly_6 * poly_1
    poly_69 = poly_2 * poly_23
    poly_70 = poly_2 * poly_7
    poly_71 = poly_3 * poly_23
    poly_72 = poly_2 * poly_9
    poly_73 = poly_1 * poly_25
    poly_74 = poly_2 * poly_25
    poly_75 = poly_10 * poly_3
    poly_76 = poly_1 * poly_11 - poly_33
    poly_77 = poly_1 * poly_12 - poly_33
    poly_78 = poly_2 * poly_13
    poly_79 = poly_3 * poly_14 - poly_47
    poly_80 = poly_3 * poly_17 - poly_47
    poly_81 = poly_4 * poly_11 - poly_50 - poly_41 - poly_39 - poly_41
    poly_82 = poly_1 * poly_26 - poly_81
    poly_83 = poly_2 * poly_26
    poly_84 = poly_4 * poly_14 - poly_52 - poly_49 - poly_48 - poly_52
    poly_85 = poly_4 * poly_15 - poly_53
    poly_86 = poly_4 * poly_16 - poly_53
    poly_87 = poly_3 * poly_26 - poly_84
    poly_88 = poly_4 * poly_18 - poly_53
    poly_89 = poly_5 * poly_23
    poly_90 = poly_2 * poly_20
    poly_91 = poly_5 * poly_25
    poly_92 = poly_5 * poly_26
    poly_93 = poly_5 * poly_19
    poly_94 = poly_2 * poly_27
    poly_95 = poly_5 * poly_21
    poly_96 = poly_5 * poly_22
    poly_97 = poly_1 * poly_23 - poly_68
    poly_98 = poly_2 * poly_24
    poly_99 = poly_3 * poly_25 - poly_75
    poly_100 = poly_4 * poly_26 - poly_88 - poly_86 - poly_85
    poly_101 = poly_5 * poly_27

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
                      poly_71,    poly_72,    poly_73,    poly_74,    poly_75,
                      poly_76,    poly_77,    poly_78,    poly_79,    poly_80,
                      poly_81,    poly_82,    poly_83,    poly_84,    poly_85,
                      poly_86,    poly_87,    poly_88,    poly_89,    poly_90,
                      poly_91,    poly_92,    poly_93,    poly_94,    poly_95,
                      poly_96,    poly_97,    poly_98,    poly_99,    poly_100,
                      poly_101,])

    return poly
