import jax
import jax.numpy as jnp
from jax import jit

from molpipx.msa_files.molecule_A2BC.monomials_MOL_2_1_1_4 import f_monomials as f_monos

# File created from ./MOL_2_1_1_4.POLY

N_POLYS = 120

# Total number of monomials = 120


@jit
def f_polynomials(r):

    mono = f_monos(r.ravel())

    poly = jnp.zeros(120)

    poly_0 = jnp.take(mono, 0)
    poly_1 = jnp.take(mono, 1)
    poly_2 = jnp.take(mono, 2) + jnp.take(mono, 3)
    poly_3 = jnp.take(mono, 4) + jnp.take(mono, 5)
    poly_4 = jnp.take(mono, 6)
    poly_5 = poly_1 * poly_2
    poly_6 = jnp.take(mono, 7)
    poly_7 = poly_1 * poly_3
    poly_8 = jnp.take(mono, 8) + jnp.take(mono, 9)
    poly_9 = jnp.take(mono, 10)
    poly_10 = poly_2 * poly_3 - poly_8
    poly_11 = poly_1 * poly_4
    poly_12 = poly_4 * poly_2
    poly_13 = poly_4 * poly_3
    poly_14 = poly_1 * poly_1
    poly_15 = poly_2 * poly_2 - poly_6 - poly_6
    poly_16 = poly_3 * poly_3 - poly_9 - poly_9
    poly_17 = poly_4 * poly_4
    poly_18 = poly_1 * poly_6
    poly_19 = poly_1 * poly_8
    poly_20 = poly_1 * poly_9
    poly_21 = poly_1 * poly_10
    poly_22 = poly_6 * poly_3
    poly_23 = poly_9 * poly_2
    poly_24 = poly_1 * poly_12
    poly_25 = poly_4 * poly_6
    poly_26 = poly_1 * poly_13
    poly_27 = poly_4 * poly_8
    poly_28 = poly_4 * poly_9
    poly_29 = poly_4 * poly_10
    poly_30 = poly_1 * poly_5
    poly_31 = poly_1 * poly_15
    poly_32 = poly_6 * poly_2
    poly_33 = poly_1 * poly_7
    poly_34 = poly_2 * poly_8 - poly_22
    poly_35 = poly_2 * poly_10 - poly_22
    poly_36 = poly_1 * poly_16
    poly_37 = poly_3 * poly_8 - poly_23
    poly_38 = poly_9 * poly_3
    poly_39 = poly_2 * poly_16 - poly_37
    poly_40 = poly_1 * poly_11
    poly_41 = poly_4 * poly_15
    poly_42 = poly_4 * poly_16
    poly_43 = poly_1 * poly_17
    poly_44 = poly_4 * poly_12
    poly_45 = poly_4 * poly_13
    poly_46 = poly_1 * poly_14
    poly_47 = poly_2 * poly_15 - poly_32
    poly_48 = poly_3 * poly_16 - poly_38
    poly_49 = poly_4 * poly_17
    poly_50 = poly_1 * poly_22
    poly_51 = poly_1 * poly_23
    poly_52 = poly_6 * poly_9
    poly_53 = poly_1 * poly_25
    poly_54 = poly_1 * poly_27
    poly_55 = poly_1 * poly_28
    poly_56 = poly_1 * poly_29
    poly_57 = poly_4 * poly_22
    poly_58 = poly_4 * poly_23
    poly_59 = poly_1 * poly_18
    poly_60 = poly_1 * poly_32
    poly_61 = poly_1 * poly_19
    poly_62 = poly_1 * poly_34
    poly_63 = poly_1 * poly_20
    poly_64 = poly_1 * poly_21
    poly_65 = poly_6 * poly_8
    poly_66 = poly_1 * poly_35
    poly_67 = poly_6 * poly_10
    poly_68 = poly_9 * poly_15
    poly_69 = poly_1 * poly_37
    poly_70 = poly_1 * poly_38
    poly_71 = poly_9 * poly_8
    poly_72 = poly_1 * poly_39
    poly_73 = poly_6 * poly_16
    poly_74 = poly_9 * poly_10
    poly_75 = poly_1 * poly_24
    poly_76 = poly_1 * poly_41
    poly_77 = poly_4 * poly_32
    poly_78 = poly_1 * poly_26
    poly_79 = poly_4 * poly_34
    poly_80 = poly_4 * poly_35
    poly_81 = poly_1 * poly_42
    poly_82 = poly_4 * poly_37
    poly_83 = poly_4 * poly_38
    poly_84 = poly_4 * poly_39
    poly_85 = poly_1 * poly_44
    poly_86 = poly_4 * poly_25
    poly_87 = poly_1 * poly_45
    poly_88 = poly_4 * poly_27
    poly_89 = poly_4 * poly_28
    poly_90 = poly_4 * poly_29
    poly_91 = poly_1 * poly_30
    poly_92 = poly_1 * poly_31
    poly_93 = poly_6 * poly_6
    poly_94 = poly_1 * poly_47
    poly_95 = poly_6 * poly_15
    poly_96 = poly_1 * poly_33
    poly_97 = poly_2 * poly_34 - poly_65
    poly_98 = poly_2 * poly_35 - poly_67
    poly_99 = poly_1 * poly_36
    poly_100 = poly_2 * poly_37 - poly_73
    poly_101 = poly_9 * poly_9
    poly_102 = poly_2 * poly_39 - poly_73
    poly_103 = poly_1 * poly_48
    poly_104 = poly_3 * poly_37 - poly_71
    poly_105 = poly_9 * poly_16
    poly_106 = poly_2 * poly_48 - poly_104
    poly_107 = poly_1 * poly_40
    poly_108 = poly_4 * poly_47
    poly_109 = poly_4 * poly_48
    poly_110 = poly_1 * poly_43
    poly_111 = poly_4 * poly_41
    poly_112 = poly_4 * poly_42
    poly_113 = poly_1 * poly_49
    poly_114 = poly_4 * poly_44
    poly_115 = poly_4 * poly_45
    poly_116 = poly_1 * poly_46
    poly_117 = poly_2 * poly_47 - poly_95
    poly_118 = poly_3 * poly_48 - poly_105
    poly_119 = poly_4 * poly_49

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
                      poly_101,    poly_102,    poly_103,    poly_104,    poly_105,
                      poly_106,    poly_107,    poly_108,    poly_109,    poly_110,
                      poly_111,    poly_112,    poly_113,    poly_114,    poly_115,
                      poly_116,    poly_117,    poly_118,    poly_119,])

    return poly
