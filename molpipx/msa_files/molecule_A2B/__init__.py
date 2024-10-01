from .monomials_MOL_2_1_3 import f_monomials as f_mono_p_3
from .monomials_MOL_2_1_4 import f_monomials as f_mono_p_4
from .monomials_MOL_2_1_5 import f_monomials as f_mono_p_5
from .monomials_MOL_2_1_6 import f_monomials as f_mono_p_6
from .monomials_MOL_2_1_7 import f_monomials as f_mono_p_7
from .monomials_MOL_2_1_8 import f_monomials as f_mono_p_8


from .polynomials_MOL_2_1_3 import f_polynomials as f_poly_p_3
from .polynomials_MOL_2_1_4 import f_polynomials as f_poly_p_4
from .polynomials_MOL_2_1_5 import f_polynomials as f_poly_p_5
from .polynomials_MOL_2_1_6 import f_polynomials as f_poly_p_6
from .polynomials_MOL_2_1_7 import f_polynomials as f_poly_p_7
from .polynomials_MOL_2_1_8 import f_polynomials as f_poly_p_8


def get_functions():

    return {
        'poly_2_1_3': f_poly_p_3,
        'mono_2_1_3': f_mono_p_3,
        'poly_2_1_4': f_poly_p_4,
        'mono_2_1_4': f_mono_p_4,
        'poly_2_1_5': f_poly_p_5,
        'mono_2_1_5': f_mono_p_5,
        'poly_2_1_6': f_poly_p_6,
        'mono_2_1_6': f_mono_p_6,
        'poly_2_1_7': f_poly_p_7,
        'mono_2_1_7': f_mono_p_7,
        'poly_2_1_8': f_poly_p_8,
        'mono_2_1_8': f_mono_p_8
    }
