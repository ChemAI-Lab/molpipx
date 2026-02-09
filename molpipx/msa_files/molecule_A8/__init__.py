from .monomials_MOL_8_3 import f_monomials as f_mono_p_3
from .monomials_MOL_8_4 import f_monomials as f_mono_p_4
from .monomials_MOL_8_5 import f_monomials as f_mono_p_5

from .polynomials_MOL_8_3 import f_polynomials as f_poly_p_3
from .polynomials_MOL_8_4 import f_polynomials as f_poly_p_4
from .polynomials_MOL_8_5 import f_polynomials as f_poly_p_5


def get_functions():

    return {
        'poly_8_3': f_poly_p_3,
        'mono_8_3': f_mono_p_3,
        'poly_8_4': f_poly_p_4,
        'mono_8_4': f_mono_p_4,
        'poly_8_5': f_poly_p_5,
        'mono_8_5': f_mono_p_5,
    }
