from .monomials_MOL_6_3 import f_monomials as f_mono_p_3
from .monomials_MOL_6_4 import f_monomials as f_mono_p_4
from .monomials_MOL_6_5 import f_monomials as f_mono_p_5
from .monomials_MOL_6_6 import f_monomials as f_mono_p_6

from .polynomials_MOL_6_3 import f_polynomials as f_poly_p_3
from .polynomials_MOL_6_4 import f_polynomials as f_poly_p_4
from .polynomials_MOL_6_5 import f_polynomials as f_poly_p_5
from .polynomials_MOL_6_6 import f_polynomials as f_poly_p_6


def get_functions():

    return {
        'poly_6_3': f_poly_p_3,
        'mono_6_3': f_mono_p_3,
        'poly_6_4': f_poly_p_4,
        'mono_6_4': f_mono_p_4,
        'poly_6_5': f_poly_p_5,
        'mono_6_5': f_mono_p_5,
        'poly_6_6': f_poly_p_6,
        'mono_6_6': f_mono_p_6,
    }
