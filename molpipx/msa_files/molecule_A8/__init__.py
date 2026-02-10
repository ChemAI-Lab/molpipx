def __getattr__(name):
    if name == "f_mono_p_3":
        from .monomials_MOL_8_3 import f_monomials as f_mono_p_3
        return f_mono_p_3
    if name == "f_mono_p_4":
        from .monomials_MOL_8_4 import f_monomials as f_mono_p_4
        return f_mono_p_4
    if name == "f_poly_p_3":
        from .polynomials_MOL_8_3 import f_polynomials as f_poly_p_3
        return f_poly_p_3
    if name == "f_poly_p_4":
        from .polynomials_MOL_8_4 import f_polynomials as f_poly_p_4
        return f_poly_p_4
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_functions():
    f_mono_p_3 = __getattr__("f_mono_p_3")
    f_mono_p_4 = __getattr__("f_mono_p_4")
    f_poly_p_3 = __getattr__("f_poly_p_3")
    f_poly_p_4 = __getattr__("f_poly_p_4")

    return {
        'poly_8_3': f_poly_p_3,
        'mono_8_3': f_mono_p_3,
        'poly_8_4': f_poly_p_4,
        'mono_8_4': f_mono_p_4,
    }
