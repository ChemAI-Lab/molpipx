import pytest
import os
import numpy as np
from molpipx.pip_generator import f_monomial_flag_0, msa_file_generator


def test_f_monomial_flag_0():
    """Test the index parsing helper function."""
    # Case 1: Single 1 (Sum == 1)
    # Example: [0, 1, 0] -> Index 1
    arr1 = np.array([0, 1, 0])
    assert f_monomial_flag_0(arr1) == 1 # Returns index of the 1
    
    # Case 2: Multiple non-zeros (Sum > 0) 
    # Example: [2, 1, 0] -> [2, 1, 0]
    arr2 = np.array([2, 1, 0])
    assert f_monomial_flag_0(arr2) == [2, 1, 0] # Returns list of values (exponents)
    
    # Case 3: All zeros
    arr3 = np.array([0, 0, 0])
    assert f_monomial_flag_0(arr3) == -1 # Returns -1


def test_msa_file_generator(tmp_path):
    """
    Integration test that creates dummy .MONO and .POLY files,
    runs the generator, and verifies the output Python files.
    """
    # Setup dummy file paths
    mol_name = "MOL_test"
    mono_file = tmp_path / f"{mol_name}.MONO"
    poly_file = tmp_path / f"{mol_name}.POLY"
    
  
    ############# Create Dummy .MONO Content #################
    # Line 1: 0 1 0 0 -> Sum=1 (Index 0). Base distance r[0].
    # Line 2: 0 0 1 0 -> Sum=1 (Index 1). Base distance r[1].
    # Line 3: 0 0 0 1 -> Sum=1 (Index 2). Base distance r[2].
    # Line 4: 0 1 1 0 -> Sum=2 (Powers). r^1 * r^1 * r^0.
    # Line 5: 1 0 3   -> Product. mono_0 * mono_3.
    #########################################################
    
    mono_content = """0 1 0 0
0 0 1 0
0 0 0 1
0 1 1 0
1 0 3
"""
    mono_file.write_text(mono_content)
    
    ################### Create Dummy .POLY Content ###################
    # Line 1: 2 0 1 -> poly_0 = mono_0 + mono_1
    # Line 2: 3 0 0 -> poly_1 = poly_0 * poly_0 (Logic: A * B - C...)
    ##################################################################
    
    poly_content = """2 0 1
3 0 0
"""
    poly_file.write_text(poly_content)
    
    
    msa_file_generator(
        filename=mol_name, 
        path=str(tmp_path), # tmp_path is passed as the directory
        label=mol_name, 
        parents_path='molpipx' # parents_path is used for the import statement in the generated poly file.
    )
    

    py_mono = tmp_path / f"monomials_{mol_name}.py"
    py_poly = tmp_path / f"polynomials_{mol_name}.py"
    
    # Verify output files
    assert py_mono.exists(), "monomials_*.py was not created"
    assert py_poly.exists(), "polynomials_*.py was not created"
    
    # Verify Monomial File Content
    mono_text = py_mono.read_text()
    
    # Check Header/Imports
    assert "import jax" in mono_text
    assert "N_DISTANCES = 3" in mono_text # Because 3 lines had sum==1
    
    # Check generated functions
    # Line 0 (0 1 0 0) -> mono_0 = jnp.take(r,0)
    assert "mono_0 = jnp.take(r,0)" in mono_text
    
    # Line 3 (0 1 1 0) -> mono_3 = prod(power(r, [1, 1, 0]))
    assert "mono_3 = jnp.prod(jnp.power(r,jnp.array([1, 1, 0],dtype=jnp.int32)))" in mono_text
    
    # Line 4 (1 0 3) -> mono_4 = mono_0 * mono_3
    assert "mono_4 = mono_0 * mono_3" in mono_text
    
    # Verify Polynomial File Content
    poly_text = py_poly.read_text()
    
    # Check Imports
    assert f"from molpipx.monomials_{mol_name} import f_monomials as f_monos" in poly_text
    
    # Check generated functions
    # Line 0 (2 0 1) -> poly_0 = take(mono,0) + take(mono,1)
    assert "poly_0 = " in poly_text
    assert "jnp.take(mono,0) + jnp.take(mono,1)" in poly_text
    
    # Line 1 (3 0 0) -> poly_1 = poly_0 * poly_0
    assert "poly_1 = poly_0 * poly_0" in poly_text