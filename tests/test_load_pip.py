import pytest
import sys
import os
from unittest.mock import patch, MagicMock, ANY
from molpipx.load_pip import parse_molecule, detect_molecule, list_of_strings, get_functions

def test_parse_molecule():
    """
    Tests logic for parsing molecule strings into atom counts and symmetry strings.
    Covers basic cases, complex cases, and validation errors.
    """
    #### Basic Cases ####
    # A2B -> A=2, B=1. Symmetry string: 2_1
    res, sym = parse_molecule("A2B")
    assert res['A'] == 2
    assert res['B'] == 1
    assert sym == "2_1"

    # AB -> A=1, B=1. Symmetry string: 1_1
    res, sym = parse_molecule("AB")
    assert res['A'] == 1
    assert res['B'] == 1
    assert sym == "1_1"

    # A -> A=1. Symmetry string: 1
    res, sym = parse_molecule("A")
    assert res['A'] == 1
    assert res['B'] == 0
    assert sym == "1"

    ##### Complex Cases #####
    # A2B3C -> A=2, B=3, C=1. Sym: 2_3_1
    res, sym = parse_molecule("A2B3C")
    assert res['A'] == 2
    assert res['B'] == 3
    assert res['C'] == 1
    assert sym == "2_3_1"

    ##### Validation/Error Cases ####
    # "B" starts without "A"
    assert parse_molecule("B") == "Error: Elements are out of order or missing"
    
    # "AC" skips "B"
    assert parse_molecule("AC") == "Error: Elements are out of order or missing"
    
    # "BA" is out of order
    assert parse_molecule("BA") == "Error: Elements are out of order or missing"


def test_detect_molecule():
    """
    Tests the wrapper function for detecting/parsing single strings or lists of strings.
    """
    # Single string input
    res, sym = detect_molecule("A2B")
    assert sym == "2_1"

    # List input
    inputs = ["A2B", "AB"]
    results = detect_molecule(inputs)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0][1] == "2_1"
    assert results[1][1] == "1_1"

    # Invalid input type
    assert detect_molecule(123) == "Invalid input type"


def test_list_of_strings():
    """Tests the helper for comma-separated string parsing."""
    assert list_of_strings("A,B,C") == ["A", "B", "C"]
    assert list_of_strings("A2B,A3") == ["A2B", "A3"]
    assert list_of_strings("Single") == ["Single"]


def test_get_functions():
    """
    Tests get_functions logic using both mocks (for logic verification) 
    and real execution.
    """
    
    ##### Mocked Logic Verification #####    
    with patch('molpipx.load_pip.importlib.util.spec_from_file_location') as mock_spec_from_file, \
         patch('molpipx.load_pip.importlib.util.module_from_spec') as mock_module_from_spec:
        
        # Setup Mock Modules
        mock_poly_module = MagicMock()
        mock_poly_module.f_polynomials = "poly_function_placeholder"
        
        mock_mono_module = MagicMock()
        mock_mono_module.f_monomials = "mono_function_placeholder"
        
        # Setup Mock Spec
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_spec_from_file.return_value = mock_spec
        
        # Sequential returns: first for poly, second for mono
        mock_module_from_spec.side_effect = [mock_poly_module, mock_mono_module]

        # Call Function
        f_mono, f_poly = get_functions("A2B", "2")
        
        # Assertions
        assert f_mono == "mono_function_placeholder"
        assert f_poly == "poly_function_placeholder"
        
        # Verify paths
        args_called = mock_spec_from_file.call_args_list
        assert len(args_called) == 2
        poly_call_path = str(args_called[0][0][1])
        mono_call_path = str(args_called[1][0][1])
        assert "polynomials_MOL_2_1_2.py" in poly_call_path
        assert "monomials_MOL_2_1_2.py" in mono_call_path

    ##### Mocked Failure Case ######
    with patch('molpipx.load_pip.importlib.util.spec_from_file_location', return_value=None):
        with pytest.raises(KeyError):
            get_functions("A2B", "99")

    ###### Attempts to load actual Methane (A4B) ######
    try:
        # A4B -> "4_1". Degree 3 is commonly used.
        f_mono, f_poly = get_functions("A4B", "3")
        assert callable(f_mono)
        assert callable(f_poly)
    except (KeyError, ImportError, FileNotFoundError):
        print("\nNote: Actual MSA files for A4B degree 3 not found. Integration check skipped.")