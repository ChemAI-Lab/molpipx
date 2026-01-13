# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# docs/source/conf.py

import os
import sys
from unittest.mock import MagicMock
import sphinx_rtd_theme

# 1. Point to your project directory
sys.path.insert(0, os.path.abspath("../.."))

# --- FAKE LIBRARY SETUP START ---
# This block mocks JAX/Flax so Sphinx can build without installing them.
# It includes specific fixes for float(), imports, and attributes.

# 1. Define a smarter Fake Jax Numpy
class MockJaxNumpy(MagicMock):
    def exp(self, x): return 2.71828
    def sqrt(self, x): return 1.0
    def mean(self, x): return 0.0
    def sum(self, *args, **kwargs): return 0.0
    def abs(self, x): return x
    def reshape(self, *args, **kwargs): return self 
    def transpose(self, *args, **kwargs): return self

class FakeJaxModule(MagicMock):
    @staticmethod
    def jit(fun=None, *args, **kwargs):
        # If called as @jit or partial(jit, args), it might receive 'fun'.
        # If called as @jit(args), 'fun' might be None.
        def wrapper(f): return f
        
        if fun is not None and callable(fun):
            return fun  # Return the function unchanged (preserves docstring)
        else:
            return wrapper # Return a dummy decorator

    @staticmethod
    def vmap(fun=None, *args, **kwargs):
        def wrapper(f): return f
        if fun is not None and callable(fun):
            return fun
        else:
            return wrapper

    @staticmethod
    def jacrev(fun, *args, **kwargs): 
        # Return a dummy function to satisfy imports/calls inside functions
        return lambda *a, **k: MagicMock()
    
    @staticmethod
    def value_and_grad(fun, *args, **kwargs):
        # Return a function that returns (val, grad) tuple
        return lambda *a, **k: (0.0, 0.0)

# 2. Define fake 'struct' for flax
class FakeStruct:
    @staticmethod
    def dataclass(cls): return cls
    @staticmethod
    def field(*args, **kwargs): return None

# 3. Define a fake 'linen' module
class FakeLinen:
    class Module:
        def __init__(self, *args, **kwargs): pass
        def setup(self): pass
        def __call__(self, *args, **kwargs): pass
        def param(self, name, init_fn, *args): return init_fn 

    @staticmethod
    def jit(obj): return obj
    
    @staticmethod
    def compact(obj): return obj

    @staticmethod
    def vmap(target, *args, **kwargs): return target
    
    @staticmethod
    def softplus(x): return x

    @staticmethod
    def tanh(x): return x

    class Dense(Module): pass

    class initializers:
        @staticmethod
        def constant(value, dtype=None):
            return lambda *args: value
        @staticmethod
        def zeros(*args, **kwargs): return 0
        @staticmethod
        def ones(*args, **kwargs): return 1

# 4. Assemble the Fake Flax and JAX Module
fake_jax = FakeJaxModule()
fake_jax.numpy = MockJaxNumpy()
fake_jax.random = MagicMock()
fake_jax.lax = MagicMock()

fake_flax = MagicMock()
fake_flax.linen = FakeLinen
fake_flax.struct = FakeStruct

# 5. Inject them into system modules
# sys.modules['jax'] = MagicMock()
# sys.modules['jax.numpy'] = MockJaxNumpy()
# sys.modules['jax.random'] = MagicMock()

sys.modules['jax'] = fake_jax
sys.modules['jax.numpy'] = fake_jax.numpy
sys.modules['jax.random'] = fake_jax.random
sys.modules['jax.lax'] = fake_jax.lax

sys.modules['flax'] = fake_flax
sys.modules['flax.linen'] = FakeLinen
sys.modules['flax.struct'] = FakeStruct

# 6. Standard Mocks for other libraries
autodoc_mock_imports = [
    'numpy',
    # 'jax', 'flax', # Handled manually above
    'jaxlib',
    'gpjax',
    'optax',
    'scipy',
    'matplotlib',
    'rdkit',
    'pyscf',
    'pandas',
    'jaxtyping',
    'beartype'
]
# --- FAKE LIBRARY SETUP END ---

project = 'MOLPIPx'
copyright = '2025'
author = 'Rodrigo A. Vargas-Hernandez, Asma Jamali, Manuel S. Drehwald'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo', 
    'sphinx.ext.viewcode', 
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.mathjax',
]
# --- NAPOLEON SETTINGS ---
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True                # <--- THIS IS THE FIX (Was default/False)
napoleon_use_param = True
napoleon_use_rtype = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = "_static/molpipx_logo.png"

html_theme_options = {
    'logo_only': False,
    'navigation_depth': 2,         
    'collapse_navigation': True,   
    'style_nav_header_background': '#ffffff',
}
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]