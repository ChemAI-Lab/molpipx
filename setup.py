"""Setup script for PIPJAX"""

import os
from setuptools import find_packages
from setuptools import setup

setup(
    name='pipjax',
    packages=find_packages(),
    version='0.1',
    description='Permutationally Invariant Polynomials in JAX',
    author='Rodrigo. A. Vargas-Hernandez',
    install_requires=[
        # List your project's dependencies here,
        # e.g., 'requests>=2.23.0',
        'jax>0.4.14',
        'jaxlib>0.4.14',
        'numpy>=1.18.0',
        'chex>=0.1.7',
        'typing_extensions>=4.8.0',
        'jaxtyping',
        'flax',
        'pytest>=7.4.3',
        'jaxopt',
        'optax>0.1.7',
        'orbax-checkpoint>0.4.4',
        #'gpjax',
    ],
    python_requires='>=3.6',
    keywords="jax, pip, computational chemistry, pes, potential energy surface, force field",
)
