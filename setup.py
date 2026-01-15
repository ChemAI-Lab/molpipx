"""Setup script for MOLPIPx"""

import os
from setuptools import find_packages
from setuptools import setup

setup(
    name='molpipx',
    packages=find_packages(),
    version='0.1',
    description='Permutationally Invariant Polynomials in JAX',
    authors=['Rodrigo. A. Vargas-Hernandez', 'Manuel Drehwald', 'Asma Jamali'],
    
    include_package_data=True, 
    package_data={
        'molpipx': ['data/*.xyz', 'data/*.npy', 'data/*/*.xyz', 'data/*/*.npy'],
    },
 
    install_requires=[
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
        'ml-collections',
        'gpjax',
    ],
    python_requires='>=3.10',
    keywords="jax, pip, computational chemistry, pes, potential energy surface, force field",
)