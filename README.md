<h1 align='center'><img src="https://github.com/ChemAI-Lab/molpipx/blob/main/Images/molpipx_logo.png?raw=true" alt="MOLPIPx Logo" width="400"></h1>
<h2 align='center'>Differentiable version of Permutationally Invariant Polynomial (PIP) models in JAX and Rust. 



[![DOI:10.1063/5.0250837](https://img.shields.io/badge/DOI-10.1063%2F5.0250837-blue)](https://doi.org/10.1063/5.0250837)

</h2>

**MOLPIPx** is a [JAX](https://jax.readthedocs.io/en/latest/)-based library that provides an implementation of PIP models compatible with,
1. [FLAX](flax.readthedocs.io/): Neural network library.
2. GPU friendly.
3. Fully differentiable.

## MSA files to JAX ##
This library translates the [MSA](https://scholarblogs.emory.edu/bowman/msa/) files, specifically the ``_file_.MONO`` and ``_file_.POLY`` files to the corresponding JAX version, ``_file_mono.py`` and ``_file_poly.py``. 
The MSA files must be generated before, for more information please see https://github.com/szquchen/MSA-2.0


**MSA References:**
* Xie, Z.; Bowman, J.M. Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization. J. Chem. Theory Comput. 2010, 6, 26-34.

## Installation ##
MOLPIPx requires Python 3.10 or later. We recommend installing it in a virtual environment to avoid conflicts.

```bash
python -m venv molpipx
source molpipx/bin/activate
```

The latest stable release is available through PyPI:
```bash
pip install molpipx
```
To install the latest development version from source, clone the source code from the GitHub and install with pip from local file:

 ```bash 
git clone https://github.com/ChemAI-Lab/molpipx.git
cd molpipx
pip install .
```

### MSA-JAX files generation ###
MOLPIPx package includes `msa_file_generator`, which translates monomial and polynomial files from MSA to JAX and Rust for molecules.
Check out an [example on generating msa files](https://github.com/ChemAI-Lab/molpipx/blob/edd3d052d15fdda9b24ad1c528870542073e6268/molpipx/data/README.md)


```python
from molpipx import msa_file_generator

head_files = 'MOL_<info>_<deg>'
path = '<path_to_the_files>'
label = '<file_label>'
msa_file_generator(head_files, path, label)
```


The structure of the library is kept simple, as each molecular system could need individual elements. 


## Models ##
MOLPIPx incorporated PIPs with three main regression models, i.e., linear regression, neural networks and Gaussian processes. This library leverages two main automatic differentiation engines, JAX for
The Python version and Enzyme-AD for the Rust version improve the simulation of a wide range of chemical systems.

<h1 align='center'><img src="https://github.com/ChemAI-Lab/molpipx/blob/main/Images/molpipx_diagram.png?raw=true" alt="diagram" width="700"></h1>


## Rust Version ##
The Rust version makes use of [std::autodiff](https://doc.rust-lang.org/nightly/std/autodiff/attr.autodiff.html), an experimental feature of Rust which is currently in the process of upstreaming.
While upstreaming is in progress, you will need to build our custom fork of Rust which already includes autodiff.
Instruction for how to do so are available [here](https://enzyme.mit.edu/index.fcgi/rust/installation.html).
Once upstreaming completed, you will be able to use any nightly Rust version.
This [tracking issue](https://github.com/rust-lang/rust/issues/124509) shows the progress in upstreaming the remaining autodiff pieces. 



## Tutorials ##
Check out our tutorials to get started with MOLPIPx. These tutorials define inputs for different regression approaches, train machine learning models with or without forces, and make predictions.

1. [Linear regression with permutationally invariant polynomials (Linear PIP)](https://github.com/ChemAI-Lab/molpipx/tree/edd3d052d15fdda9b24ad1c528870542073e6268/examples/linear_pip)
2. [Anisotropic linear regression with permutationally invariant polynomials (Anisotropic Linear PIP)](https://github.com/ChemAI-Lab/molpipx/tree/edd3d052d15fdda9b24ad1c528870542073e6268/examples/aniso_pip)
3. [Permutationally Invariant Polynomial Neural Networks (PIP-NN)](https://github.com/ChemAI-Lab/molpipx/tree/edd3d052d15fdda9b24ad1c528870542073e6268/examples/pipnn)
4. [Permutationally Invariant Polynomial Gaussian Process (PIP-GP)](https://github.com/ChemAI-Lab/molpipx/tree/edd3d052d15fdda9b24ad1c528870542073e6268/examples/pipgp)


# Reference
```latex
@article{molpipx,
    author = {Drehwald, Manuel S. and Jamali, Asma and Vargas-Hernández, Rodrigo A.},
    title = {MOLPIPx: An end-to-end differentiable package for permutationally invariant polynomials in Python and Rust},
    journal = {The Journal of Chemical Physics},
    volume = {162},
    number = {8},
    pages = {084115},
    year = {2025},
    month = {02},
    issn = {0021-9606},
    doi = {10.1063/5.0250837},
    url = {https://doi.org/10.1063/5.0250837},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0250837/20416060/084115_1_5.0250837.pdf},
}
```
