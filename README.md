<h1 align='center'>PIPJAX</h1>
<h2 align='center'>Differentiable version of Permutationally Invariant Polynomial (PIP) models in JAX, compatible with Flax and the JAX-ecosystem. </h2>

<<<<<<< HEAD
**PIPx** is a [JAX](https://jax.readthedocs.io/en/latest/)-based library that provides an implementation of PIP models compatible with,
1. [FLAX](flax.readthedocs.io/): Nueral network library.
=======
**PIPJAX** is a [JAX](https://jax.readthedocs.io/en/latest/)-based library that provides an implementation of PIP models compatible with,
1. [FLAX](flax.readthedocs.io/): Neural network library.
>>>>>>> c5279d36fc6112b26067e0502f8c7524ed4a4f82
2. GPU friendly.
3. Fully differentiable.

## MSA files to JAX ##
This library translates the [MSA](https://scholarblogs.emory.edu/bowman/msa/) files, specifically the ``_file_.MONO`` and ``_file_.POLY`` files to the corresponding JAX version, ``_file_mono.py`` and ``_file_poly.py``. 
The MSA files must be generated before, for more information please see https://github.com/szquchen/MSA-2.0


**MSA References:**
* Xie, Z.; Bowman, J.M. Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization. J. Chem. Theory Comput. 2010, 6, 26-34.


### MSA-JAX files generation ###
```python
import jax
import pipx

head_files = 'MOL_<info>_<deg>'
path = '<path_to_the_files>'
label = '<file_label'
pipx.msa_file_generator()
```


The structure of the library is kept simple, as each molecular system could need individual elements. 
