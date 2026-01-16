Generating MSA Files
====================

This library translates the MSA files, specifically the ``_file_.MONO`` and ``_file_.POLY`` files, to the corresponding JAX versions, ``_file_mono.py`` and ``_file_poly.py``.

The MSA files must be generated beforehand using the `MSA Algorithm <https://scholarblogs.emory.edu/bowman/msa/>`_. For more information and the source code, please see `MSA-2.0 <https://github.com/szquchen/MSA-2.0>`_.

**MSA References:**

    Xie, Z.; Bowman, J.M. Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization. *J. Chem. Theory Comput.* **2010**, 6, 26-34.

Molecule Structure Notation
---------------------------

The algorithm supports molecules with the structure

.. math::

   A_i B_j C_k D_l E_m

where the subscripts (:math:`i, j, k...`) represent the number of atoms of a specific type.


**Examples:**

* **Methane:** :math:`A_4 B` (4 atoms of type A, 1 atom of type B).
* **HCN:** :math:`A B C` (1 atom each of types A, B, and C).

Monomial and polynomial files generator for an ABC2D3EF molecule
----------------------------------------------------------------

To generate the monomial and polynomial files in a format compatible with JAX and Rust, users can use the ``msa_file_generator`` function from the ``molpipx`` library.

Below is an example for Ethanol, represented as :math:`A B C_2 D_3 E F`, for generating monomial and polynomial functions translated from MSA:

.. code-block:: python

    from molpipx import msa_file_generator

    # The base name of the MSA files (e.g., MOL_1_1_2_3_1_1_2.MONO)
    msa_head = 'MOL_1_1_2_3_1_1_2'
    
    # Label to append to the generated python files
    label = '1_1_2_3_1_1_2'
    
    # Path to the directory containing the MSA files
    path = 'Ethanol/'
    
    msa_file_generator(msa_head, path, label)