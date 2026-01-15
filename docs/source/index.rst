MOLPIPx
=======

MOLPIPx seamlessly integrates Permutationally Invariant Polynomials (PIPs) with modern machine learning frameworks. Built on JAX and EnzymeAD-Rust, it enables efficient, differentiable modeling of potential energy surfaces using linear models, neural networks, and Gaussian processes.

.. image:: _static/molpipx_diagram.png
   :width: 500px
   :align: center
   :alt: MOLPIPx Diagram

Capabilities at a Glance
------------------------

MOLPIPx currently includes the following implementations:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Model Type
     - Energy
     - Forces
   * - Linear Models
     - ✓
     - ✓
   * - Neural Networks (Flax)
     - ✓
     - ✓
   * - Gaussian Processes (GP)
     - ✓
     - ✓
   * - Anisotropic Models
     - ✓
     - ✓

Check the :doc:`tutorials` for more information.

Documentation for the source code :doc:`can be found here <api>`. The full source code with examples and tests can be explored at `github <https://github.com/ChemAI-Lab/molpipx>`_.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   api
   publications
   citing
   about