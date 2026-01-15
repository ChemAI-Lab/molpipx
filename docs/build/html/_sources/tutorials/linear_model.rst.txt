Linear PIP Tutorial
===================

In this tutorial, we will learn how to train a Linear Permutationally Invariant Polynomial (PIP) model from scratch using the **Methane** dataset.

Configuration
-------------

We define the hyperparameters for our training. In a script, these might come from command-line arguments, but here we define them directly.

.. code-block:: python

   # Configuration parameters
   cfg = config_dict.ConfigDict()
   cfg.molecule = 'A4B'      # Methane symmetry
   cfg.poly_degree = 3       # Polynomial degree
   cfg.ntr = 1000            # Number of training points
   cfg.nval = 1000           # Number of validation points
   cfg.seed = 0              # Random seed

Data Loading
------------
.. code-block:: python

   import jax
   import jax.numpy as jnp
   from ml_collections import config_dict

   # MOLPIPx imports
   from molpipx import EnergyPIP, PIPlayer as PIP
   from molpipx import training, flax_params, mse_loss
   from molpipx.pip_generator import get_functions, detect_molecule
   from molpipx.utils_training import split_train_and_test_data
   
   # Import the built-in data loader
   from molpipx.data import load_methane
   
First, We need to load the Methane geometry and energy data.

.. code-block:: python

   # 1. Load Data
   # Returns: (Geometries, Forces, Energies, Atoms)
   X_all, _, y_all, atoms = load_methane(energy_normalization=False)

   # 2. Split into Training and Validation sets
   rng = jax.random.PRNGKey(cfg.seed)
   _, key = jax.random.split(rng)

   (X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
       X_all, y_all, cfg.ntr, key, cfg.nval
   )

Model Initialization
--------------------

We need to generate the basis functions (monomials and polynomials) specific to the molecule's symmetry before initializing the model.

.. code-block:: python

   # Detect molecule symmetry and generate functions
   mol_dict, mol_sym = detect_molecule(cfg.molecule)
   f_mono, f_poly = get_functions(cfg.molecule, cfg.poly_degree)

   # Initialize the PIP Model
   # We use 'PIP' for the model structure and 'EnergyPIP' for inference
   model_pip = PIP(f_mono, f_poly)
   model_energy = EnergyPIP(f_mono, f_poly)

   # Initialize parameters using a single data point
   params = model_energy.init(key, X_tr[:1])

Training
--------

Since this is a linear model, we can solve for the optimal weights directly using the ``training`` function (Linear Least Squares).

.. code-block:: python

   # Solve for weights 'w'
   w = training(model_pip, X_tr, y_tr)

   # Update the parameters with the optimized weights
   params = flax_params(w, params)
   

Evaluation
----------

Finally, we use the trained parameters to predict energies on the validation set and calculate the Mean Squared Error (MSE).

.. code-block:: python

   # Predict energies
   y_pred_tr = model_energy.apply(params, X_tr)
   y_pred_val = model_energy.apply(params, X_val)

   # Calculate Loss
   loss_tr = mse_loss(y_pred_tr, y_tr)
   loss_val = mse_loss(y_pred_val, y_val)


Training Utilities
------------------

For training linear models, MOLPIPx provides specialized functions to handle the linear least squares optimization. We use ``flax_params`` to handle the conversion between the raw optimized weights and the Flax Pytree structure.

.. code-block:: python

   from molpipx import training, flax_params

   # 1. Initialize the models
   model_pip = PIP(f_mono, f_poly)
   model_energy = EnergyPIP(f_mono, f_poly)
   
   # 2. Initialize parameters with dummy data
   params = model_energy.init(key, X_tr[:1])

   # 3. Run Training (Linear Solve)
   # Returns the optimized weights 'w' as a flat array
   w = training(model_pip, X_tr, y_tr)
   
   # 4. Update Parameters
   # The function flax_params() copies the parameters to the Pytree object used by Flax
   params_opt = flax_params(w, params)

Energy and Forces
-----------------

Given the flexibility of JAX, we can jointly compute the energy and the forces (gradients) efficiently. We provide the ``get_energy_and_forces`` utility for this purpose.

.. code-block:: python

   from molpipx.utils_gradients import get_energy_and_forces

   # Define the model for inference
   e_pip_model = EnergyPIP(f_mono, f_poly) 

   # Option 1: Predict Energy Only
   y_pred = e_pip_model.apply(params_opt, X_val) 

   # Option 2: Predict Energy and Forces Jointly
   # This computes the gradient of energy with respect to positions
   y_pred, f_pred = get_energy_and_forces(e_pip_model.apply, X_val, params_opt)