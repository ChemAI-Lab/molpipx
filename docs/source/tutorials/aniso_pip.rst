Aniso PIP Tutorial
==================

.. note::
   Please read the :doc:`linear_model` before proceeding with this one.

In this tutorial, we show how to train a linear **Anisotropic PIP** (AnisoPIP) from scratch.

Standard PIP models use a single length-scale parameter (:math:`\lambda`) for the Morse variables 
(:math:`\bar{\gamma} = e^{-\lambda r}`). However, in molecules like Methane, the **C-H** bond length is very different from the **H-H** distance. 

**AnisoPIP** allows us to learn a specific :math:`\lambda` for each unique type of atom pair while maintaining permutational invariance.

Configuration
-------------

.. code-block:: python

   # Hyperparameters
   MOLECULE = 'A4B'        # Methane
   POLY_DEGREE = 3
   N_TR = 1000
   N_VAL = 1000
   
   # Optimizer settings
   LEARNING_RATE = 2e-3
   NUM_EPOCHS = 100
   OPT_TOL = 1e-4          # Convergence tolerance

Data Loading & Masking
----------------------
.. code-block:: python

   import pandas as pd
   import jax
   import jax.numpy as jnp
   import optax
   from flax import linen as nn

   # MOLPIPx Imports
   from molpipx import LayerPIPAniso, EnergyPIPAniso
   from molpipx import get_mask, get_f_mask, lambda_random_init
   from molpipx import flax_params, mse_loss
   from molpipx.pip_generator import get_functions, detect_molecule
   from molpipx.utils_training import split_train_and_test_data
   from molpipx.data import load_methane

This is the key step for AnisoPIP. We must identify which atom pairs correspond to which interaction type (e.g., C-H vs H-H). We use ``get_mask`` to generate this mapping automatically.

.. code-block:: python

   # 1. Load Data
   X_all, _, y_all, atoms_list = load_methane(energy_normalization=False)
   
   # Take the atom types from the first configuration (assuming constant composition)
   atoms = atoms_list[0] 
   
   # 2. Generate Masks
   # mask: Maps every pair distance to a type index
   mask, unique_pairs = get_mask(atoms)
   n_pairs = mask.shape[0] # Number of unique interaction types
   
   # f_mask: A function that applies the correct lambda to each distance
   f_mask = get_f_mask(mask)

   print(f"Unique Interaction Types: {n_pairs}")
   print(f"Pairs detected: {unique_pairs}")

   # 3. Split Data
   key = jax.random.PRNGKey(0)
   (X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
       X_all, y_all, N_TR, key, N_VAL
   )

Model Initialization
--------------------

We initialize two models:
1. ``LayerPIPAniso``: Used to compute the PIP matrix (basis functions).
2. ``EnergyPIPAniso``: Used to predict energies.

.. code-block:: python

   # Generate basis functions
   f_mono, f_poly = get_functions(MOLECULE, POLY_DEGREE)

   # Initialize the PIP Layer (for computing basis functions)
   model_pip = LayerPIPAniso(f_mono, f_poly, f_mask, n_pairs)
   params_pip = model_pip.init(key, X_tr[:1])
   
   # Randomly initialize the lambda parameters
   params_pip = lambda_random_init(params_pip, key)
   
   # Initialize the Energy Model (for predictions)
   model_energy = EnergyPIPAniso(f_mono, f_poly, f_mask, n_pairs)
   params_energy = model_energy.init(key, X_tr[:1])
   
   print("Initial Parameters:", params_pip)

Nested Optimization Strategy
----------------------------

We use a **bi-level optimization** strategy:

1. **Inner Loop**: For a fixed set of :math:`\lambda` (length scales), the problem is linear. We solve for the optimal polynomial coefficients :math:`\mathbf{w}` using Linear Least Squares (``lstsq``).
2. **Outer Loop**: We use **Adam** to optimize the non-linear :math:`\lambda` parameters by minimizing the validation loss.

.. code-block:: python

   @jax.jit
   def validation_loss(params_pip, data, params_energy_template):
       (X_tr, y_tr), (X_val, y_val) = data

       # --- INNER LOOP: Solve Linear System ---
       # Compute PIP basis matrix for training set
       Pip_tr = model_pip.apply(params_pip, X_tr)
       
       # Solve P * w = y
       results = jnp.linalg.lstsq(Pip_tr, y_tr)
       w_opt = results[0] # Optimal weights
       
       # Update energy model parameters with new weights
       params_energy = flax_params(w_opt, params_energy_template)
       
       # --- OUTER LOOP: Evaluate on Validation Set ---
       y_val_pred = model_energy.apply(params_energy, X_val)
       loss_val = mse_loss(y_val_pred, y_val)
       
       # Also return training loss for logging
       loss_tr = mse_loss(model_energy.apply(params_energy, X_tr), y_tr)
       
       return loss_val, (params_energy, loss_tr)

Training Loop
-------------

We now run the outer optimization loop using Optax to tune the :math:`\lambda` parameters.

.. code-block:: python

   # Setup Optimizer
   optimizer = optax.adam(LEARNING_RATE)
   opt_state = optimizer.init(params_pip)

   # Gradient Function
   # We differentiate validation_loss w.r.t params_pip (lambdas)
   grad_fn = jax.value_and_grad(validation_loss, argnums=0, has_aux=True)

   @jax.jit
   def train_step(params_pip, opt_state, data):
       (loss_val, (params_e, loss_tr)), grads = grad_fn(
           params_pip, data, params_energy
       )
       
       updates, new_opt_state = optimizer.update(grads, opt_state, params_pip)
       new_params_pip = optax.apply_updates(params_pip, updates)
       
       return new_params_pip, new_opt_state, loss_val, loss_tr, params_e

   # Run Training
   data_tuple = ((X_tr, y_tr), (X_val, y_val))
   
   for epoch in range(1, NUM_EPOCHS + 1):
       params_pip, opt_state, loss_val, loss_tr, params_final = train_step(
           params_pip, opt_state, data_tuple
       )
       
       # Extract current lambda values (softplus ensures positivity)
       current_lambdas = nn.softplus(
           params_pip['params']['VmapJitPIPAniso_0']['lambda']
       )

       if epoch % 10 == 0:
           print(f"Epoch {epoch}: Val Loss={loss_val:.6f} | Lambdas={current_lambdas}")

   print("Training complete.")

Training with Forces
--------------------

We can also train the AnisoPIP model using both energies and forces. Since the model is linear with respect to the polynomial coefficients (:math:`E = \mathbf{P} \cdot \mathbf{w}`), the forces are simply the gradients of the basis functions multiplied by the same coefficients (:math:`F = -\nabla \mathbf{P} \cdot \mathbf{w}`).

We can solve for the optimal weights :math:`\mathbf{w}` by stacking the energy equations and the force equations into one large linear system:

.. math::

   \begin{bmatrix} \mathbf{P} \\ -\nabla \mathbf{P} \end{bmatrix} \mathbf{w} = \begin{bmatrix} \mathbf{y} \\ \mathbf{F} \end{bmatrix}

Here is how to implement this "Stack and Solve" strategy:

.. code-block:: python

   from molpipx import split_train_and_test_data_w_forces, get_pip_grad

   # 1. Load Data with Forces
   # Returns: (Geometries, Forces, Energies)
   (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
       X_all, F_all, y_all, N_TR, key, N_VAL
   )
   
   # We need dimensions for reshaping
   n_samples, n_atoms, _ = X_tr.shape
   data_w_forces = ((X_tr, F_tr, y_tr), (X_val, F_val, y_val))

   # 2. Define Loss with Forces
   @jax.jit
   def validation_loss_forces(params_pip, data, params_energy_template):
       (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = data

       # --- INNER LOOP: Stack and Solve ---
       def inner_solve(params_pip):
           # A. Compute Basis Functions (Energies) -> Shape: (N, n_poly)
           Pip_tr = model_pip.apply(params_pip, X_tr)
           n_poly = Pip_tr.shape[-1]
           
           # B. Compute Gradients of Basis Functions (Forces)
           # Returns shape: (N, n_atoms, 3, n_poly)
           Gpip_tr = get_pip_grad(model_pip.apply, X_tr, params_pip)
           
           # Flatten forces to match linear system rows: (N * Atoms * 3, n_poly)
           Gpip_tr_flat = Gpip_tr.reshape(n_samples * n_atoms * 3, n_poly)

           # C. Stack Matrices (The 'A' in Ax=b)
           # We stack Energy rows on top of Force rows
           A_matrix = jax.lax.concatenate((Pip_tr, Gpip_tr_flat), dimension=0)

           # D. Stack Targets (The 'b' in Ax=b)
           # Flatten target forces: (N * Atoms * 3, 1)
           F_tr_flat = F_tr.reshape(n_samples * n_atoms * 3, 1)
           b_vector = jax.lax.concatenate((y_tr, F_tr_flat), dimension=0)

           # E. Solve Linear System
           results = jnp.linalg.lstsq(A_matrix, b_vector)
           return results[0] # The optimal weights w

       w_opt = inner_solve(params_pip)
       
       # Update energy model
       params_energy = flax_params(w_opt, params_energy_template)
       
       # --- OUTER LOOP: Evaluate ---
       # We optimize lambda based on Validation Energy Error
       y_val_pred = model_energy.apply(params_energy, X_val)
       loss_val = mse_loss(y_val_pred, y_val)
       
       # Return training loss for logging
       loss_tr = mse_loss(model_energy.apply(params_energy, X_tr), y_tr)
       
       return loss_val, (params_energy, loss_tr)

   # 3. Run Optimization
   print("Starting training with forces...")
   grad_fn_forces = jax.value_and_grad(validation_loss_forces, argnums=0, has_aux=True)
   
   # Re-initialize optimizer state
   opt_state = optimizer.init(params_pip)

   for epoch in range(1, NUM_EPOCHS + 1):
       (loss_val, (params_e, loss_tr)), grads = grad_fn_forces(
           params_pip, data_w_forces, params_energy
       )
       
       updates, opt_state = optimizer.update(grads, opt_state, params_pip)
       params_pip = optax.apply_updates(params_pip, updates)

       if epoch % 10 == 0:
           # Check convergence of lambda parameters
           l_vals = nn.softplus(params_pip['params']['VmapJitPIPAniso_0']['lambda'])
           print(f"Epoch {epoch}: Val Loss={loss_val:.6f}, Lambdas={l_vals}")