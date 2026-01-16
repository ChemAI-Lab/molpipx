PIP-NN Tutorial
===============

In this tutorial, we will train a Permutationally Invariant Polynomial Neural Network (PIP-NN) using the **Methane** dataset.

This model uses PIPs as the input layer to a standard Multi-Layer Perceptron (MLP), ensuring that the learned potential energy surface respects molecular symmetry.

Configuration
-------------

We define the architecture of the neural network and the training hyperparameters.

.. code-block:: python

   # Hyperparameters
   MOLECULE = 'A4B'        # Methane symmetry
   POLY_DEGREE = 3         # PIP degree
   
   # Network Architecture
   N_LAYERS = 2            # Hidden layers
   N_NEURONS = 128         # Neurons per layer
   features = (N_NEURONS,) * N_LAYERS  # (128, 128)
   
   # Training Settings
   LEARNING_RATE = 2e-3
   BATCH_SIZE = 128
   NUM_EPOCHS = 100
   N_TR = 1000             # Training samples
   N_VAL = 1000            # Validation samples

Data Loading
------------

We load the Methane dataset using the built-in loader and split it into training and validation sets.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import optax
   from flax.training import train_state
   from flax import linen as nn

   # MOLPIPx Imports
   from molpipx import PIPNN
   from molpipx import mse_loss
   from molpipx.pip_generator import get_functions, detect_molecule
   from molpipx.utils_training import split_train_and_test_data
   from molpipx.data import load_methane

.. code-block:: python

   # 1. Load Data
   X_all, _, y_all, atoms = load_methane(energy_normalization=False)
   
   # 2. Split Data
   key = jax.random.PRNGKey(0)
   (X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
       X_all, y_all, N_TR, key, N_VAL
   )


Model Initialization
--------------------

We generate the basis functions and initialize the ``PIPNN`` model. We also create a ``TrainState`` which holds the parameters and the optimizer state.

.. code-block:: python

   # 1. Generate Basis Functions
   f_mono, f_poly = get_functions(MOLECULE, POLY_DEGREE)
   
   # 2. Initialize Model
   pipnn = PIPNN(f_mono, f_poly, features)
   
   # Initialize parameters with dummy input
   na = 5
   x0_dummy = jnp.ones((1, na, 3))
   params = pipnn.init(key, x0_dummy)['params']
   
   # 3. Create Train State (Optimizer + Params)
   tx = optax.adam(LEARNING_RATE)
   state = train_state.TrainState.create(
       apply_fn=pipnn.apply, 
       params=params, 
       tx=tx
   )

Training Loop
-------------

We define the training step using ``jax.jit`` for speed. This function computes gradients of the MSE loss and updates the model parameters.

.. code-block:: python

   @jax.jit
   def train_step(state, batch_x, batch_y):
       """Computes gradients and updates the model."""
       
       def loss_fn(params):
           # Predict
           e_pred = state.apply_fn({'params': params}, batch_x)
           # Compute Loss
           loss = mse_loss(e_pred, batch_y)
           return loss

       # Compute Gradients
       grad_fn = jax.value_and_grad(loss_fn)
       loss, grads = grad_fn(state.params)
       
       # Update Parameters
       new_state = state.apply_gradients(grads=grads)
       return new_state, loss
   
   for epoch in range(1, NUM_EPOCHS + 1):
       # Shuffle Data
       key, input_rng = jax.random.split(key)
       perms = jax.random.permutation(input_rng, len(X_tr))
       
       epoch_loss = []
       
       # Mini-batch Training
       for i in range(0, len(X_tr), BATCH_SIZE):
           idx = perms[i:i+BATCH_SIZE]
           batch_x, batch_y = X_tr[idx], y_tr[idx]
           
           # Run one step
           state, loss = train_step(state, batch_x, batch_y)
           epoch_loss.append(loss)
       
       # Print progress every 10 epochs
       if epoch % 10 == 0:
           mean_loss = jnp.mean(jnp.array(epoch_loss))
           print(f"Epoch {epoch}: Loss = {mean_loss:.6f}")

Evaluation and Forces
---------------------

Once trained, we can evaluate the model. PIP-NNs are fully differentiable, so we can compute forces (gradients of energy w.r.t positions) automatically.

.. code-block:: python

   from molpipx.utils_gradients import get_energy_and_forces

   # 1. Predict Energy on Validation Set
   # Note: apply_fn expects a dictionary for params
   y_pred_val = state.apply_fn({'params': state.params}, X_val)
   val_loss = mse_loss(y_pred_val, y_val)
   
   print(f"Final Validation Loss: {val_loss:.6f}")

   # 2. Predict Forces (Differentiable)
   y_pred, f_pred = get_energy_and_forces(
       state.apply_fn, 
       X_val[:5],           # Take first 5 samples
       state.params         # Pass trained parameters
   )

Training with Forces (Gradients)
--------------------------------

We can significantly improve the accuracy of the potential energy surface by training on both energies and forces. Since forces are the negative gradient of energy with respect to positions (:math:`F = -\nabla E`), we can include them in the loss function.

.. code-block:: python

   from molpipx.utils_gradients import get_energy_and_forces
   
   # 1. Update Loss Function
   # We introduce a weighting factor 'l0' to balance energy and force errors
   # Loss = l0 * MSE(Energy) + Norm(Forces_pred - Forces_true)
   
   L0_WEIGHT = 1.0  # Weight for energy term

   @jax.jit
   def train_step_with_forces(state, batch_x, batch_f, batch_y):
       
       def loss_fn(params):
           # Use helper to get Energy AND Forces
           # We use state.apply_fn which is set to 'get_energy_and_forces' wrapper below
           e_pred, f_pred = state.apply_fn(params, batch_x)
           
           # Expand dims for broadcasting if needed
           e_pred = jnp.expand_dims(e_pred, axis=1)
           
           # Compute Individual Losses
           loss_e = mse_loss(e_pred, batch_y)
           loss_f = jnp.linalg.norm(f_pred - batch_f)
           
           # Total Weighted Loss
           total_loss = L0_WEIGHT * loss_e + loss_f
           return total_loss, (loss_e, loss_f)

       # Compute Gradients
       grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
       (loss, (l_e, l_f)), grads = grad_fn(state.params)
       
       # Update
       new_state = state.apply_gradients(grads=grads)
       return new_state, loss, l_e, l_f

   # 2. Re-Initialize Train State
   # Important: When training with forces, the model's apply function needs to output 
   # both energy and gradients.
   
   @jax.jit
   def forward_with_grad(params, geoms):
       return get_energy_and_forces(pipnn.apply, geoms, params)

   state_forces = train_state.TrainState.create(
       apply_fn=forward_with_grad,  # Use the wrapper
       params=params, 
       tx=optax.adam(LEARNING_RATE)
   )

   # 3. Load Data with Forces
   X_all, F_all, y_all, atoms = load_methane(energy_normalization=False)
   (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
       X_all, F_all, y_all, N_TR, key, N_VAL
   )
   
   # Simple Training Loop
   for epoch in range(1, NUM_EPOCHS + 1):
       key, input_rng = jax.random.split(key)
       perms = jax.random.permutation(input_rng, len(X_tr))
       
       for i in range(0, len(X_tr), BATCH_SIZE):
           idx = perms[i:i+BATCH_SIZE]
           
           # One step using Energy (y), Forces (F), and Geometry (x)
           state_forces, loss, l_e, l_f = train_step_with_forces(
               state_forces, 
               X_tr[idx], 
               F_tr[idx], 
               y_tr[idx]
           )

       if epoch % 10 == 0:
           print(f"Epoch {epoch}: Total Loss={loss:.4f} (E={l_e:.4f}, F={l_f:.4f})")
   