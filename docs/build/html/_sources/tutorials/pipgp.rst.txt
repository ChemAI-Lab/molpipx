PIPGP Tutorial
==============

In this tutorial, we show how to train a Gaussian Process (GP) model enhanced with Permutationally Invariant Polynomials (PIPGP) using the **Methane** dataset.

Configuration
-------------

We define the hyperparameters for the GP kernel and the training schedule.

.. code-block:: python

   # Hyperparameters
   MOLECULE = 'A4B'        # Methane symmetry
   POLY_DEGREE = 3         # PIP degree
   KERNEL_TYPE = 'Matern52'
   N_TR = 1000             # Training samples
   N_TST = 5000            # Test samples
   NUM_ITER = 800          # Optimization steps
   
   # Optimizer schedule settings
   INIT_LR = 0.0
   PEAK_LR = 0.01
   END_LR = 0.0
   WARMUP_STEPS = 75
   DECAY_STEPS = 700

Data Loading
------------
.. code-block:: python

   import jax
   import jax.numpy as jnp
   import gpjax as gpx
   import optax as ox
   from gpjax import Dataset

   # MOLPIPx Imports
   from molpipx import PIPlayerGP, PIPLayerKernel, get_forces_gp
   from molpipx.pip_generator import get_functions, detect_molecule
   from molpipx.utils_training import split_train_and_test_data
   from molpipx.data import load_methane

We load the Methane dataset and prepare it for ``GPJax``. Note that ``GPJax`` typically expects 2D inputs, so we reshape the geometry array ``(N, Atoms, 3)`` into ``(N, Atoms*3)``.

.. code-block:: python

   # 1. Load Data
   X_all, _, y_all, atoms = load_methane(energy_normalization=False)
   
   # 2. Split Data
   key = jax.random.PRNGKey(123)
   (X_tr, y_tr), (X_tst, y_tst) = split_train_and_test_data(
       X_all, y_all, N_TR, key, N_TST
   )

   # 3. Create GPJax Dataset
   # Reshape inputs to (N, Na * 3)
   X_tr_flat = X_tr.reshape(X_tr.shape[0], -1).astype(jnp.float64)
   y_tr_64 = y_tr.astype(jnp.float64)
   
   train_ds = Dataset(X=X_tr_flat, y=y_tr_64)

   # Prepare Test Data (Dictionary format for evaluation)
   X_tst_flat = X_tst.reshape(X_tst.shape[0], -1).astype(jnp.float64)
   tst_ds = {'x': X_tst_flat, 'e': y_tst.astype(jnp.float64)}

Model Initialization
--------------------

We initialize the PIP layer (feature extractor) and wrap it inside a ``PIPLayerKernel``. This kernel projects the atomic coordinates into the PIP space before applying the base GP kernel.

.. code-block:: python

   # 1. Generate Basis Functions
   f_mono, f_poly = get_functions(MOLECULE, POLY_DEGREE)
   
   # 2. Initialize PIP Layer (Feature Extractor)
   # We use a dummy input to initialize parameters
   na = 5
   x0_dummy = jnp.ones((1, na, 3))
   
   pipgp_layer = PIPlayerGP(f_mono, f_poly, trainable_l=False)
   params = pipgp_layer.init(key, x0_dummy)
   
   # Determine output dimension (number of polynomials)
   output = pipgp_layer.apply(params, x0_dummy)
   n_poly = output.shape[1]

   # 3. Define Base Kernel (Matern 5/2)
   base_kernel = gpx.kernels.Matern52(
       active_dims=list(range(n_poly)),
       lengthscale=jnp.ones((n_poly,))
   )

   # 4. Create the Composite PIP Kernel
   kernel = PIPLayerKernel(
       network=pipgp_layer, 
       base_kernel=base_kernel,
       key=key, 
       dummy_x=x0_dummy
   )

Gaussian Process Setup
----------------------

We construct the Gaussian Process by defining the prior (Mean + Kernel) and the likelihood.

.. code-block:: python

   # Zero Mean Function
   meanf = gpx.mean_functions.Zero()

   # Prior Distribution
   prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

   # Gaussian Likelihood
   likelihood = gpx.likelihoods.Gaussian(num_datapoints=train_ds.n)

   # Posterior
   posterior = prior * likelihood

Training
--------

We optimize the GP hyperparameters using ``optax`` and the negative log-marginal likelihood objective.

.. code-block:: python

   # 1. Define Optimizer Schedule
   schedule = ox.warmup_cosine_decay_schedule(
       init_value=INIT_LR,
       peak_value=PEAK_LR, 
       warmup_steps=WARMUP_STEPS,
       decay_steps=DECAY_STEPS, 
       end_value=END_LR
   )
   
   optimiser = ox.chain(
       ox.clip(1.0),
       ox.adamw(learning_rate=schedule),
   )

   # 2. Run Optimization
   print("Starting training...")
   
   opt_posterior, history = gpx.fit(
       model=posterior,
       objective=jax.jit(gpx.objectives.conjugate_mll),
       train_data=train_ds,
       optim=optimiser,
       num_iters=NUM_ITER,
       key=key,
       safe=False
   )

Prediction (Energy & Forces)
----------------------------

Finally, we predict the potential energy and forces on the test set.

.. code-block:: python

   # 1. Predict Energy (Mean and Std Dev)
   latent_dist = opt_posterior(tst_ds['x'], train_data=train_ds)
   predictive_dist = opt_posterior.likelihood(latent_dist)
   
   pred_mean = predictive_dist.mean
   pred_std = jnp.sqrt(predictive_dist.variance)

   # 2. Predict Forces (Gradients)
   # We use the helper function 'get_forces_gp' which handles gradients automatically
   forces_pred, _ = get_forces_gp(
       gp_model=opt_posterior, 
       train_data=train_ds, 
       x=tst_ds['x']
   )
