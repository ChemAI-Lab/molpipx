### PIPGP Training tutorial ###
In this tutorial we show how to train a PIP Gaussian Process from the scratch.<br>
We will use methane as an example. 

The ```main.py``` file contains the following ```argparse```:

```python

def parse_args():
    parser = argparse.ArgumentParser(description="Configure training parameters.")

    # Adding arguments for each configuration option
    parser.add_argument('--workdir', type=str, default=None, help='Working directory.')
    parser.add_argument('--kernel_type', type=str, default='Matern52', help='The type of kernel to use in the Gaussian Process.')
    parser.add_argument('--decay_steps', type=int, default=700, help=' Positive integer, the total length of the schedule.')
    parser.add_argument('--peak_value', type=float, default=0.01, help='Peak value for scalar to be annealed at end of warmup.')
    parser.add_argument('--warmup_steps', type=int, default=75, help='Positive integer, the length of the linear warmup.')
    parser.add_argument('--end_value', type=float, default=0.0, help='End value of the scalar to be annealed.')
    parser.add_argument('--trainable_l', type=bool, default=False, help=' trainable morse variables length scale parameter.')
    parser.add_argument('--molecule', type=str, default='A4B', help='Type of molecule.')
    parser.add_argument('--poly_degree', type=int, default=3, help='Degree of the polynomial.')
    parser.add_argument('--ntr', type=int, default=1000, help='Number of training examples.')
    parser.add_argument('--ntst', type=int, default=5000, help='Number of test examples.')
    parser.add_argument('--num_iter', type=int, default=800, help='Number of iterations.')
    

    return parser.parse_args()


```

The main training code can be executed from terminal,
```bash
python main.py --workdir /directory/path
```
Another hyper-parameters from the default one can be considered, for example, a GP model can be trained using Matern5/2 kernel and 5000 points,
```bash
python main.py --workdir /directory/path --kernel_type 'Matern52' --ntr 5000
```

The code will save the training history (loss values) to CSV file: ```his_{kernel_type}_{n_tr}.csv```.


The main training code is contained in the ```train.py``` file, where for methane the ```PIPGP``` model initialized the following way,

```python
from pipx import PIPlayerGP
from pipx import PIPLayerKernel
from pipx import get_functions, detect_molecule
import gpjax as gpx
import optax as ox

molecule_type = 'A4B' # molecular symmetry
na = 5 #number of atoms
poly_degree = 3 # polynomial degree for PIP
kernel_type = 'Matern52' # kernel type

# load f_mono and f_poly functions from PIPx
f_mono, f_poly = get_functions(molecule_type, poly_degree)

x0 = jnp.ones([1, na, 3]) #dummy geometry for methane
key = jax.random.key(123)

# Create pipgp_layer 
pipgp_layer = PIPlayerGP(f_mono, f_poly)
params = pipgp_layer.init(key, x0)
output = pipgp_layer.apply(params, x0)
n_poly = output.shape[1]

# PIPGP model in GPJAX
kernel = kernel_init[kernel_type](active_dims=list(range(n_poly)),
                                                     lengthscale=jnp.ones((n_poly,)),)

    
kernel = PIPLayerKernel(network = pipgp_layer, base_kernel = kernel,
                        key = key, dummy_x = x0)
    
mean = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function = mean, kernel = kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints = X_train.n)
posterior = prior * likelihood
```
Next, we define the optimizer according to its input parameters and proceed with predicting the energy and standard deviation on the test data:
```python
# Define the optimizer
schedule = ox.warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps,
                                            decay_steps, end_value)
optimiser = ox.chain(ox.clip(1.0),ox.adamw(learning_rate = schedule),)

# Start training process
opt_posterior, history = gpx.fit(model = posterior,
                                objective = jax.jit(gpx.objectives.ConjugateML (negative=True)),
                                train_data = X_train, optim = optimiser, num_iters = num_iter, key = key,safe = False)

# Energy prediction (mean and standard deviation)
latent_dist = opt_posterior(x_tst['x'], train_data = x_tr)
pred_dist = opt_posterior.likelihood(latent_dist)
pred_mean = prede_dist.mean()
pred_std = pred_dist.stddev()
```


Given the flexibility of JAX, we can compute the force using ```jax.value_and_grad```:

```python
from pipx import get_forces_gp

forces, _ = get_forces_gp(gp_model=opt_posterior, train_data=x_tr, x=x_tst['x'])
```

See [train_and_evaluate](train.py) for training a PIPGP using GPJAX.