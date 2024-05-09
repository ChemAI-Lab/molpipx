### Linear PIP tutorial ###
In this tutorial we show how to train a linear PIP from scratch.<br>
We will use methane as an example. 

The ```main.py``` file contains the following ```argparse```,

```python

def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure training parameters.")

    # Adding arguments for each configuration option
    parser.add_argument('--workdir', type=str,
                        default=None, help='Working directory')
    parser.add_argument('--molecule', type=str,
                        default='A4B', help='Type of molecule.')
    parser.add_argument('--poly_degree', type=int, default=3,
                        help='Degree of the polynomial.')
    parser.add_argument('--ntr', type=int, default=1000,
                        help='Number of training examples.')
    parser.add_argument('--nval', type=int, default=1000,
                        help='Number of validation examples.')
    parser.add_argument('--ntst', type=int, default=2000,
                        help='Number of test examples.')
    parser.add_argument('--grad_bool', type=bool, default=False,
                        help='True to training with Forces.')

    return parser.parse_args()


```

The main training code can be executed from terminal,
```bash
python main.py --workdir /directory/path
```
Another hyper-parameters from the default one can be considered, for example, a linear PIP with 2000 points,
```bash
python main.py --workdir /directory/path --ntr 2000
```

The code will save the predicted values filename: ```'<workdir>/results_N_<ntr>_deg_<poly_degree>.npy'```.

The main training code is contained in the ```train.py``` file,

```python
from pipjax import PIPlayer as PIP
from pipjax import get_functions, detect_molecule, 

molecule_type = 'A4B' # molecular symmetry
na = 5 # number of atoms
poly_degree = 3 # polynomial degree for PIP
features = (128,128,) # number of neurons per layer

# load f_mono and f_poly functions from PIPx
f_mono, f_poly = get_functions(molecule_type, poly_degree)

# PIP model in flax
pip = PIP(f_mono, f_poly,)

x0 = jnp.ones([1, na, 3]) #dummy geometry for methane
params = pip.init(rng, x0)['params']

# energy prediction
y = pip.apply(params,x0)
```

Other value of the **Morse variable** can be used during the initialization of the model,
```python

l = jnp.array(2.)
l0 = nn.softplus(l) 
pip = PIP(f_mono, f_poly,l0)
```

Given the flexibility of JAX, we can jointly compute the energy and the force using ```jax.value_and_grad```,
```python
from pipjax import get_energy_and_forces
@jax.jit
def f_w_grad(params, geoms): return get_energy_and_forces(
    pipnn.apply, geoms, params)
```

## Training parameters ##

For training these linear models with and without forces we have implemented the following functions,
```python
from pipjax import training, training_w_gradients

# load PIP-Flax model
model_pip = PIP(f_mono, f_poly)
model_energy = EnergyPIP(f_mono, f_poly)
params = model_energy.init(key, X_tr[:1])

# training
w = training(model_pip, X_tr, y_tr)  # jnp.array
params = flax_params(w, params)  # transform to Pytree
```
the function ```flax_params()``` copies the parameters to the Pytree object that is used in flax.

