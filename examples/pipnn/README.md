### PIP-NN Training tutorial ###
In this tutorial we show how to train a PIP neural network from scratch.<br>
We will use methane as example. 

The ```main.py``` file contains the following ```argparse```,

```python

def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure training parameters.")

    # Adding arguments for each configuration option
    parser.add_argument('--workdir', type=str,
                        default=None, help='Working directory')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training.')
    parser.add_argument('--molecule', type=str,
                        default='A4B', help='Type of molecule.')
    parser.add_argument('--poly_degree', type=int, default=3,
                        help='Degree of the polynomial.')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of layers in the neural network.')
    parser.add_argument('--n_neurons', type=int, default=128,
                        help='Number of neurons per layer.')
    parser.add_argument('--ntr', type=int, default=1000,
                        help='Number of training examples.')
    parser.add_argument('--nval', type=int, default=1000,
                        help='Number of validation examples.')
    parser.add_argument('--ntst', type=int, default=2000,
                        help='Number of test examples.')

    return parser.parse_args()


```

The main training code can be executed from terminal,
```bash
python main.py --workdir /directory/path
```
Another hyper-parameters from the default one can be considered, for example, a single layer PIPNN trained with 5000 points,
```bash
python main.py --workdir /directory/path --n_layers 1 --ntr 5000
```

The code will save the optimal parameters in the last epoch, combined with a CSV file with the information of the training trajectory, filename: ```training_trajectory_ema.csv```.


The main training code is contained in the ```train.py``` file, where for methane the ```PIPNN``` model initialized the following way,

```python
from pipx import PIPNN
from pipx import get_functions, detect_molecule, 

molecule_type = 'A4B' # molecular symmetry
na = 5 #number of atoms
poly_degree = 3 # polynomial degree for PIP
features = (128,128,) # number of neurons per layer

# load f_mono and f_poly functions from PIPx
f_mono, f_poly = get_functions(molecule_type, poly_degree)

# PIPNN model in flax
pipnn = PIPNN(f_mono, f_poly, features,)

x0 = jnp.ones([1, na, 3]) #dummy geometry for methane
params = pipnn.init(rng, x0)['params']

# energy prediction
y = pipnn.apply(params,x0)
```


Given the flexibility of JAX, we can jointly compute the energy and the force using ```jax.value_and_grad```,
<!-- ```python
from pipx import get_energy_and_forces
@jax.jit
def f_w_grad(params, geoms): return get_energy_and_forces(
    pipnn.apply, geoms, params)
``` -->

```python
from pipx import get_energy_and_forces

y, f = get_energy_and_forces(pipnn.apply, X, params_opt) # energy and forces
```

See [train_and_evaluate](train.py), and with forces [train_and_evaluate](train_w_grad.py), for training a PIP-NN using Optax.