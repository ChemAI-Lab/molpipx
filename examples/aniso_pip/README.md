### Linear AnisoPIP tutorial ###

**Please read the linear PIP tutorial before proceeding with this one :**


In this tutorial, we show how to train a linear **Aniso**PIP from scratch.<br>
PIP models depend on the computation of the Morse variables,
$$
\bar{\boldsymbol{\gamma}} = e^{-\lambda \mathbf{r}},
$$
where $\lambda$ is a length scale parameter, and $\mathbf{r}$ pair-wise distances between the nuclei. 
We could extend this function to a length-scale parameter for each individual type of atom-atom distance. For example, for methane we have ```H-H``` and ```C-H``` distances, meaning we could define two different length-scale parameters; $\lambda_{\text{HH}}$ and $\lambda_{\text{CH}}$.
For the model to remain permutational invariant, each length-scale parameter must only affect the same type of distance. <br>
For this, we will use a ```mask``` vector, and ```f_mask``` function,

```python
from molpipx import get_mask, get_f_mask

atoms = ['H','H','H','H','C']
mask, unique_pairs = get_mask(atoms)

# define a mask function
f_mask = get_f_mask(mask)

n_pairs = 2 # number of unique type of distances
l = jnp.ones(n_pairs)
d = jnp.ones(10) # atom-atom distances, 10 is for CH4
z = f_mask(l, d)
morse = jnp.exp(-1*z) # Morse variables with mask
```


We will use methane as an example. 

The ```main.py``` file contains the following ```argparse```,

```python

def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure training parameters.")

    # Adding arguments for each configuration option
    parser.add_argument('--workdir', type=str,
                        default=None, help='Working directory')
    parser.add_argument('--learning_rate', type=float,
                        default=2E-3, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Number of epochs for training.')
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
    parser.add_argument('--opt_tol', type=float, default=1E-4,
                        help='Tolerance parameters.')
    parser.add_argument('--grad_bool', type=bool, default=False,
                        help='If True training with Forces.')

    return parser.parse_args()

```

The main training code can be executed from terminal,
```bash
python main.py --workdir /directory/path
```
Another hyper-parameters from the default one can be considered, for example, a linear PIP with 2000 points and a learning rate of 0.1
```bash
python main.py --workdir /directory/path --ntr 2000 --learning_rate 0.1
```

The code will save the predicted values filename: ```'<workdir>/training_trajectory_and_l_params.csv'```.

The main training code is contained in the ```train.py``` file. 


This demo uses **Adam**, using Optax, to optimize the validation loss function with respect to the length-scale parameters. 
This nested loss function is defined as, 

```python 
    @jax.jit
    def validation_loss(params_pip, data_, params_energy):
        (X_tr, y_tr), (X_val, y_val) = data_

        def inner_loss(params_pip, X_tr, y_tr):
            Pip_tr = model_pip.apply(params_pip, X_tr)
            results = jnp.linalg.lstsq(Pip_tr, y_tr)
            w = results[0]
            return w

        w_opt = inner_loss(params_pip, X_tr, y_tr)
        params_energy = flax_params(w_opt, params_energy)
        y_val_pred = model_energy.apply(params_energy, X_val)

        loss_val = mse_loss(y_val_pred, y_val)
        loss_tr = mse_loss(model_energy.apply(params_energy, X_tr))
        return loss_val, (params_energy, loss_tr)
```


