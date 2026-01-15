from typing import Callable, Tuple, List
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import flax.linen as nn
import optax

from molpipx.utils import softplus_inverse, flax_params, flax_params_aniso
from molpipx.pip_anisotropic_flax import get_mask, LayerPIPAniso, EnergyPIPAniso
from molpipx.utils import mse_loss

def _train_aniso_adam(data:Tuple, atoms:List,
                      f_mono:Callable,f_poly:Callable, 
                      optimizer_info:PyTree) -> Tuple[Float, PyTree]:
    """Trains an anisotropic PIP model using the Adam optimizer.

    Args:
        data (Tuple): A tuple containing training and validation data:
            ``((X_tr, y_tr), (X_val, y_val))``.
        atoms (List): A list of atom types (e.g., ``['H', 'C', 'H']``) used to generate the mask.
        f_mono (Callable): Function that returns the monomials.
        f_poly (Callable): Function that returns the polynomials.
        optimizer_info (PyTree): A dictionary or PyTree containing optimizer settings:
            * ``'tol'``: Convergence tolerance.
            * ``'Maxiters'``: Maximum number of epochs.
            * ``'learning_rate'``: Learning rate for Adam.

    Returns:
        Tuple[Float, PyTree]: A tuple containing:
            * The final optimized parameters (``l_opt[-1]``).
            * The history of parameters over all epochs (``l_opt``).
    """
    opt_tol = optimizer_info['tol']
    n_epochs = optimizer_info['Maxiters']
    lr = optimizer_info['learning_rate']
    
    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)
    
    (X_tr,y_tr), (X_val, y_val) = data
    
    mask = get_mask(atoms)
    n_pairs = mask.shape[0]
    
    # initialize models
    model_pip = LayerPIPAniso(f_mono, f_poly, n_pairs)
    params_pip = model_pip.init(key, X_tr[:1], mask)
    print(params_pip)
    
    model_energy = EnergyPIPAniso(f_mono, f_poly, n_pairs)
    params_energy = model_energy.init(key, X_tr[:1], mask)
    
    f_pip = lambda params,x: model_pip.apply(params, x, mask)
    f_pip_energy = lambda params, x: model_energy.apply(params, x, mask)
    
    # random initial parameters
    _, key = jax.random.split(key)
    l_init = jax.random.uniform(key, shape=(2, ), minval=1., maxval=2.)
    l_init = softplus_inverse(l_init)


    @jax.jit
    def validation_loss(l:Float[Array,"npairs"], data_:Tuple, params_:PyTree)->Tuple:
        (X_tr, y_tr), (X_val, y_val) = data_
        params_pip, params_energy = params_

        params_pip = flax_params_aniso(l, params_pip)
        Pip_tr = f_pip(params_pip, X_tr)
        
        def inner_loss(Pip_tr, y_tr):
            results = jnp.linalg.lstsq(Pip_tr, y_tr)
            w = results[0]
            return w

        w_opt = inner_loss(Pip_tr, y_tr)
        loss_tr = mse_loss(f_pip(params_pip, X_tr), y_tr)

        params_energy = flax_params(w_opt, params_energy)
        y_val_pred = f_pip_energy(params_energy, X_val)
            
        loss_val = mse_loss(y_val_pred, y_val)
        return loss_val, (w_opt,loss_tr)

    params_all = (params_pip, params_energy)
    
    # optax loop
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(l_init)
    
    grad_fn = jax.jit(jax.value_and_grad(validation_loss, argnums=(0,), has_aux=True))
    
    @jax.jit
    def train_step(li, optimizer_state):
        (loss, (_,loss_tr)), grads = grad_fn(li, data, params_all)
        updates, opt_state = optimizer.update(
            grads[0], optimizer_state, li)
        return optax.apply_updates(li, updates), opt_state, loss, loss_tr
    
    l_params = l_init
    l_ = []
    for i in range(n_epochs):
        l0_params = l_params
        l_params, opt_state, loss_val_i, loss_tr_i = train_step(l_params,opt_state)
        print(i, loss_val_i, loss_tr_i, l_params, nn.softplus(l_params), 
              jnp.linalg.norm(l_params - l0_params))
        l_.append(l_params)
        if jnp.linalg.norm(l_params - l0_params) < opt_tol:
            break
    l_opt = jnp.array(l_)
    
    return l_opt[-1], l_opt 