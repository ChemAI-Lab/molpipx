from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import flax
from flax import linen as nn

from dataclasses import (
    dataclass,
    field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.base import static_field




@dataclass
class PIPLayerKernel(AbstractKernel):

    """
    A kernel class for combining a PIP with a base kernel in a Gaussian Process.

    Attributes:
        base_kernel: The base kernel used for the Gaussian Process.
        network: The PIP used to transform the input data.
        dummy_x: Dummy input data used for initializing the model.
        key: Random key for initializing the neural network.
        nn_params: Parameters of PIP, initialized in the post-init method.
    """

    base_kernel: AbstractKernel = None
    network: nn.Module = static_field(None)
    dummy_x: jax.Array = static_field(None)
    key: jax.Array = static_field(jax.random.key(123))
    nn_params: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")
        if self.network is None:
            raise ValueError("network must be specified")
        self.nn_params = flax.core.unfreeze(self.network.init(self.key, self.dummy_x))

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "1"]:
        
        if x.ndim == 1:
             x = x.reshape(1, x.shape[0]//3, 3)
        
        state = self.network.init(self.key, x)
        xt = self.network.apply(state, x)
        yt = self.network.apply(state, y)
        return self.base_kernel(xt, yt)
    
