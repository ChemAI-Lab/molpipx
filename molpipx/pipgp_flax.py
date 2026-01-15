from typing import Any
import jax
import flax
from flax import nnx
from flax import linen as nn, struct
from dataclasses import dataclass, field
from jaxtyping import Array, Float
import gpjax as gpx
from gpjax.kernels.computations import DenseKernelComputation


@dataclass
class PIPLayerKernel(gpx.kernels.base.AbstractKernel):
    """A GPJax kernel that applies a Neural Network transformation (PIP) before computing the base kernel.

    Attributes:
        base_kernel (gpx.kernels.base.AbstractKernel): The kernel function to apply after the transformation.
        network (nn.Module): A Flax Neural Network module (e.g., ``PIPlayer``) that transforms the input.
        dummy_x (jax.Array): A sample input array used to initialize the neural network parameters.
        key (jax.Array): A JAX random key used for parameter initialization. Defaults to ``jax.random.PRNGKey(123)``.
        compute_engine (DenseKernelComputation): The computation engine for the kernel matrix.
        nn_params (Any): The initialized parameters of the neural network (automatically generated in ``__post_init__``).
    """
    base_kernel: gpx.kernels.base.AbstractKernel = None
    network: nn.Module = struct.field(pytree_node=False, default=None)
    dummy_x: jax.Array = struct.field(pytree_node=True, default=None)
    
    key: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jax.random.PRNGKey(123)
    )
    nn_params: nnx.Data[Any] = None
    compute_engine: DenseKernelComputation = struct.field(pytree_node=False,
    default_factory=lambda: DenseKernelComputation()
)

    

    def __post_init__(self):
        """Initializes the network parameters using the provided dummy input."""
        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")
        if self.network is None:
            raise ValueError("network must be specified")
        self.nn_params = flax.core.unfreeze(self.network.init(self.key, self.dummy_x))

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "1"]:
        """Computes the kernel value between two inputs after transforming them via the network.

        Args:
            x (Array): First input vector. If 1D, it is reshaped to (1, N_atoms, 3).
            y (Array): Second input vector.

        Returns:
            Array: The scalar kernel value.
        """
        if x.ndim == 1:
            x = x.reshape(1, x.shape[0] // 3, 3)

        xt = self.network.apply(self.nn_params, x)
        yt = self.network.apply(self.nn_params, y)
        return self.base_kernel(xt, yt)
