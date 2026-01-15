from typing import Any, Callable, Tuple
from flax import linen as nn
from molpipx.pip_flax import PIPlayer


@nn.jit
class MLP(nn.Module):
    """A standard Multi-Layer Perceptron (MLP) module.

    Attributes:
        features (Tuple[int]): A tuple defining the number of neurons in each hidden layer.
        act_fun (Callable): The activation function applied after each hidden layer. Defaults to ``nn.tanh``.
    """
    features: Tuple[int]
    act_fun: Callable = nn.tanh

    def setup(self):
        """Initializes the dense layers based on the features tuple."""
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  
        self.last_layer = nn.Dense(1)

    @nn.compact
    def __call__(self, x):
        """Applies the MLP to the input.

        Args:
            x (Array): Input tensor.

        Returns:
            Array: The output of the network (scalar per batch item).
        """

        z = x
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = self.act_fun(z)

        return self.last_layer(z)


@nn.jit
class PIPNN(nn.Module):
    """Neural Network model that uses PIPs as input features.

    Attributes:
        f_mono (Callable): Function that returns the monomials.
        f_poly (Callable): Function that returns the polynomials.
        features (Tuple[int]): A tuple defining the number of neurons in each hidden layer of the MLP.
        l (float): Initial value of the Morse variables length scale parameter.
        act_fun (Callable): The activation function used in the MLP. Defaults to ``nn.tanh``.
    """
    f_mono: Callable
    f_poly: Callable
    features: Tuple[int]  # fetures per layer in Tuple
    l: float = float(1.)
    act_fun: Callable = nn.tanh  # lambda x:  f(x)

    def setup(self):
        """Initializes the PIP layer and the MLP layers."""
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  
        self.last_layer = nn.Dense(1)
        self.pip_layer = PIPlayer(self.f_mono, self.f_poly, self.l)

    @nn.compact
    def __call__(self, x):
        """Computes the energy for the input geometries.

        Args:
            x (Array): Batch of geometries (Batch, Na, 3).

        Returns:
            Array: Predicted energy values (Batch, 1).
        """

        z = self.pip_layer(x)
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = self.act_fun(z)

        return self.last_layer(z)
