import numpy as np

from glassboxdl.core.parameter import Parameter


def zeros(*shape):
    """Initializes a parameter with zeros."""
    return Parameter(np.zeros(shape))


def ones(*shape):
    """Initializes a parameter with ones."""
    return Parameter(np.ones(shape))


def xavier_uniform(fan_in, fan_out):
    """
    Xavier (Glorot) Uniform initialization.
    Best for Sigmoid or Tanh activation functions.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    data = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
    return Parameter(data)


def he_normal(fan_in, fan_out):
    """
    He (Kaiming) Normal initialization.
    Best for ReLU or LeakyReLU activation functions.
    """
    std = np.sqrt(2.0 / fan_in)
    data = np.random.normal(0, std, size=(fan_in, fan_out))
    return Parameter(data)
