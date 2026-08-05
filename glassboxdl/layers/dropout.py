import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class Dropout(Module):
    """
    Randomly zeroes some of the elements of the input tensor with
    probability 'p' during training.

    Scales the remaining elements by 1 / (1 - p).
    """

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in range [0,1), got {p}")

        self.p = p

    def forward(self, x):
        # If in evaluation mode or p = 0, do nothing
        if not self.training or self.p == 0.0:
            return x

        # Creating a binomial mask (1 for keep, 0 for drop) and scale it immediately
        # using numpy directly for random generation
        scale = 1.0 / (1.0 - self.p)
        mask = np.random.binomial(1, 1.0 - self.p, size=x.shape) * scale

        # Now multiply the input by the mask wrapped in a Tensor which needs 'requires_grad = False' by default
        # Doing this automatically builds the computational graph for autograd
        return x * Tensor(mask)
