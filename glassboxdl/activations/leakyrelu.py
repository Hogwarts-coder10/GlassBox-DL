"""
Standard ReLU suffers from 'dying ReLU' problem where negative inputs map to
an absolute zero gradient, effectively killing the neuron.

LeakyReLU fixes this by allowing a tiny, non-zero gradient for negative values.
"""

from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class LeakyReLU(Module):
    """
    Applies the LeakyReLU function element-wise

    Formula:
        f(x) = max(ax,x), where a = small slope (usually it's 0.01)
        the gradient is 1, if x > 0 else a.
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Any, **kwargs: Any) -> Any:
        # Forward pass is max(alpha*x, x)
        out_data = np.where(x.data > 0, x.data, self.negative_slope * x.data)

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Gradient is 1 where x > 0, and negative_slope where x <= 0
                local_grad = np.where(x.data > 0, 1.0, self.negative_slope)
                dx = out.grad * local_grad

                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad += dx

        out._backward = _backward
        return out
