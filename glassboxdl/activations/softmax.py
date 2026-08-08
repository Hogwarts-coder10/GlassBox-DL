from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class Softmax(Module):
    """
    Takes a vector of arbitrary real scores and squashes them into a probability distribution that sums to 1.

    Formula:
        f(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Any, **kwargs: Any) -> Any:
        # Subtract max for numerical stability (prevents overflow)
        x_max = np.max(x.data, axis=self.axis, keepdims=True)
        exp_x = np.exp(x.data - x_max)
        out_data = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # The gradient of softmax interacting with upstream grad (Jacobian vector product)
                sum_grad_out = np.sum(
                    out.grad * out_data, axis=self.axis, keepdims=True
                )
                dx = out_data * (out.grad - sum_grad_out)

                # Type-safe gradient routing
                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad += dx

        out._backward = _backward
        return out
