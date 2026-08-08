from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class Sigmoid(Module):
    """
    Applies Sigmoid function element-wise

    Formula:
        s(x) = 1 / (1 + exp(-x))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Any, **kwargs) -> Any:
        # Clipping the input to prevent numpy overflow warnings with exp()
        clipped_data = np.clip(x.data, -250, 250)
        out_data = 1.0 / (1.0 + np.exp(-clipped_data))

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # d/dx (s(x)) = out * (1 - out) where s(x) is sigmoid function
                dx = out.grad * (out_data * (1 - out_data))

                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad += dx

        out._backward = _backward
        return out
