from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class Tanh(Module):
    """
    Applies the hyperbolic tangent (Tanh) function element-wise

    Formula:
        f(x) = tanh(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Any, **kwargs: Any) -> Any:
        out_data = np.tanh(x)

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # d/dx(tanh(x)) = 1 - (tanh(x))^ 2
                dx = out.grad * (1.0 - out_data**2)

                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad += dx

        out._backward = _backward
        return out
