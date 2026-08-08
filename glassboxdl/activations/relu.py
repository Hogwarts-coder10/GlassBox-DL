from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class ReLU(Module):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    Formula:
        f(x) = max(0,x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Any, **kwargs: Any) -> Any:
        out_data = np.maximum(0, x.data)

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # The gradient is simply the upstream gradient passed where x > 0
                dx = out.grad * (x.data > 0).astype(float)

                if x.data is None:
                    x.grad = dx
                else:
                    x.grad += dx

        out._backward = _backward
        return out
