from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class GELU(Module):
    """
    Applies the Gaussian Error Linear Unit (GELU) function.
    Uses the standard Tanh approximation for pure NumPy compatibility.

    Formula:
        f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Any, **kwargs: Any) -> Any:
        # Constants for the approximation
        self.sqrt_2_pi = np.sqrt(2.0 / np.pi)
        self.coeff = 0.044715

        # Intermediate calculations for readability
        x_cubed = x.data**3
        self.inner = self.sqrt_2_pi * (x.data + self.coeff * x_cubed)
        self.tanh_inner = np.tanh(self.inner)

        # Forward pass
        out_data = 0.5 * x.data * (1.0 + self.tanh_inner)

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Derivative of the GELU tanh approximation
                # d/dx [ 0.5 * x * (1 + tanh(z)) ]
                term1 = 0.5 * (1.0 + self.tanh_inner)

                # Derivative of the inner tanh portion
                sech_squared = 1.0 - self.tanh_inner**2
                d_inner_dx = self.sqrt_2_pi * (1.0 + 3.0 * self.coeff * (x.data**2))
                term2 = 0.5 * x.data * sech_squared * d_inner_dx

                dx = out.grad * (term1 + term2)

                # Type-safe gradient routing
                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad += dx

        out._backward = _backward
        return out
