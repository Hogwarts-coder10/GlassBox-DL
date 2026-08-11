import math
from typing import Any

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.parameter import Parameter
from glassboxdl.core.tensor import Tensor
from glassboxdl.utils.convolution.naive import conv2d_naive, conv2d_naive_backward


class Conv2D(Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        algorithm="naive",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.algorithm = algorithm

        # Handle both integer and tuple kernel sizes
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Kaiming Uniform Initialization
        k = 1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        bound = math.sqrt(k)

        weight_data = np.random.uniform(
            -bound,
            bound,
            (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
        )
        self.weight = Parameter(weight_data, requires_grad=True)

        bias_data = np.random.uniform(-bound, bound, out_channels)
        self.bias = Parameter(bias_data, requires_grad=True)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        if self.algorithm == "naive":
            out_data = conv2d_naive(
                x.data, self.weight.data, self.bias.data, self.stride, self.padding
            )
        else:
            raise NotImplementedError(
                f"Convolution algorithm '{self.algorithm}' is not yet implemented."
            )

        out = Tensor(out_data, _children=(x, self.weight, self.bias))
        out.requires_grad = (
            x.requires_grad or self.weight.requires_grad or self.bias.requires_grad
        )

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():

            if out.grad is not None:
                if self.algorithm == "naive":
                    dx, dweight, dbias = conv2d_naive_backward(
                        out.grad, x.data, self.weight.data, self.stride, self.padding
                    )
                else:
                    raise NotImplementedError(
                        f"Convolution algorithm '{self.algorithm}' is not yet implemented."
                    )

                # Route the gradients back to the respective tensors
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = dx
                    else:
                        x.grad += dx

                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = dweight
                    else:
                        self.weight.grad += dweight

                if self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = dbias
                    else:
                        self.bias.grad += dbias

        out._backward = _backward
        return out
