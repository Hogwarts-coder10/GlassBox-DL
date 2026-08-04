import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor
from glassboxdl.utils.pooling.naive import (
    avgpool2d_naive,
    avgpool2d_naive_backward,
    maxpool2d_naive,
    maxpool2d_naive_backward,
)


class MaxPool2D(Module):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # If kernel_size is an integer, we use it for both height and width
        self.kernel_size = kernel_size

        # In PyTorch, default stride is equal to kernel_size for pooling
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        """
        Forward pass for MaxPool2D.
        """
        # 1. Compute the raw NumPy forward pass using our utility
        out_data, cache = maxpool2d_naive(
            x.data, self.kernel_size, self.kernel_size, self.stride, self.padding
        )

        # 2. Wrap the output in a Tensor to connect it to the autograd graph
        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        # 3. Define the custom backward closure
        def _backward():
            # Added the 'out.grad is not None' check for your Pyright linter!
            if x.requires_grad and out.grad is not None:
                dx = maxpool2d_naive_backward(out.grad, cache)
                x.grad += dx

        out._backward = _backward

        return out


class AvgPool2D(Module):
    """
    Applies a 2D average pooling over an input signal composed of several input planes.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        """
        Forward pass for AvgPool2D.
        """
        out_data, cache = avgpool2d_naive(
            x.data, self.kernel_size, self.kernel_size, self.stride, self.padding
        )

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                dx = avgpool2d_naive_backward(out.grad, cache)
                x.grad += dx

        out._backward = _backward

        return out
