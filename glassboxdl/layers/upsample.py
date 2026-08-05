import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class Upsample(Module):
    """
    Upsamples a given 2D spatial tensor using nearest-neighbor interpolation.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale = int(scale_factor)

    def forward(self, x):
        # Repeat elements across Height (axis 2) and Width (axis 3)
        out_data = np.repeat(np.repeat(x.data, self.scale, axis=2), self.scale, axis=3)

        out = Tensor(out_data, _children=(x,))
        out.requires_grad = x.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if x.requires_grad and out.grad is not None:
                N, C, H, W = out.grad.shape

                # Reshape and sum block-wise to reverse the nearest-neighbor duplication
                dx = out.grad.reshape(
                    N, C, H // self.scale, self.scale, W // self.scale, self.scale
                ).sum(axis=(3, 5))

                x.grad += dx

        out._backward = _backward
        return out
