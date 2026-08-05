import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor
from glassboxdl.utils.normalization.batchnorm import (
    batchnorm2d_backward,
    batchnorm2d_forward,
)


class BatchNorm2D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (gamma and beta)
        self.weight = Tensor(np.ones(num_features), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not tracked by autograd)
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, x):
        out_data, self.running_mean, self.running_var, cache = batchnorm2d_forward(
            x.data,
            self.weight.data,
            self.bias.data,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            self.training,
        )

        out = Tensor(out_data, _children=(x, self.weight, self.bias))
        out.requires_grad = x.requires_grad or self.weight.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if out.grad is not None:
                dx, dgamma, dbeta = batchnorm2d_backward(out.grad, cache)
                if x.requires_grad:
                    x.grad += dx
                if self.weight.requires_grad:
                    self.weight.grad += dgamma
                if self.bias.requires_grad:
                    self.bias.grad += dbeta

        out._backward = _backward
        return out
