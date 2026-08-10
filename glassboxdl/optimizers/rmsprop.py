import numpy as np

from glassboxdl.core.tensor import Tensor
from glassboxdl.optimizers.optimizer import Optimizer


class RMSprop(Optimizer):
    """
    RMSprop Optimizer.
    Automatically adapts the learning rate for each parameter by dividing
    by the root mean square of recent gradients.

    Update Rule:
        S = (beta * S) + (1 - beta) * (Gradient^2)
        Weight = Weight - (Learning Rate / (sqrt(S) + epsilon)) * Gradient
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        beta: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(parameters, lr)
        self.beta = beta
        self.epsilon = epsilon
        # Keep track of the moving average of squared gradients
        self.sq_grads = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """
        Performs a single optimization step using RMSprop.
        """
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                # 1. Calculate the exponentially weighted average of the squared gradient
                self.sq_grads[i] = (self.beta * self.sq_grads[i]) + (
                    (1.0 - self.beta) * (p.grad**2)
                )

                # 2. Adapt the learning rate and update the parameter
                adapted_lr = self.lr / (np.sqrt(self.sq_grads[i]) + self.epsilon)
                p.data -= adapted_lr * p.grad
