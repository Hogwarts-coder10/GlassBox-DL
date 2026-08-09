import numpy as np

from glassboxdl.core.tensor import Tensor
from glassboxdl.optimizers.optimizer import Optimizer


class SGDWithMomentum(Optimizer):
    """
    Stochastic Gradient Descent with Momentum.
    Accelerates SGD in the relevant direction and dampens oscillations.

    Update Rule:
        Velocity = (momentum * Velocity) + Gradient
        Weight = Weight - (Learning Rate * Velocity)
    """

    def __init__(
        self, parameters: list[Tensor], lr: float = 0.01, momentum: float = 0.9
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum

        # Initialize velocity for each paramter as an array of zeros
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """
        Performs a single optimization steps using Momentum
        """

        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                # Update the velocities
                self.velocities[i] = (self.momentum * self.velocities[i]) + p.grad

                # Update the parameter using the velocity
                p.data -= self.lr * self.velocities[i]
