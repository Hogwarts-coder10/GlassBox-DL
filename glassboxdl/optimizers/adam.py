import numpy as np

from glassboxdl.core.tensor import Tensor
from glassboxdl.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam (Adapative Moment Estimation) Optimizer.
    Combines the benefits of Momentum (First Moment) and RMSprop (Second Moment).

    Update Rule:
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        Weight = Weight - (Learning Rate * m_hat / (sqrt(v_hat) + epsilon))
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # time step for bias correction

        # First Moment vector (Momentum)
        self.m = [np.zeros_like(p.data) for p in self.parameters]

        # Second Moment vector (RMSprop)
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """
        Performs a single optimization step using Adam.
        """
        self.t += 1

        for i, p in enumerate(self.parameters):
            grad = p.grad  # Assign to a local variable for the type checker

            if grad is not None:
                # 1. Update biased first moment estimate (Momentum)
                self.m[i] = (self.beta1 * self.m[i]) + ((1.0 - self.beta1) * grad)

                # 2. Update biased second raw moment estimate (RMSprop)
                self.v[i] = (self.beta2 * self.v[i]) + ((1.0 - self.beta2) * (grad**2))

                # 3. Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1.0 - (self.beta1**self.t))

                # 4. Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1.0 - (self.beta2**self.t))

                # 5. Update the parameter
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
