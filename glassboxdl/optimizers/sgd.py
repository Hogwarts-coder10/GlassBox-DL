from glassboxdl.core.tensor import Tensor
from glassboxdl.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) Optimizer.

    Update Rule:
        Weight = Weight - (Learning Rate * Gradient)
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01):
        super().__init__(parameters, lr)

    def step(self):
        """
        Performs a single optimization step, updating all parameters.
        """

        for p in self.parameters:
            if p.grad is not None:
                # Updating the raw NumPy array directly
                p.data -= self.lr * p.grad
