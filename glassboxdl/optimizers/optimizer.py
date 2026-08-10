from glassboxdl.core.tensor import Tensor


class Optimizer:
    """
    Base class for all optmization algorithms.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01):
        # we only optimize parameters that actually require gradients
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def step(self):
        """
        Update the parameters. Must be implemented by subclasses.
        """

        raise NotImplementedError("Subclasses must implement the step() method.")

    def zero_grad(self):
        """
        Clears the gradients of all optimized Tensors.
        This is cruical to call before a backward pass so that
        gradients don't accumulate.
        """

        for p in self.parameters:
            if p.grad is not None:
                # reset them to zeros or None for next pass
                import numpy as np

                p.grad = np.zeros_like(p.grad)
