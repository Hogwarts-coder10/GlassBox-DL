from glassboxdl.core.tensor import Tensor


class Parameter(Tensor):
    """
    A subclass of Tensor that indicates to the framework that this is a
    learnable weight (e.g., weights and biases in a neural network).
    """

    def __init__(self, data, requires_grad=True):
        # Parameters always require gradients by default
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Parameter({data_str}, requires_grad={self.requires_grad})"
