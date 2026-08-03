from glassboxdl.core.initialization import he_normal, zeros
from glassboxdl.core.module import Module


class Linear(Module):
    """
    Applies a linear transformation to the incoming data
    Y = X @ W + b
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initializing the weights and biases as parameters
        # Shape is (in_features,out_features) so forward is simply X @ W

        self.weight = he_normal(in_features, out_features)

        if bias:
            # Shape (1,out_features) allows it to cleanly broadcast
            # across the batch dimmension during addition
            self.bias = zeros(1, out_features)

        else:
            self.bias = None

    def forward(self, x):
        """
        Computes the forward pass of linear layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size,in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """

        # Matrix Multiplication builds the graph via Tensor.__matmul__
        out = x @ self.weight

        if self.bias is not None:
            out += self.bias

        return out
