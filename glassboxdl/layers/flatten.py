from glassboxdl.core.module import Module


class Flatten(Module):
    """
    Flattens a contiguous range for dims into a tensor.
    Defaults to keeping the batch dimension (dim 0) and flattening the rest
    """

    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
