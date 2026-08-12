import numpy as np

from glassboxdl.core import Tensor


class DataLoader:
    """
    Handles batching and shuffling of datasets for the training loop.
    """

    def __init__(self, x_data, y_data, batch_size: int = 32, shuffle: bool = True):
        # Validate that X and Y have the same number of samples
        if len(x_data) != len(y_data):
            raise ValueError(
                f"Length mismatch: x_data has {len(x_data)} samples, but y_data has {len(y_data)} samples."
            )

        self.x = x_data
        self.y = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(x_data)

        # An array of indices [0, 1, 2, ..., n_samples - 1]
        self.indices = np.arange(self.n_samples)

    def __iter__(self):
        """
        Called when a loop over the DataLoader begins.
        Shuffles the indices once per epoch if shuffle = True.
        """

        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current_index = 0
        return self

    def __next__(self):
        """
        Yields the next mini-batch of Tensors.
        """

        if self.current_index >= self.n_samples:
            raise StopIteration

        # Slice the next batch of indices
        batch_idx = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]

        # Increment the index for the next call
        self.current_index += self.batch_size

        batch_x = Tensor(self.x[batch_idx], requires_grad=True)
        batch_y = Tensor(self.y[batch_idx], requires_grad=False)

        return batch_x, batch_y
