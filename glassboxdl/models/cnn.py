from typing import Generator

from glassboxdl.activations.relu import ReLU
from glassboxdl.core.sequential import Sequential
from glassboxdl.core.tensor import Tensor
from glassboxdl.layers.convolution import Conv2D
from glassboxdl.layers.flatten import Flatten
from glassboxdl.layers.linear import Linear
from glassboxdl.layers.pooling import MaxPool2D


class CNN:
    """
    A standard Convolutional Neural Network for Image Classification.
    Architecture: 2x (Conv2D -> ReLU -> MaxPool) -> Flatten -> Dense -> Dense
    """

    def __init__(self, in_channels: int, num_classes: int, image_size: int = 28):
        """
        Args:
            in_channels: Number of channels in the input image (e.g., 1 for Grayscale).
            num_classes: Number of output categories for classification.
            image_size: The height/width of the input image (used to calculate Flatten size).
        """
        # Calculate the spatial dimensions after two 2x2 max pooling layers
        # (e.g., 28 -> 14 -> 7)
        final_size = image_size // 4
        flattened_dim = 32 * final_size * final_size

        self.model = Sequential(
            # Feature Extraction Block 1
            Conv2D(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            # Feature Extraction Block 2
            Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            # Classification Head
            Flatten(),
            Linear(flattened_dim, 128),
            ReLU(),
            Linear(128, num_classes),
        )

    def __call__(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass through the network.
        """
        return self.model(x)

    def parameters(self) -> Generator[Tensor, None, None]:
        """
        Returns a list of all learnable parameters (weights and biases).
        """

        yield from self.model.parameters()
