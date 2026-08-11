from typing import Generator, Optional

from glassboxdl.activations.relu import ReLU
from glassboxdl.activations.sigmoid import Sigmoid
from glassboxdl.core.sequential import Sequential
from glassboxdl.core.tensor import Tensor
from glassboxdl.layers.linear import Linear


class ANN:
    """
    A standard Artificial Neural Network (Multi-Layer Perceptron).
    Constructs a feedforward network with fully connected layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        hidden_activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        """
        Args:
            input_dim: The number of features in the input data.
            hidden_dims: A list of integers representing the number of neurons in each hidden layer.
            output_dim: The number of neurons in the final output layer.
            hidden_activation: The activation function to use for hidden layers ('relu' or 'sigmoid').
            output_activation: The activation function to use for the output layer ('sigmoid', or None for linear output).
        """
        layers = []

        # 1. Build the Input to First Hidden Layer
        current_dim = input_dim

        # 2. Build the Hidden Layers
        for h_dim in hidden_dims:
            # Add the fully connected Linear layer
            layers.append(Linear(current_dim, h_dim))

            # Add the non-linear activation function
            if hidden_activation.lower() == "relu":
                layers.append(ReLU())
            elif hidden_activation.lower() == "sigmoid":
                layers.append(Sigmoid())
            else:
                raise ValueError(f"Unsupported hidden activation: {hidden_activation}")

            current_dim = h_dim

        # 3. Build the Output Layer
        layers.append(Linear(current_dim, output_dim))

        if output_activation:
            if output_activation.lower() == "sigmoid":
                layers.append(Sigmoid())
            else:
                raise ValueError(f"Unsupported output activation: {output_activation}")

        # 4. Wrap everything in our Sequential container
        self.model = Sequential(*layers)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass through the network.
        """
        return self.model(x)

    def parameters(self) -> Generator[Tensor, None, None]:
        yield from self.model.parameters()
