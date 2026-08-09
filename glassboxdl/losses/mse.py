from typing import Any, Union

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class MSELoss(Module):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Formula:
        L = (1/n) sum(y_pred - y_true) ^ 2
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, predictions: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        # Extract raw NumPy arrays for computation
        pred_data = predictions.data

        # Targets might be passed as a raw numpy array or a Tensor, handle both
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # Calculate the difference and the final mean squared error
        diff = pred_data - target_data
        loss_data = np.mean(diff**2)

        # The loss is a scalar, but we wrap it in a Tensor to track gradients
        out = Tensor(loss_data, _children=(predictions,))
        out.requires_grad = predictions.requires_grad

        if out.requires_grad:
            # Base gradient for a scalar loss is 1.0, but we initialize to 0.0
            # and let the user call backward() to set it to 1.0 implicitly.
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if predictions.requires_grad and out.grad is not None:
                # N is the total number of elements in the prediction tensor
                N = pred_data.size

                # The local gradient of MSE: (2 / N) * (y_pred - y_true)
                local_grad = (2.0 / N) * diff

                # Multiply by upstream gradient (usually 1.0 for the final loss)
                dx = out.grad * local_grad

                # Route the gradient back to the predictions tensor
                if predictions.grad is None:
                    predictions.grad = dx
                else:
                    predictions.grad += dx

        out._backward = _backward
        return out
