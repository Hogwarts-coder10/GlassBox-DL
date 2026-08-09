from typing import Any, Union

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class BCELoss(Module):
    """
    Computes the Binary Cross-Entropy Loss between predictions and targets.

    Formula:
        L = -(1/N) * sum(y_true * log(y_pred + epsilon) + (1 - y_true) * log(1 - y_pred + epsilon))
    """

    def __init__(self, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, predictions: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        pred_data = predictions.data
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # Batch size (N) for averaging
        N = pred_data.size

        # Clip predictions to prevent log(0)
        clipped_preds = np.clip(pred_data, self.epsilon, 1.0 - self.epsilon)

        # Calculate BCE
        term1 = target_data * np.log(clipped_preds)
        term2 = (1.0 - target_data) * np.log(1.0 - clipped_preds)
        loss_data = -np.sum(term1 + term2) / N

        out = Tensor(loss_data, _children=(predictions,))
        out.requires_grad = predictions.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if predictions.requires_grad and out.grad is not None:
                # The local gradient of BCE
                local_grad = (
                    (clipped_preds - target_data)
                    / (clipped_preds * (1.0 - clipped_preds))
                    / N
                )

                # Chain rule with upstream gradient
                dx = out.grad * local_grad

                if predictions.grad is None:
                    predictions.grad = dx
                else:
                    predictions.grad += dx

        out._backward = _backward
        return out
