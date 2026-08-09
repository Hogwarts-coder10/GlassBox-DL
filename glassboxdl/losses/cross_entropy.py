from typing import Any, Union

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class CrossEntropyLoss(Module):
    """
    Computes the Categorical Cross-Entropy Loss between predictions and one-hot encoded targets.
    Expects predictions to be probabilities (e.g., output of a Softmax layer).

    Formula:
        L = -(1/n) sum(y_true) * log(y_pred + ∈) where ∈ = epsilon
    """

    def __init__(self, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, predictions: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        pred_data = predictions.data
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # Batch size (N) for averaging the loss
        N = pred_data.shape[0] if len(pred_data.shape) > 1 else 1

        # Clip predictions to prevent log(0)
        clipped_preds = np.clip(pred_data, self.epsilon, 1.0 - self.epsilon)

        # Calculate categorical cross-entropy
        loss_data = -np.sum(target_data * np.log(clipped_preds)) / N

        out = Tensor(loss_data, _children=(predictions,))
        out.requires_grad = predictions.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if predictions.requires_grad and out.grad is not None:
                # The local gradient of Cross-Entropy
                local_grad = -(target_data / clipped_preds) / N

                # Chain rule with upstream gradient
                dx = out.grad * local_grad

                if predictions.grad is None:
                    predictions.grad = dx
                else:
                    predictions.grad += dx

        out._backward = _backward
        return out
