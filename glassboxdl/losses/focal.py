from typing import Any, Union

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class FocalLoss(Module):
    """
    Computes the Binary Focal Loss to address extreme class imbalance.
    It down-weights well-classified examples to focus on hard, misclassified ones.

    Formula:
        For y = 1: L = -alpha * (1 - p)^gamma * log(p)
        For y = 0: L = -(1 - alpha) * p^gamma * log(1 - p)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, epsilon: float = 1e-9):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(
        self, predictions: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        pred_data = predictions.data
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # Batch size for averaging
        N = pred_data.size

        # Clip predictions to prevent log(0) and numerical instability in powers
        p = np.clip(pred_data, self.epsilon, 1.0 - self.epsilon)

        # Calculate Focal Loss components for true class (y=1) and false class (y=0)
        term1 = -self.alpha * ((1.0 - p) ** self.gamma) * np.log(p)
        term0 = -(1.0 - self.alpha) * (p**self.gamma) * np.log(1.0 - p)

        # Combine based on actual targets and average over the batch
        loss_array = target_data * term1 + (1.0 - target_data) * term0
        loss_data = np.sum(loss_array) / N

        out = Tensor(loss_data, _children=(predictions,))
        out.requires_grad = predictions.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if predictions.requires_grad and out.grad is not None:
                # Derivative for y = 1
                # d/dp [ -alpha * (1-p)^gamma * log(p) ]
                grad_1 = -self.alpha * (
                    -self.gamma * ((1.0 - p) ** (self.gamma - 1.0)) * np.log(p)
                    + ((1.0 - p) ** self.gamma) / p
                )

                # Derivative for y = 0
                # d/dp [ -(1-alpha) * p^gamma * log(1-p) ]
                grad_0 = -(1.0 - self.alpha) * (
                    self.gamma * (p ** (self.gamma - 1.0)) * np.log(1.0 - p)
                    - (p**self.gamma) / (1.0 - p)
                )

                # Combine local gradients based on true targets
                local_grad = (target_data * grad_1 + (1.0 - target_data) * grad_0) / N

                # Chain rule with upstream gradient
                dx = out.grad * local_grad

                if predictions.grad is None:
                    predictions.grad = dx
                else:
                    predictions.grad += dx

        out._backward = _backward
        return out
