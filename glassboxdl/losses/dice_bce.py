from typing import Any, Union

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class DiceBCELoss(Module):
    """
    Computes a combination of Dice Loss and Binary Cross-Entropy (BCE) Loss.
    This hybrid is highly effective for image segmentation tasks.

    Formula:
        BCE  = -(1/N) * sum(y_true * log(y_pred + eps) + (1 - y_true) * log(1 - y_pred + eps))
        Dice = 1 - (2 * sum(y_pred * y_true) + smooth) / (sum(y_pred) + sum(y_true) + smooth)
        Loss = BCE + Dice
    """

    def __init__(self, smooth: float = 1e-5, epsilon: float = 1e-9):
        super().__init__()
        self.smooth = smooth
        self.epsilon = epsilon

    def forward(
        self, predictions: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        pred_data = predictions.data
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # --- BCE Loss Calculation ---
        N = pred_data.size
        p = np.clip(pred_data, self.epsilon, 1.0 - self.epsilon)

        term1 = target_data * np.log(p)
        term2 = (1.0 - target_data) * np.log(1.0 - p)
        bce_loss = -np.sum(term1 + term2) / N

        # --- Dice Loss Calculation ---
        p_flat = pred_data.flatten()
        t_flat = target_data.flatten()

        intersection = np.sum(p_flat * t_flat)
        denominator = np.sum(p_flat) + np.sum(t_flat)
        dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_coeff

        # --- Combined Total Loss ---
        loss_data = bce_loss + dice_loss

        out = Tensor(loss_data, _children=(predictions,))
        out.requires_grad = predictions.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if predictions.requires_grad and out.grad is not None:
                # 1. BCE Gradient
                bce_grad = (p - target_data) / (p * (1.0 - p)) / N

                # 2. Dice Gradient
                term_a = 2.0 * t_flat * (denominator + self.smooth)
                term_b = 2.0 * intersection + self.smooth
                denom_squared = (denominator + self.smooth) ** 2

                dice_grad_flat = -(term_a - term_b) / denom_squared
                dice_grad = dice_grad_flat.reshape(pred_data.shape)

                # 3. Combined Gradient (The magic of addition!)
                local_grad = bce_grad + dice_grad

                # Chain rule with upstream gradient
                dx = out.grad * local_grad

                if predictions.grad is None:
                    predictions.grad = dx
                else:
                    predictions.grad += dx

        out._backward = _backward
        return out
