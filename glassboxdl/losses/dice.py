from typing import Any, Union

import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor


class DiceLoss(Module):
    """
    Computes the Sørensen-Dice coefficient loss for image segmentation tasks.
    It measures the overlap between predictions and targets.

    Formula:
        Intersection = sum(y_pred * y_true)
        Denominator = sum(y_pred) + sum(y_true)
        Loss = 1 - ((2 * Intersection + smooth) / (Denominator + smooth))

        We use a smooth factor (usually a tiny number like 1e-5) in both the numerator and denominator
        to prevent dividing by zero if the true mask and prediction are both completely blank.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, predictions: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        pred_data = predictions.data
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # Flattening allows us to easily compute the global overlap
        p_flat = pred_data.flatten()
        t_flat = target_data.flatten()

        # Calculate intersection and the denominator
        intersection = np.sum(p_flat * t_flat)
        denominator = np.sum(p_flat) + np.sum(t_flat)

        # Calculate the Dice coefficient and final loss
        dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss_data = 1.0 - dice_coeff

        out = Tensor(loss_data, _children=(predictions,))
        out.requires_grad = predictions.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if predictions.requires_grad and out.grad is not None:
                # Apply the quotient rule for the derivative of the Dice coefficient
                # dL/dp = - [ (2 * t_flat * (Denom + smooth)) - (2 * Intersect + smooth) ] / (Denom + smooth)^2
                term1 = 2.0 * t_flat * (denominator + self.smooth)
                term2 = 2.0 * intersection + self.smooth
                denom_squared = (denominator + self.smooth) ** 2

                grad_flat = -(term1 - term2) / denom_squared

                # Reshape the flat gradient back to the original prediction shape
                local_grad = grad_flat.reshape(pred_data.shape)

                # Chain rule with upstream gradient
                dx = out.grad * local_grad

                if predictions.grad is None:
                    predictions.grad = dx
                else:
                    predictions.grad += dx

        out._backward = _backward
        return out
