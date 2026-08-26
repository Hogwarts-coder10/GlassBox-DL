from typing import Any, Union
import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.tensor import Tensor
from glassboxdl.activations.softmax import Softmax # Import your Softmax!

class DiceLoss(Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        # Initialize the Softmax layer right in the constructor
        self.softmax = Softmax(axis=1) 

    def forward(
        self, logits: Tensor, targets: Union[Tensor, np.ndarray], **kwargs: Any
    ) -> Tensor:
        # 1. GRAPH LINKING: Convert raw logits to probabilities 
        # This automatically tracks the Softmax gradients in autograd!
        probs = self.softmax(logits)

        # 2. Extract the bounded probability data
        pred_data = probs.data
        target_data = targets.data if isinstance(targets, Tensor) else targets

        # Flatten arrays for the global overlap calculation
        p_flat = pred_data.flatten()
        t_flat = target_data.flatten()

        # Calculate intersection and the denominator
        intersection = np.sum(p_flat * t_flat)
        denominator = np.sum(p_flat) + np.sum(t_flat)

        # Calculate the Dice coefficient and final loss
        dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss_data = 1.0 - dice_coeff

        # 3. GRAPH LINKING: Wrap the output and link it to `probs` (not logits)
        out = Tensor(
            loss_data, requires_grad=probs.requires_grad,
            _children=(probs,)
        )
        

        def _backward():
            if probs.requires_grad:
                # Apply your exact quotient rule for the Dice derivative
                term1 = 2.0 * t_flat * (denominator + self.smooth)
                term2 = 2.0 * intersection + self.smooth
                denom_squared = (denominator + self.smooth) ** 2

                grad_flat = -(term1 - term2) / denom_squared

                # Reshape back to the prediction shape
                local_grad = grad_flat.reshape(pred_data.shape)

                # Initialize or accumulate gradients on the probability tensor
                dx = out.grad * local_grad
                if probs.grad is None:
                    probs.grad = dx
                else:
                    probs.grad += dx

        out._backward = _backward
        return out
