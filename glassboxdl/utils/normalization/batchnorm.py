import numpy as np


def batchnorm2d_forward(
    x, gamma, beta, running_mean, running_var, momentum, eps, training
):
    """
    Forward pass for 2D Batch Normalization.
    x shape: (N, C, H, W)
    """
    if training:
        # Calculate mean and variance along Batch, Height, and Width (keeping Channels)
        mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        # Update running statistics
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        mu = running_mean
        var = running_var

    # Normalize
    x_norm = (x - mu) / np.sqrt(var + eps)

    # Scale and shift
    # reshape gamma and beta to (1, C, 1, 1) for broadcasting
    out = gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)

    cache = (x, x_norm, mu, var, gamma, eps)
    return out, running_mean, running_var, cache


def batchnorm2d_backward(dout, cache):
    """
    Backward pass for 2D Batch Normalization using the simplified gradient formula.
    """
    x, x_norm, mu, var, gamma, eps = cache
    N, C, H, W = dout.shape
    m = N * H * W  # Total number of elements normalized per channel

    # Reshape gamma for broadcasting
    gamma = gamma.reshape(1, C, 1, 1)

    # Gradients with respect to gamma and beta
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))
    dbeta = np.sum(dout, axis=(0, 2, 3))

    # Gradient with respect to x (optimized formula)
    std_inv = 1.0 / np.sqrt(var + eps)
    dx = (
        (1.0 / m)
        * gamma
        * std_inv
        * (
            m * dout
            - np.sum(dout, axis=(0, 2, 3), keepdims=True)
            - x_norm * np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
        )
    )

    return dx, dgamma, dbeta
