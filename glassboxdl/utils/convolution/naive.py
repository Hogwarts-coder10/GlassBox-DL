import numpy as np


def conv2d_naive(x, weight, bias=None, stride=1, padding=0):
    """
    Naive 2D convolution forward pass using nested loops.

    Args:
        x: Input data of shape (N, C, H, W)
        weight: Filter weights of shape (F, C, HH, WW)
        bias: Biases of shape (F,)
        stride: The number of pixels between adjacent receptive fields
        padding: The number of pixels of zero-padding around the spatial boundary

    Returns:
        out: Output data of shape (N, F, H_out, W_out)
    """
    N, C, H, W = x.shape
    F, C_w, HH, WW = weight.shape

    # Calculate output dimensions
    H_out = 1 + (H + 2 * padding - HH) // stride
    W_out = 1 + (W + 2 * padding - WW) // stride

    # Pad the input spatially (dims 2 and 3)
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    # Initialize the output tensor
    out = np.zeros((N, F, H_out, W_out))

    # The infamous nested loops
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # Define the sliding window boundaries
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the patch
                    patch = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # Perform the element-wise multiplication and sum
                    out[n, f, i, j] = np.sum(patch * weight[f])

                    # Add bias if present
                    if bias is not None:
                        out[n, f, i, j] += bias[f]

    return out


def conv2d_naive_backward(dout, x, weight, stride=1, padding=0):
    """
    Backward pass for naive 2D convolution.

    Args:
        dout: Upstream derivatives of shape (N, F, H_out, W_out)
        x: Input data of shape (N, C, H, W)
        weight: Filter weights of shape (F, C, HH, WW)
        stride: The number of pixels between adjacent receptive fields
        padding: The number of pixels of zero-padding around the spatial boundary

    Returns:
        dx: Gradient with respect to x
        dweight: Gradient with respect to weight
        dbias: Gradient with respect to bias
    """
    N, F, H_out, W_out = dout.shape
    _, C, H, W = x.shape
    _, _, HH, WW = weight.shape

    # Initialize gradients with zeros
    dx_pad = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    dweight = np.zeros_like(weight)
    dbias = np.zeros(F)

    # Pad x just like we did in the forward pass
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    # The reverse nested loops
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # 1. Gradient of the bias is simply the sum of the upstream gradients
                    dbias[f] += dout[n, f, i, j]

                    # 2. Gradient of the weights is the input patch multiplied by the upstream gradient
                    patch = x_pad[n, :, h_start:h_end, w_start:w_end]
                    dweight[f] += patch * dout[n, f, i, j]

                    # 3. Gradient of the input is the weights multiplied by the upstream gradient
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += (
                        weight[f] * dout[n, f, i, j]
                    )

    # Strip the padding from dx_pad to return the gradient in the original shape of x
    if padding > 0:
        dx = dx_pad[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_pad

    return dx, dweight, dbias
