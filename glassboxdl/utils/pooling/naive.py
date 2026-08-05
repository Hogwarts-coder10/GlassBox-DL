import numpy as np


def maxpool2d_naive(x, pool_height, pool_width, stride=1, padding=0):
    """
    Naive 2D Max Pooling forward pass.
    """
    N, C, H, W = x.shape

    H_out = 1 + (H + 2 * padding - pool_height) // stride
    W_out = 1 + (W + 2 * padding - pool_width) // stride

    x_pad = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    patch = x_pad[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(patch)

    # We return x_pad and the dimensions as a cache for the backward pass
    cache = (x, x_pad, pool_height, pool_width, stride, padding)
    return out, cache


def maxpool2d_naive_backward(dout, cache):
    """
    Backward pass for naive 2D Max Pooling.
    """
    x, x_pad, pool_height, pool_width, stride, padding = cache
    N, C, H_out, W_out = dout.shape

    dx_pad = np.zeros_like(x_pad)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    patch = x_pad[n, c, h_start:h_end, w_start:w_end]

                    # Create a boolean mask of the max element(s) in the patch
                    mask = patch == np.max(patch)

                    # Route the upstream gradient only to the max elements
                    dx_pad[n, c, h_start:h_end, w_start:w_end] += (
                        mask * dout[n, c, i, j]
                    )

    # Strip padding to return original shape
    if padding > 0:
        dx = dx_pad[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_pad

    return dx


def avgpool2d_naive(x, pool_height, pool_width, stride=1, padding=0):
    """
    Naive 2D Average Pooling forward pass.
    """
    N, C, H, W = x.shape

    H_out = 1 + (H + 2 * padding - pool_height) // stride
    W_out = 1 + (W + 2 * padding - pool_width) // stride

    x_pad = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    patch = x_pad[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.mean(patch)

    # We only need the shapes and params for the backward pass, not the actual data
    cache = (x.shape, x_pad.shape, pool_height, pool_width, stride, padding)
    return out, cache


def avgpool2d_naive_backward(dout, cache):
    """
    Backward pass for naive 2D Average Pooling.
    """
    x_shape, x_pad_shape, pool_height, pool_width, stride, padding = cache
    N, C, H_out, W_out = dout.shape

    dx_pad = np.zeros(x_pad_shape)

    # The gradient is distributed equally to all pixels in the window
    pool_area = pool_height * pool_width

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Distribute the gradient
                    grad_dist = dout[n, c, i, j] / pool_area
                    dx_pad[n, c, h_start:h_end, w_start:w_end] += grad_dist

    # Strip padding to return original shape
    if padding > 0:
        dx = dx_pad[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_pad

    return dx
