import numpy as np

from glassboxdl.core.module import Module
from glassboxdl.core.parameter import Parameter
from glassboxdl.core.tensor import Tensor
from glassboxdl.utils.convolution.im2col import col2im_indices, im2col_indices


class Conv2D(Module):
    """
    High-Speed Convolutional Layer supporting multiple execution algorithms
    via an O(1) dispatch table.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        algo: str = "im2col",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.algo = algo

        # O(1) Dispatch Table for instantaneous algorithm routing
        self._dispatch = {
            "im2col": self._forward_im2col,
            "naive": self._forward_naive,
            "fft": self._forward_fft,
            "winograd": self._forward_winograd,
        }

        if self.algo not in self._dispatch:
            raise ValueError(
                f"Unknown algo '{self.algo}'. Choose from {list(self._dispatch.keys())}."
            )

        # He Initialization for weights
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        weight_data = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        ).astype(np.float32)

        bias_data = np.zeros((out_channels, 1)).astype(np.float32)

        # Wrapped in the custom Parameter class so the optimizer updates them!
        self.weight = Parameter(weight_data, requires_grad=True)
        self.bias = Parameter(bias_data, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Routes the forward pass instantly to the chosen algorithm.
        """
        return self._dispatch[self.algo](x)

    def _forward_im2col(self, x: Tensor) -> Tensor:
        """
        High-speed matrix multiplication execution.
        """
        N, C, H, W = x.data.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Extract image patches and flatten them into columns
        x_col = im2col_indices(
            x.data,
            self.kernel_size,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        # Flatten the convolutional filters
        w_col = self.weight.data.reshape(self.out_channels, -1)

        # The core matrix multiplication
        out = w_col @ x_col + self.bias.data

        # Reshape the flat output back into a 4D Image Tensor
        out = out.reshape(self.out_channels, N, out_h, out_w).transpose(1, 0, 2, 3)

        # Wrap in a Tensor for Autograd
        out_tensor = Tensor(
            out, requires_grad=(x.requires_grad or self.weight.requires_grad)
        )

        # The Backward Pass Closure
        def _backward():
            if out_tensor.grad is None:
                return

            # Flatten the incoming gradient to match our matrix shapes
            dout = out_tensor.grad.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

            # Gradient with respect to weights
            if self.weight.requires_grad:
                dw = dout @ x_col.T
                dw = dw.reshape(self.weight.data.shape)
                self.weight.grad = (
                    self.weight.grad + dw if self.weight.grad is not None else dw
                )

            # Gradient with respect to bias
            if self.bias.requires_grad:
                db = np.sum(dout, axis=1).reshape(self.out_channels, 1)
                self.bias.grad = (
                    self.bias.grad + db if self.bias.grad is not None else db
                )

            # Gradient with respect to the input image (routed back through col2im)
            if x.requires_grad:
                dx_col = w_col.T @ dout
                dx = col2im_indices(
                    dx_col,
                    x.data.shape,
                    self.kernel_size,
                    self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
                x.grad = x.grad + dx if x.grad is not None else dx

        out_tensor._backward = _backward
        return out_tensor

    def _forward_naive(self, x: Tensor) -> Tensor:
        """
        Standard nested loops execution. Great for educational comparison,
        but extremely slow compared to im2col.
        """
        N, C, H, W = x.data.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 1. Pad the input
        p = self.padding
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

        # 2. Initialize output
        out = np.zeros((N, self.out_channels, out_h, out_w))

        # 3. The infamous nested loops (Forward Pass)
        for n in range(N):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Extract the patch and do element-wise multiplication
                        patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = (
                            np.sum(patch * self.weight.data[f]) + self.bias.data[f, 0]
                        )

        out_tensor = Tensor(
            out, requires_grad=(x.requires_grad or self.weight.requires_grad)
        )

        # 4. The Autograd Closure (Backward Pass)
        def _backward():
            if out_tensor.grad is None:
                return

            dout = out_tensor.grad

            # Initialize gradients
            dx_padded = np.zeros_like(x_padded)
            dw = np.zeros_like(self.weight.data)
            db = np.zeros_like(self.bias.data)

            # 5. The infamous nested loops (Backward Pass)
            for n in range(N):
                for f in range(self.out_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = j * self.stride
                            w_end = w_start + self.kernel_size

                            # Gradient w.r.t. input (routing dout back to the patch)
                            dx_padded[n, :, h_start:h_end, w_start:w_end] += (
                                self.weight.data[f] * dout[n, f, i, j]
                            )

                            # Gradient w.r.t. weights (patch * dout)
                            patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                            dw[f] += patch * dout[n, f, i, j]

                            # Gradient w.r.t. bias (just summing dout)
                            db[f, 0] += dout[n, f, i, j]

            # Unpad dx to match the original input shape
            if self.padding != 0:
                dx = dx_padded[:, :, p:-p, p:-p]
            else:
                dx = dx_padded

            # Apply gradients
            if self.weight.requires_grad:
                self.weight.grad = (
                    self.weight.grad + dw if self.weight.grad is not None else dw
                )
            if self.bias.requires_grad:
                self.bias.grad = (
                    self.bias.grad + db if self.bias.grad is not None else db
                )
            if x.requires_grad:
                x.grad = x.grad + dx if x.grad is not None else dx

        out_tensor._backward = _backward
        return out_tensor

    def _forward_fft(self, x: Tensor) -> Tensor:
        """
        Fast Fourier Transform execution. Ideal for large kernels.
        """
        raise NotImplementedError("FFT convolution is under construction!")

    def _forward_winograd(self, x: Tensor) -> Tensor:
        """
        Winograd execution. Highly optimized for 3x3 kernels with stride 1.
        """
        raise NotImplementedError("Winograd convolution is under construction!")

    def parameters(self):
        """
        Yields the learnable parameters for the optimizer.
        """
        yield self.weight
        yield self.bias
