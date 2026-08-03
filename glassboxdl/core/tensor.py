import numpy as np


class Tensor:
    """
    A foundational Tensor class for GlassBoxDL.
    Wraps a NumPy array and tracks the computational graph for automatic differentiation.
    """

    def __init__(self, data, requires_grad=False, _children=()):
        # 1. NumPy Data Storage
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        # 2. Graph State
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._backward = lambda: None

        # 3. Gradient Tracking
        # Initialize as zeros if gradients are required, else None to save memory
        self.grad = np.zeros_like(self.data, dtype=float) if requires_grad else None

    # --- Properties ---

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    # --- Representation ---

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n" + " " * 7)
        return f"Tensor({data_str}, requires_grad={self.requires_grad})"

    # --- Conversion Helpers ---

    def numpy(self):
        """Returns a copy of the underlying NumPy array."""
        return self.data.copy()

    def tolist(self):
        """Converts the underlying data to a nested Python list."""
        return self.data.tolist()

    # --- Indexing ---

    def __getitem__(self, item):
        """Basic indexing. (Note: Backward pass for slicing requires advanced handling, kept read-only for now)."""
        return Tensor(
            self.data[item], requires_grad=self.requires_grad, _children=(self,)
        )

    # --- Mathematical Dunder Methods ---

    def _unbroadcast_grad(self, grad, target_shape):
        """Helper to sum out broadcasted dimensions during backpropagation."""
        ndims_added = grad.ndim - len(target_shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast_grad(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast_grad(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast_grad(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast_grad(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast_grad(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast_grad(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast_grad(out.grad / other.data, self.shape)
            if other.requires_grad:
                self_grad_other = -out.grad * self.data / (other.data**2)
                other.grad += self._unbroadcast_grad(self_grad_other, other.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other))
        out.requires_grad = self.requires_grad or other.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                # Gradient of A @ B with respect to A is dY @ B.T
                self.grad += out.grad @ other.data.swapaxes(-1, -2)
            if other.requires_grad:
                # Gradient of A @ B with respect to B is A.T @ dY
                other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return Tensor(other) @ self

    def __pow__(self, power):
        # Limiting power to scalar values for simplicity and educational readability
        assert isinstance(power, (int, float)), (
            "Only integer/float powers are supported."
        )
        out = Tensor(self.data**power, _children=(self,))
        out.requires_grad = self.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, _children=(self,))
        out.requires_grad = self.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Triggers backpropogation through the computational graph
        """

        # Doing this to prevent circular prevent issues
        from glassboxdl.core.autograd import backward as autograd_backward

        autograd_backward(self)

    def reshape(self, *shape):
        """
        Reshapes the tensor while tracking gradients.
        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        out = Tensor(self.data.reshape(*shape), _children=(self,))
        out.requires_grad = self.requires_grad

        if out.requires_grad:
            out.grad = np.zeros_like(out.data, dtype=float)

        def _backward():
            if self.requires_grad and out.grad is not None:
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out
