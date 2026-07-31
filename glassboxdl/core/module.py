from abc import ABC, abstractmethod
from collections import OrderedDict

# We import Tensor for type checking, but do it safely to avoid circular imports later
from glassboxdl.core.tensor import Tensor


class Module(ABC):
    """
    Base class for all neural network modules in GlassBoxDL.
    Your models should also subclass this class.
    """

    def __init__(self):
        # OrderedDicts preserve the order in which layers/parameters are added,
        # which is crucial for reproducibility and predictable state_dicts.
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self.training = True

    def __setattr__(self, name, value):
        """
        Intercepts attribute assignment to automatically register
        parameters (Tensors with requires_grad=True) and sub-modules.
        """
        # Protect against assigning before super().__init__() is called in subclasses
        if not hasattr(self, "_modules") or not hasattr(self, "_parameters"):
            super().__setattr__(name, value)
            return

        # Automatically register sub-modules
        if isinstance(value, Module):
            self._modules[name] = value

        # Automatically register parameters
        elif isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value

        # Standard attribute assignment
        super().__setattr__(name, value)

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Defines the computation performed at every call."""
        pass

    def __call__(self, *inputs, **kwargs):
        """Allows the module to be called like a function."""
        return self.forward(*inputs, **kwargs)

    # --- Parameter & Module Management ---

    def parameters(self):
        """Recursively yields all parameters in this module and its sub-modules."""
        yield from self._parameters.values()

        for module in self._modules.values():
            yield from module.parameters()

    def children(self):
        """Yields immediate sub-modules."""
        yield from self._modules.values()

    def modules(self):
        """Recursively yields all modules in the network, including self."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    # --- Training State ---

    def train(self, mode=True):
        """Sets the module and all sub-modules to training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Sets the module and all sub-modules to evaluation mode."""
        return self.train(False)

    # --- Gradient Management ---

    def zero_grad(self):
        """Zeroes out the gradients of all parameters."""
        for param in self.parameters():
            if param.grad is not None:
                # Modifying in-place using NumPy
                param.grad.fill(0.0)

    # --- Serialization ---

    def state_dict(self):
        """Returns a dictionary containing a whole state of the module."""
        raise NotImplementedError("state_dict() is not yet implemented.")

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from state_dict into this module."""
        raise NotImplementedError("load_state_dict() is not yet implemented.")
