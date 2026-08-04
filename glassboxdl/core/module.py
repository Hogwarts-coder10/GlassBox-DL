from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from glassboxdl.core.parameter import Parameter


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

        # Automatically register explicit parameters
        elif isinstance(value, Parameter):
            self._parameters[name] = value

        # Standard attribute assignment
        super().__setattr__(name, value)

    @abstractmethod
    def forward(self, *inputs, **kwargs) -> Any:
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """

        raise NotImplementedError("Subclasses must immplement the forward method")

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
    # --- Parameter & Module Management (Add this new method) ---

    def named_parameters(self, prefix=""):
        """
        Recursively yields (name, parameter) tuples for all parameters.
        This creates hierarchical names like 'fc1.weight' for state_dicts.
        """
        # Yield direct parameters of this module
        for name, param in self._parameters.items():
            yield prefix + name, param

        # Recursively yield parameters of sub-modules
        for module_name, module in self._modules.items():
            sub_prefix = prefix + module_name + "."
            yield from module.named_parameters(prefix=sub_prefix)

    def state_dict(self):
        """
        Returns an OrderedDict containing the whole state of the module.
        Extracts the pure NumPy arrays to ensure safe and clean serialization.
        """
        from collections import OrderedDict

        state = OrderedDict()
        for name, param in self.named_parameters():
            # Store a copy of the pure NumPy array, not the Parameter object
            state[name] = param.numpy()
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters from a state_dict into this module's parameters.

        Args:
            state_dict (dict): A dictionary containing parameters (NumPy arrays).
            strict (bool): If True, strictly enforces that the keys in state_dict
                            match the keys returned by this module's state_dict().
        """
        own_state = self.state_dict()

        for name, param in self.named_parameters():
            if name in state_dict:
                saved_array = state_dict[name]
                # Validate shapes to prevent silent broadcasting bugs
                if param.shape != saved_array.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected {param.shape}, got {saved_array.shape}"
                    )

                # Overwrite the underlying NumPy data in-place
                param.data = saved_array.copy()
            elif strict:
                raise KeyError(f"Missing key in state_dict: '{name}'")

        if strict:
            # Check for unexpected keys in the loaded state_dict
            unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
            if unexpected_keys:
                raise KeyError(f"Unexpected key(s) in state_dict: {unexpected_keys}")
