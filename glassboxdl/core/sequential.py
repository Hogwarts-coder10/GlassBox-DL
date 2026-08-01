from glassboxdl.core.module import Module


class Sequential(Module):
    """
    A sequential container for neural network modules.

    Modules are added to it in the exact order they are passed to the constructor.
    The forward() method automatically chains them together, feeding the output
    of one module as the input to the next.
    """

    def __init__(self, *args):
        super().__init__()

        for idx, module in enumerate(args):
            if not isinstance(module, Module):
                raise TypeError(
                    f"Argument at index {idx} is not a subclass of Module. Got {type(module)}."
                )

            # Using setattr triggers the base Module.__setattr__,
            # automatically registering it in self._modules with the index as its name.
            setattr(self, str(idx), module)

    def forward(self, x):
        """
        Passes the input tensor sequentially through all registered modules.
        """

        for module in self._modules.values():
            x = module(x)

        return x

    # -- List-Like behaviour --
    # These dunder methods make the container feel pythonic and intuitive

    def __len__(self):
        """
        Returns the number of modules in a sequential container.
        """

        return len(self._modules)

    def __iter__(self):
        """
        Allows iterating over the modules.
        """

        return iter(self._modules.values())

    def __getitem__(self, idx):
        """
        Allows indexing and slicing into the sequential container.
        Example: model[0] returns the first layer.
        """

        if isinstance(idx, slice):
            # if sliced, return a new Sequential container with the subset of modules
            return Sequential(*list(self._modules.values())[idx])

        # Standard integer indexing
        return list(self._modules.values())[idx]
