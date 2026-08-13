"""
GlassBoxDL - A deep learning framework built from scratch.
"""

__version__ = "0.1.0"

# Expose the absolute core essentials at the top level
# You can also import whole submodules so users can do `glassboxdl.models.UNet`
from . import core, data, layers, models
from .core.module import Module
from .core.parameter import Parameter
from .core.sequential import Sequential
from .core.tensor import Tensor
from .data.dataloader import DataLoader

__all__ = [
    "DataLoader",
    "Module",
    "Parameter",
    "Sequential",
    "Tensor",
    "core",
    "data",
    "layers",
    "models",
]
