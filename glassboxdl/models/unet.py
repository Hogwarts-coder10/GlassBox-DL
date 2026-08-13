from typing import Generator

from glassboxdl.activations.relu import ReLU
from glassboxdl.activations.sigmoid import Sigmoid
from glassboxdl.core.sequential import Sequential
from glassboxdl.core.tensor import Tensor

# Importing your completed layers!
from glassboxdl.layers.convolution import Conv2D
from glassboxdl.layers.pooling import MaxPool2D
from glassboxdl.layers.upsample import Upsample


class DoubleConv:
    """
    Two Convolutions, each followed by a ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.block = Sequential(
            Conv2D(in_channels, out_channels, kernel_size=3, padding=1),
            ReLU(),
            Conv2D(out_channels, out_channels, kernel_size=3, padding=1),
            ReLU(),
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.block(x)

    def parameters(self) -> Generator[Tensor, None, None]:
        yield from self.block.parameters()


class UNet:
    """
    Standard U-Net Architecture for Image Segmentation.
    """

    def __init__(self, in_channels: int = 1, out_classes: int = 1):
        # --- ENCODER ---
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = MaxPool2D(kernel_size=2, stride=2)

        # --- BOTTLENECK ---
        self.bottleneck = DoubleConv(512, 1024)

        # --- DECODER ---
        self.up4 = Upsample(scale_factor=2)
        self.dec4 = DoubleConv(1024 + 512, 512)

        self.up3 = Upsample(scale_factor=2)
        self.dec3 = DoubleConv(512 + 256, 256)

        self.up2 = Upsample(scale_factor=2)
        self.dec2 = DoubleConv(256 + 128, 128)

        self.up1 = Upsample(scale_factor=2)
        self.dec1 = DoubleConv(128 + 64, 64)

        # --- OUTPUT ---
        self.out_conv = Conv2D(64, out_classes, kernel_size=1, padding=0)
        self.out_activation = Sigmoid() if out_classes == 1 else None

    def __call__(self, x: Tensor) -> Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with Skip Connections (concatenating along the channel axis)
        d4 = self.up4(b).concat(e4, axis=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4).concat(e3, axis=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3).concat(e2, axis=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2).concat(e1, axis=1)
        d1 = self.dec1(d1)

        # Output
        out = self.out_conv(d1)
        return self.out_activation(out) if self.out_activation else out

    def parameters(self) -> Generator[Tensor, None, None]:
        # Yield parameters from every single module
        modules = [
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
            self.bottleneck,
            self.dec4,
            self.dec3,
            self.dec2,
            self.dec1,
            self.out_conv,
        ]
        for module in modules:
            yield from module.parameters()
