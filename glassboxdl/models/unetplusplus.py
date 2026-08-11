from typing import Generator

from glassboxdl.activations.sigmoid import Sigmoid
from glassboxdl.core.tensor import Tensor
from glassboxdl.layers.convolution import Conv2D
from glassboxdl.layers.pooling import MaxPool2D
from glassboxdl.layers.upsample import Upsample
from glassboxdl.models.unet import DoubleConv


class UNetPlusPlus:
    """
    U-Net++ Architecture with densely connected nested decoder pathways.
    """

    def __init__(self, in_channels: int = 1, out_classes: int = 1):
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = MaxPool2D(kernel_size=2, stride=2)
        self.up = Upsample(scale_factor=2)

        # Standard Encoder
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # Nested Skip Pathways
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.out_conv = Conv2D(nb_filter[0], out_classes, kernel_size=1, padding=0)
        self.out_activation = Sigmoid() if out_classes == 1 else None

        self.modules = [
            self.conv0_0,
            self.conv1_0,
            self.conv2_0,
            self.conv3_0,
            self.conv4_0,
            self.conv0_1,
            self.conv1_1,
            self.conv2_1,
            self.conv3_1,
            self.conv0_2,
            self.conv1_2,
            self.conv2_2,
            self.conv0_3,
            self.conv1_3,
            self.conv0_4,
            self.out_conv,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested Decoder Pathways
        x0_1 = self.conv0_1(x0_0.concat(self.up(x1_0), axis=1))
        x1_1 = self.conv1_1(x1_0.concat(self.up(x2_0), axis=1))
        x2_1 = self.conv2_1(x2_0.concat(self.up(x3_0), axis=1))
        x3_1 = self.conv3_1(x3_0.concat(self.up(x4_0), axis=1))

        x0_2 = self.conv0_2(x0_0.concat(x0_1, axis=1).concat(self.up(x1_1), axis=1))
        x1_2 = self.conv1_2(x1_0.concat(x1_1, axis=1).concat(self.up(x2_1), axis=1))
        x2_2 = self.conv2_2(x2_0.concat(x2_1, axis=1).concat(self.up(x3_1), axis=1))

        x0_3 = self.conv0_3(
            x0_0.concat(x0_1, axis=1).concat(x0_2, axis=1).concat(self.up(x1_2), axis=1)
        )
        x1_3 = self.conv1_3(
            x1_0.concat(x1_1, axis=1).concat(x1_2, axis=1).concat(self.up(x2_2), axis=1)
        )

        x0_4 = self.conv0_4(
            x0_0.concat(x0_1, axis=1)
            .concat(x0_2, axis=1)
            .concat(x0_3, axis=1)
            .concat(self.up(x1_3), axis=1)
        )

        out = self.out_conv(x0_4)
        return self.out_activation(out) if self.out_activation else out

    def parameters(self) -> Generator[Tensor, None, None]:
        yield from self.parameters()
