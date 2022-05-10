"""This module implements the RCTNet model used in Kim et al. (2021)
 https://ieeexplore.ieee.org/document/9710400"""
from typing import List

import torch
from torch import nn


def convBnSwish(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1) -> nn.Sequential:
    """Conv-bn-swish block.

    This function implements a Conv-bn-swish block as described in 
    Kim et al. (2021). Each block contains a convolution followed by
    a Batch Normalization and a Swish activation layer.

    Args:
        - in_channels (int) : Input channel dimensions
        - out_channels (int) : Output channel dimensions
        - kernel_size (int) : Kernel size for the convolution (Default: 3)
        - stride (int) : Stride for the convolution (Default: 2)
        - padding (int) : Padding size for the convolution (Default: 1)

    Returns:
        A Conv-bn-swish sequential block.

    """
    block = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    )
    return block


class Encoder(nn.Module):
    """Encoder network as described in Kim et al. (2021).

    Args:
        - input_dim (int) : input channel dimension
        - hidden_dims (List) : dimensions of the hidden layers

    Forward: 

    The output of the forward pass is a list containing the feature maps of the 
    last four blocks of the Encoder that will be combined in the feature fusion
    module.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List = [16, 32, 64, 128, 256, 1024]) -> None:
        super(Encoder, self).__init__()

        self.initial_blocks = nn.Sequential(
            # IN: input_dimx256x256 / OUT: hidden_dims[0]x128x128
            convBnSwish(input_dim, hidden_dims[0]),

            # IN: hidden_dims[0]x128x128 / OUT: hidden_dims[1]x64x64
            convBnSwish(hidden_dims[0], hidden_dims[1])
        )

        # IN: hidden_dims[1]x64x64 / OUT: hidden_dims[2]x32x32
        self.b3 = convBnSwish(hidden_dims[1], hidden_dims[2])

        # IN: hidden_dims[2]x32x32 / OUT: hidden_dims[3]x16x16
        self.b4 = convBnSwish(hidden_dims[2], hidden_dims[3])

        # IN: hidden_dims[3]x16x16 / OUT: hidden_dims[4]x8x8
        self.b5 = convBnSwish(hidden_dims[3], hidden_dims[4])

        # IN: hidden_dims[4]x8x8 / OUT: hidden_dims[5]x1x1
        self.b6 = nn.Sequential(
            convBnSwish(hidden_dims[4], hidden_dims[5], kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x: torch.Tensor) -> List:
        h2 = self.initial_blocks(x)
        h3 = self.b3(h2)
        h4 = self.b4(h3)
        h5 = self.b5(h4)
        out = self.b6(h5)

        return [h3, h4, h5, out]
