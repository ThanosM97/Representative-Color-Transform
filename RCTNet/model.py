"""This module implements the RCTNet model used in Kim et al. (2021)
 https://ieeexplore.ieee.org/document/9710400"""
from typing import List

import torch
import torch.nn.functional as F
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
                               (Default: [16, 32, 64, 128, 256, 1024])

    Forward: 
        The output of the forward pass is a list containing the feature maps of 
        the last four blocks of the Encoder that will be combined in the 
        feature fusion module.
    """

    def __init__(
            self, input_dim: int,
            hidden_dims: List[int] = [16, 32, 64, 128, 256, 1024]) -> None:
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h2 = self.initial_blocks(x)
        h3 = self.b3(h2)
        h4 = self.b4(h3)
        h5 = self.b5(h4)
        out = self.b6(h5)

        return [h3, h4, h5, out]


class FeatureFusion(nn.Module):
    """Feature Fusion as described in Kim et al. (2021).

    This class implements the Feature Fusion as described in Kim et al. (2021). 
    In particular it uses a Weighted Bi-directional Feature Pyramid Network 
    (BiFPN), introduced in EfficientDet (https://arxiv.org/abs/1911.09070v7), 
    to fuse multi-scale feature maps, through cross-scale bidirectional 
    connections and a weighted average. The following formula is used for 
    fusion:

            O = \sum_{i=1}^{M} \\frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i

    where w_i is a non-negative learnable weight for the ith feature map I_i


    Args:
        - n_filters (int) : number of filters for the convolutions of the 
                            feature fusion module (Default: 128)
        - input_filters (List) : number of channels of the hidden layers
                                 (Default: [64, 128, 256, 1024])
        - epsilon (float) : epsilon value used in the fusion formula
                            (Default: 0.0001)

    Forward: 
        The output of the forward pass is a list containing the fused maps of 
        the features given as input.
    """

    def __init__(self,
                 n_filters: int = 128,
                 input_filters: List[int] = [64, 128, 256, 1024],
                 epsilon: float = 1e-4) -> None:
        super(FeatureFusion, self).__init__()
        self.input_filters = input_filters
        self.epsilon = epsilon

        # Convolutions for the initial single input nodes
        self.l1_out = convBnSwish(
            in_channels=input_filters[3],
            out_channels=n_filters, kernel_size=1, stride=1, padding=0)
        self.l1_h5 = convBnSwish(
            in_channels=input_filters[2],
            out_channels=n_filters, stride=1, padding=1)
        self.l1_h4 = convBnSwish(
            in_channels=input_filters[1],
            out_channels=n_filters, stride=1, padding=1)
        self.l1_h3 = convBnSwish(
            in_channels=input_filters[0],
            out_channels=n_filters, stride=1, padding=1)

        # Convolutions for the intermediate multi-input nodes
        self.l2_h5 = convBnSwish(
            in_channels=n_filters, out_channels=n_filters, stride=1, padding=1)
        self.l2_h4 = convBnSwish(
            in_channels=n_filters, out_channels=n_filters, stride=1, padding=1)

        # Convolutions for the final nodes
        self.l3_h3 = convBnSwish(
            in_channels=n_filters, out_channels=n_filters, stride=1, padding=1)
        self.l3_h4 = convBnSwish(
            in_channels=n_filters, out_channels=n_filters, stride=1, padding=1)
        self.l3_h5 = convBnSwish(
            in_channels=n_filters, out_channels=n_filters, stride=1, padding=1)
        self.l3_out = convBnSwish(
            in_channels=n_filters, out_channels=n_filters, stride=1, padding=1)

        # Initialize weights for the fusion
        self.l2_w = nn.parameter.Parameter(
            torch.ones((2, 2), dtype=torch.float32))
        self.l3_w1 = nn.parameter.Parameter(
            torch.ones((2, 2), dtype=torch.float32))
        self.l3_w2 = nn.parameter.Parameter(
            torch.ones((2, 3), dtype=torch.float32))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        h3, h4, h5, out = features

        assert h3.shape[1] == self.input_filters[0], \
            ("Number of channels specified in input_filters for h3, does not "
             f"match the input's ({self.input_filters[0]} <> {h3.shape[1]})")
        assert h4.shape[1] == self.input_filters[1], \
            ("Number of channels specified in input_filters for h4, does not "
             f"match the input's ({self.input_filters[1]} <> {h4.shape[1]})")
        assert h5.shape[1] == self.input_filters[2], \
            ("Number of channels specified in input_filters for h5, does not "
             f"match the input's ({self.input_filters[2]} <> {h5.shape[1]})")
        assert out.shape[1] == self.input_filters[3], \
            ("Number of channels specified in input_filters for the output "
             "layer, does not match the input's"
             f" ({self.input_filters[3]} <> {out.shape[1]})")

        # We use ReLU to make sure the weights are non-negative
        l2_w = F.relu(self.l2_w / (torch.sum(self.l2_w, dim=1) + self.epsilon))
        l3_w1 = F.relu(
            self.l3_w1 / (torch.sum(self.l3_w1, dim=1) + self.epsilon))
        l3_w2 = F.relu(
            self.l3_w2 /
            (torch.sum(self.l3_w2, dim=1).unsqueeze(dim=1) + self.epsilon))

        # Initial convolutions of the inputs
        l1_out = self.l1_out(out)
        l1_h5 = self.l1_h5(h5)
        l1_h4 = self.l1_h4(h4)
        l1_h3 = self.l1_h3(h3)

        # Top to bottom calculations
        l2_h5 = self.l2_h5(
            l2_w[0, 0] * F.interpolate(l1_out, size=l1_h5.shape[-1])
            + l2_w[0, 1] * l1_h5
        )
        l2_h4 = self.l2_h4(
            l2_w[1, 0] * F.interpolate(l2_h5, size=l1_h4.shape[-1])
            + l2_w[1, 1] * l1_h4
        )
        l3_h3 = self.l3_h3(
            l3_w1[1, 0] * F.interpolate(l2_h4, size=l1_h3.shape[-1])
            + l3_w1[1, 1] * l1_h3
        )

        # Bottom to top calculations
        l3_h4 = self.l3_h4(
            l3_w2[1, 0] * l2_h4
            + l3_w2[1, 1] * l1_h4
            + l3_w2[1, 2] * F.interpolate(l3_h3, size=l1_h4.shape[-1])
        )
        l3_h5 = self.l3_h5(
            l3_w2[0, 0] * l2_h5
            + l3_w2[0, 1] * l1_h5
            + l3_w2[0, 2] * F.interpolate(l3_h4, size=l1_h5.shape[-1])
        )
        l3_out = self.l3_out(
            l3_w1[0, 0] * l1_out
            + l3_w1[0, 1] * F.interpolate(l3_h5, size=l1_out.shape[-1])
        )

        return [l3_h3, l3_h4, l3_h5, l3_out]
