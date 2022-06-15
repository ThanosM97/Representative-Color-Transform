"""This module implements the RCTNet model used in Kim et al. (2021)
 https://ieeexplore.ieee.org/document/9710400"""
import torch
import torch.nn.functional as F
from torch import nn

from .utils import padding, remove_padding


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
        - in_channels (int) : Number of channels in input (Default: 3)
        - hidden_dims (list) : Dimensions of filters in hidden layers
                               (Default: [16, 32, 64, 128, 256, 1024])

    Forward: 
        The output of the forward pass is a list containing the feature maps of 
        the last four blocks of the Encoder that will be combined in the 
        feature fusion module.
    """

    def __init__(
            self, in_channels: int = 3,
            hidden_dims: list[int] = [16, 32, 64, 128, 256, 1024]) -> None:
        """
        Args:
            - in_channels (int) : Number of channels in input (Default: 3)
            - hidden_dims (list) : Dimensions of filters in hidden layers
                            (Default: [16, 32, 64, 128, 256, 1024])
        """
        super(Encoder, self).__init__()

        self.initial_blocks = nn.Sequential(
            # IN: in_channelsx256x256 / OUT: hidden_dims[0]x128x128
            convBnSwish(in_channels, hidden_dims[0]),

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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Resize image to 256x256
        x = F.interpolate(x, size=256)

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
        - n_filters (int) : Number of filters for the convolutions of the 
                            feature fusion module (Default: 128)
        - input_filters (list) : Number of channels of the hidden layers
                                 (Default: [64, 128, 256, 1024])
        - epsilon (float) : Epsilon value used in the fusion formula
                            (Default: 0.0001)

    Forward: 
        The output of the forward pass is a list containing the fused maps of 
        the features given as input.
    """

    def __init__(self,
                 n_filters: int = 128,
                 input_filters: list[int] = [64, 128, 256, 1024],
                 epsilon: float = 1e-4) -> None:
        """
        Args:
            - n_filters (int) : Number of filters for the convolutions of the 
                                feature fusion module (Default: 128)
            - input_filters (list) : Number of channels of the hidden layers
                                    (Default: [64, 128, 256, 1024])
            - epsilon (float) : Epsilon value used in the fusion formula
                                (Default: 0.0001)
        """
        super(FeatureFusion, self).__init__()
        self.input_filters = input_filters
        self.epsilon = epsilon

        # Convolutions for the initial single input nodes
        self.l1_out = convBnSwish(
            in_channels=input_filters[3], kernel_size=1,
            out_channels=n_filters, stride=1, padding=0)
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
            in_channels=n_filters, out_channels=n_filters,
            kernel_size=1, stride=1, padding=0)

        # Initialize weights for the fusion
        self.l2_w = nn.parameter.Parameter(
            torch.ones((2, 2), dtype=torch.float32))
        self.l3_w1 = nn.parameter.Parameter(
            torch.ones((2, 2), dtype=torch.float32))
        self.l3_w2 = nn.parameter.Parameter(
            torch.ones((2, 3), dtype=torch.float32))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
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


class GlobalRCT(nn.Module):
    """Global RCT as described in Kim et al. (2021).

    This class implements the Global Representative Color Transform as 
    described in Kim et al. (2021). 

    In particular, given a feature representation Z_G at the coarest-scale in 
    the feature fusion module, we extract representative features R_G and
    transformed colors T_G using two different conv-bn-swish-conv blocks. In
    addition, we extract image features F from the input image using yet 
    another conv-bn-swish-conv block. The enhanced image Y_G is then computed
    using the following formula:

            Y = A \cdot T_G^T

    where A is the attention matrix computed as follows:

            A = softmax(\\frac{F_r \cdot R_G}{\sqrt{C}})

    where F_r is the reshaped tensor of F and C is the feature dimension for
    the representative features of the GlobalRCT (`c`).


    Args:
        - c_prime (int) : Feature dimension (Default: 128)
        - c (int) : Feature dimension for the representative 
                    features of the GlobalRCT (Default: 16)
        - n_G (int) : Number of representative colors (Default: 64)

    Forward: 
        The output of the forward pass is a Tensor of the enhanced images Y_G.
    """

    def __init__(self,
                 c_prime: int = 128,
                 c: int = 16,
                 n_G: int = 64
                 ) -> None:
        """
        Args:
            - c_prime (int) : Feature dimension (Default: 128)
            - c (int) : Feature dimension for the representative 
                        features of the GlobalRCT (Default: 16)
            - n_G (int) : Number of representative colors (Default: 64)
        """
        super(GlobalRCT, self).__init__()

        self.c = c
        self.n_G = n_G

        # conv-bn-swish-conv block for the representative features
        self.convR_G = nn.Sequential(
            convBnSwish(in_channels=c_prime,
                        out_channels=c_prime,
                        stride=1,
                        padding=1),
            nn.Conv2d(
                in_channels=c_prime,
                out_channels=c*n_G,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # conv-bn-swish-conv block for the transformed colors
        self.convT_G = nn.Sequential(
            convBnSwish(in_channels=c_prime,
                        out_channels=c_prime,
                        stride=1,
                        padding=1),
            nn.Conv2d(
                in_channels=c_prime,
                out_channels=3*n_G,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self,
                f: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = f.shape

        f_r = f.reshape(batch_size, self.c, h*w)
        f_r = f_r.transpose(1, 2)

        # Get Global Representative features R_G
        r_G = self.convR_G(features)  # self.c*self.n_G x 1 x 1
        r_G = r_G.reshape(batch_size, self.c, self.n_G)

        # Get the attention matrix A
        # hw x self.n
        attention = F.softmax(torch.bmm(f_r, r_G) / self.c**0.5, dim=2)

        #  Get Global Transformed Colors T_G
        t_G = self.convT_G(features)  # 3*self.n_G x 1 x 1
        t_G = t_G.reshape(batch_size, 3, self.n_G)

        y_G = torch.bmm(attention, t_G.transpose(1, 2))  # h*w x 3

        return y_G.transpose(1, 2).reshape(batch_size, 3, h, w)


class LocalRCT(nn.Module):
    """Local RCT as described in Kim et al. (2021).

    This class implements the Local Representative Color Transform as 
    described in Kim et al. (2021). 

    In particular, given a feature representation Z_L at the finest-scale in 
    the feature fusion module, we extract representative features R_L and
    transformed colors T_L using two different conv-bn-swish-conv blocks. In
    addition, we extract image features F from the input image using yet 
    another conv-bn-swish-conv block. Subsequently, a `grid_size` uniform
    meshgrid is set on the input image and each generated grid B_k has four
    corner points (32x32 in total), which correspond to four representative
    features and transformed colors obtained from their spatial sizes at the
    corresponding positions.

    Then for each of those girds B_k, we apply the same formulaτions as in 
    Global Representative Color Transform. Specifically, the enhanced image 
    Y_L_k of B_k is computed using the following formula:

            Y_L_k = A_k \cdot T_L_k^T

    where A_k is the attention matrix computed as follows:

            A_k = softmax(\\frac{F_k \cdot R_L_k}{\sqrt{C}})

    where F_k is the grid feature map of F and C is the feature dimension for
    the representative features of the LocalRCT (`c`).


    Args:
        - grid_size (int) : Size of mesh grid (Default: 31)
        - c_prime (int) : Feature dimension (Default: 128)
        - c (int) : Feature dimension for the representative 
                    features of the LocalRCT (Default: 16)
        - n_L (int) : Number of representative colors (Default: 16)

    Forward: 
        The output of the forward pass is a Tensor of the enhanced images Y_L.
    """

    def __init__(self,
                 grid_size: int = 31,
                 c_prime: int = 128,
                 c: int = 16,
                 n_L: int = 16,
                 device: str = "cpu"
                 ) -> None:
        """
        Args:
            - grid_size (int) : Size of mesh grid (Default: 31)
            - c_prime (int) : Feature dimension (Default: 128)
            - c (int) : Feature dimension for the representative 
                        features of the LocalRCT (Default: 16)
            - n_L (int) : Number of representative colors (Default: 16)
            - device (str) : Device to use (Default: cpu)
        """
        super(LocalRCT, self).__init__()

        self.grid_size = grid_size
        self.c = c
        self.n_L = n_L
        self.device = device

        # conv-bn-swish-conv block for the representative features
        self.convR_L = nn.Sequential(
            convBnSwish(in_channels=c_prime,
                        out_channels=c_prime,
                        stride=1,
                        padding=1),
            nn.Conv2d(
                in_channels=c_prime,
                out_channels=c*n_L,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # conv-bn-swish-conv block for the transformed colors
        self.convT_L = nn.Sequential(
            convBnSwish(in_channels=c_prime,
                        out_channels=c_prime,
                        stride=1,
                        padding=1),
            nn.Conv2d(
                in_channels=c_prime,
                out_channels=3*n_L,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self,
                f: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        batch_size = f.shape[0]
        spatial_size = features.shape[-1]

        assert spatial_size == self.grid_size+1, \
            ("The number of corner points derived by the grid_size, as "
             "(grid_size + 1)x(grid_size + 1), should be the same as the "
             "spatial size of the input features "
             f"({spatial_size}x{spatial_size}).")

        # Pad features for them to be divisible by grid_size
        f, h_new, w_new, paddings = padding(f, self.grid_size)

        # Size of the image feature patches Fk
        w_patch = int(w_new / self.grid_size)
        h_patch = int(h_new / self.grid_size)

        # Get Local Representative Colors R_L
        r_L = self.convR_L(features)  # self.c*self.n x spatial x spatial
        r_L = r_L.reshape(
            batch_size, self.c, self.n_L, spatial_size, spatial_size)
        # spatial x spatial x self.c x n_L
        r_L = r_L.permute(0, 3, 4, 1, 2)

        # Get Local Transformed Colors T_L
        t_L = self.convT_L(features)  # 3*self.n x spatial x spatial
        t_L = t_L.reshape(batch_size, 3, self.n_L, spatial_size, spatial_size)
        # spatial x spatial x channels x n_L
        t_L = t_L.permute(0, 3, 4, 1, 2)

        y_L = torch.zeros(batch_size, 3, h_new, w_new).to(device=self.device)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get grid feature F_k
                f_k = f[:, :, i*h_patch:(i+1)*h_patch,
                        j*w_patch:(j+1)*w_patch]  # self.c x H x W
                f_k = f_k.reshape(batch_size, self.c, h_patch*w_patch)
                f_k = f_k.transpose(1, 2)  # HW x self.c

                # CP (Corner Point) 1
                r_k_cp = r_L[:, i, j, :, :]  # self.c x self.n_L
                t_k_cp = t_L[:, i, j, :, :]  # channels x n_L

                r_k = r_k_cp  # self.c x self.n_L
                t_k = t_k_cp  # channels x n_L

                # CP 2
                r_k_cp = r_L[:, i+1, j, :, :]
                t_k_cp = t_L[:, i+1, j, :, :]

                r_k = torch.cat((r_k, r_k_cp), dim=2)  # self.c x 2*self.n_L
                t_k = torch.cat((t_k, t_k_cp), dim=2)  # channels x 2*n_L

                # CP 3
                r_k_cp = r_L[:, i, j+1, :, :]
                t_k_cp = t_L[:, i, j+1, :, :]

                r_k = torch.cat((r_k, r_k_cp), dim=2)  # self.c x 3*self.n_L
                t_k = torch.cat((t_k, t_k_cp), dim=2)  # channels x 3*n_L

                # CP 4
                r_k_cp = r_L[:, i+1, j+1, :, :]
                t_k_cp = t_L[:, i+1, j+1, :, :]

                r_k = torch.cat((r_k, r_k_cp), dim=2)  # self.c x 4*self.n_L
                t_k = torch.cat((t_k, t_k_cp), dim=2)  # channels x 4*n_L

                # Get the attention matrix A
                # HW x 4*self.n_L
                attention = F.softmax(torch.bmm(f_k, r_k) / self.c**0.5, dim=2)

                # Enhanced B_k
                y_L_k = torch.bmm(attention, t_k.transpose(1, 2))  # HW x 3

                # Add enhanced B_k grid to Y_L
                y_L[:, :, i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)
                    * w_patch] = y_L_k.transpose(1, 2).reshape(
                        batch_size, 3, h_patch, w_patch)

        # Remove padding to return to input's size
        y_L = remove_padding(y_L, paddings)

        return y_L


class RCTNet(nn.Module):
    """RCTNet as described in Kim et al. (2021).

    This class implements the Representative Color Transform Network as 
    described in Kim et al. (2021). It utilizes the four modules: Encoder, 
    Feature Fusion, GlobalRCT, and LocalRCT. Given an image X, RCTNet produces 
    a high-quality image using the following formula:

                \\tilde{Y} = \\alpha Y_G + \\beta Y_L

    where Y_G and Y_L are the enhanced images produced from the GlobalRCT and
    LocalRCT modules, respectively. Also, alpha and beta are non-negative
    learnable weights used to combine the two enhanced images.

    Args:
        - in_channels (int) : Number of channels in input (Default: 3)
        - hidden_dims (List[int]) : Dimensions of filters in the hidden layers 
                                    of the Encoder 
                                    (Default: [16, 32, 64, 128, 256, 1024])
        - c_prime (int) : Feature dimension (Default: 128)
        - epsilon (float) : Epsilon value used in the feature fusion formula
                            (Default: 0.0001)
        - c_G (int) : Feature dimension for the representative 
                      features of the GlobalRCT (Default: 16)
        - n_G (int) : Number of representative colors for GlobalRCT 
                      (Default: 64)
        - c_L (int) : Feature dimension for the representative 
                      features of the LocalRCT (Default: 16)
        - n_L (int) : Number of representative colors for LocalRCT 
                      (Default: 16)
        - c_F (int) : Feature dimension for the image features (Default: 16)
        - grid_size (int) : Size of mesh grid for LocalRCT (Default: 31)

    Forward: 
       The output of the forward pass is a Tensor of the enhanced images Y
       of the input X.
    """

    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: list = [16, 32, 64, 128, 256, 1024],
                 c_prime: int = 128,
                 epsilon: float = 1e-4,
                 c_G: int = 16,
                 n_G: int = 64,
                 c_L: int = 16,
                 n_L: int = 16,
                 c_F: int = 16,
                 grid_size: int = 31,
                 device: str = "cpu") -> None:
        """
        Args:
        - in_channels (int) : Number of channels in input (Default: 3)
        - hidden_dims (List[int]) : Dimensions of filters in the hidden layers 
                                    of the Encoder 
                                    (Default: [16, 32, 64, 128, 256, 1024])
        - c_prime (int) : Feature dimension (Default: 128)
        - epsilon (float) : Epsilon value used in the feature fusion formula
                            (Default: 0.0001)
        - c_G (int) : Feature dimension for the representative 
                      features of the GlobalRCT (Default: 16)
        - n_G (int) : Number of representative colors for GlobalRCT 
                      (Default: 64)
        - c_L (int) : Feature dimension for the representative 
                      features of the LocalRCT (Default: 16)
        - n_L (int) : Number of representative colors for LocalRCT 
                      (Default: 16)
        - c_F (int) : Feature dimension for the image features (Default: 16)
        - grid_size (int) : Size of mesh grid for LocalRCT (Default: 31)
        """
        super(RCTNet, self).__init__()

        # Initialize Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims
        )

        # Initialize Feature Fusion module
        self.feature_fusion = FeatureFusion(
            n_filters=c_prime,
            input_filters=hidden_dims[-4:],
            epsilon=epsilon
        )

        # conv-bn-swish-conv block for the image features
        self.image_features = nn.Sequential(
            convBnSwish(in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        padding=1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=c_F,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # Initialize Local and Global RCTs
        self.global_rct = GlobalRCT(
            c_prime=c_prime,
            c=c_G,
            n_G=n_G
        )
        self.local_rct = LocalRCT(
            grid_size=grid_size,
            c_prime=c_prime,
            c=c_L,
            n_L=n_L,
            device=device
        )

        # Initialize learnable parameters alpha and beta
        self.weights = nn.parameter.Parameter(
            torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        fused_features = self.feature_fusion(features)

        image_features = self.image_features(x)

        y_G = self.global_rct(
            image_features, fused_features[-1])
        y_L = self.local_rct(image_features, fused_features[0])

        # We use ReLU to keep the learnable parameters non-negative
        y = F.relu(self.weights[0]) * y_G + F.relu(self.weights[1]) * y_L

        return y
