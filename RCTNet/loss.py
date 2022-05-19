import torch
import torchvision.transforms as T
from torch import nn
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor


class Loss():
    """Loss function as described in Kim et al. (2021). The loss function 
    comprises of two terms: the absolute error between the predicted and ground
    truth high quality image (L1 loss), and their difference in an embedding 
    space, namely the sum of the L1 losses of the embedded representations of 
    the predicted and ground truth high quality image. The embedded 
    representations used are the outputs of the 2nd, 4th, 6th layer in VGG-16 
    by Simonyan et al. (2015), which is pre-trained on the ImageNet dataset.
    """

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Definition of L1 loss function
        self.l1_criterion = nn.L1Loss()
        self.l1_criterion.to(self.device)

    def estimate_vgg_loss(
            self, Y: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """vgg16 loss function estimation.

            This function calculates the vgg16 loss.

            Args:
                - Y (torch.Tensor) : the enhanced output image Y of the RCTNet.
                - target_img (torch.Tensor) : the ground truth, enhanced image.

            Returns:
                The estimated vgg16 loss.
        """
        loss = 0
        # load pretrained vgg16 model
        vgg = vgg16(pretrained=True)

        # From these nodes we want to extract only the feature representations
        # for classification layers 2, 4, 6.
        return_layers = {'features.3': 'out_layer2',
                         'features.8': 'out_layer4',
                         'features.13': 'out_layer6'}

        # Creates a new graph module that returns intermediate nodes from a
        # given model as dictionary with user specified keys as strings, and
        # the requested outputs as values
        feature_extractor = create_feature_extractor(
            vgg, return_nodes=return_layers)

        # transform input images to match the to input format of the pytorch
        # pretrained vgg16 model.
        Y = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(
            Y / 255)
        target_img = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(
            target_img / 255)

        # forward pass to get the feature representations of the selected vgg16
        # layers for both the enhanced output image Y and the ground truth
        # enhanced image
        Y_embed = feature_extractor(Y)
        target_img_embed = feature_extractor(target_img)

        # sum the l1 losses of the embedded representations
        for key in Y_embed.keys():
            loss += self.l1_criterion(Y_embed[key], target_img_embed[key])

        return loss

    def estimate_loss(
            self, Y: torch.Tensor, target_img: torch.Tensor,
            balance_lambda: float) -> torch.Tensor:
        """Loss function estimation.

            This function calculates the final loss.

            Args:
                - Y (torch.Tensor) : the enhanced output image Y of the RCTNet.
                - target_img (torch.Tensor) : the ground truth, enhanced image.

            Returns:
                The estimated loss.

        """
        # Calculate the mean absolute error between the predicted and
        # ground-truth enhanced images.
        l1 = self.l1_criterion(Y, target_img)
        # Calculate error in the embedding space of the k-th layer in VGG-16
        vgg = self.estimate_vgg_loss(Y, target_img)
        # Balance two terms with hyper parameter balance_lambda
        loss = l1 + balance_lambda * vgg

        return loss
