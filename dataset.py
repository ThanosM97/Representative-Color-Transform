"""This module implements a custom Dataset for the training and
evaluation of the RCTNet model by Kim et al. (2021)
https://ieeexplore.ieee.org/document/9710400"""
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class RCTDataset(Dataset):
    """Dataset for RCTNet.

    This class impelments a custom Dataset for the training of the RCTNet
    model. As described in Kim et al. (2021), the data are augmented using
    a random crop, followed by a random rotate of a multiple of 90 degrees.

    Each returned sample contains a pair of a low quality image and its 
    enhanced counterpart.

    Args:
        - imx_X_dir (str) : Path to the directory containing the low quality 
                            images to be enhanced
        - img_target_dir (str) : Path to the enhanced target images
    """

    def __init__(
            self, img_X_dir, img_target_dir, augmentation=True):
        super(RCTDataset, self).__init__()
        self.img_X_dir = img_X_dir
        self.img_target_dir = img_target_dir
        self.augmentation = augmentation

        self.images = [file.name for file in Path(self.img_X_dir).glob(
            '*') if file.suffix in ['.png', '.jpg', '.jpeg']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_X_path = str(Path(self.img_X_dir, self.images[idx]))
        img_target_path = str(Path(self.img_target_dir, self.images[idx]))
        image_X = read_image(img_X_path).float()
        image_target = read_image(img_target_path).float()

        if self.augmentation:
            # Resize
            size = tuple(image_X.shape[1:])
            resized = tuple([int(d*1.25) for d in size])
            image_X = TF.resize(img=image_X, size=resized)
            image_target = TF.resize(img=image_target, size=resized)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image_X, output_size=size)
            image_X = TF.crop(image_X, i, j, h, w)
            image_target = TF.crop(image_target, i, j, h, w)

            # Random rotation by a multiple of 90 degrees
            angle = int(torch.randint(0, 3, size=(1,))) * 90
            image_X = TF.rotate(image_X, angle)
            image_target = TF.rotate(image_target, angle)

        return image_X, image_target
