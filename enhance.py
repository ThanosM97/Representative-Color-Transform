"""This module enhances an image or a directory of images using RCTNet."""
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from RCTNet.model import RCTNet


class EnhanceDataset(Dataset):
    """Dataset for image enhancement using RCTNet.

    Args:
        - imx_dir (str) : Path to the directory containing the low quality 
                          images to be enhanced
    """

    def __init__(
            self, img_dir):
        super(EnhanceDataset, self).__init__()
        self.img_dir = img_dir

        self.images = [file.name for file in Path(self.img_dir).glob(
            '*') if file.suffix in ['.png', '.jpg', '.jpeg']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_X_path = str(Path(self.img_dir, self.images[idx]))
        image_X = read_image(img_X_path).float()

        return image_X


def main(args):
    # Set path for checkpoints
    path = Path(args.image)

    # Unless otherwise specified, model runs on CUDA if available
    if args.device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Initialize RCT model
    if args.config:
        with open(args.config) as fp:
            cfg = json.load(fp)  # load model configurations
        model = RCTNet(
            in_channels=cfg["in_channels"],
            hidden_dims=cfg["hidden_dims"],
            c_prime=cfg["c_prime"],
            epsilon=cfg["epsilon"],
            c_G=cfg["c_G"],
            n_G=cfg["n_G"],
            c_L=cfg["c_L"],
            n_L=cfg["n_L"],
            grid_size=cfg["grid_size"],
            device=device
        )
    else:
        model = RCTNet(device=device)

    # Move model to device selected
    model = model.to(device)

    # Load model's weights
    model.load_state_dict(torch.load(
        args.checkpoint, map_location=torch.device(device)))

    model.eval()

    # Transform to convert torch.Tensor to PILImage
    transform = T.ToPILImage()

    if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg']:
        img = read_image(str(path)).float()
        img = torch.unsqueeze(img, 0).repeat(2, 1, 1, 1)

        with torch.no_grad():
            enhanced = torch.clamp(model(img)[0], max=255.0)
            enhanced_PIL = transform(enhanced / 255.0)

            save = path.with_stem(path.stem + "-enhanced")
            enhanced_PIL.save(save)
            exit()

    if path.is_dir():
        # Initialize dataloaders
        dataset = EnhanceDataset(path)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size)

        for batch, x in enumerate(dataloader):
            with torch.no_grad():
                y = torch.clamp(model(x), max=255.0)

            for i, enhanced_img in enumerate(y):
                enhanced_img = transform(enhanced_img / 255.0)
                save = path / Path(dataset.images[i + batch*args.batch_size])
                enhanced_img.save(save.with_stem(save.stem + "-enhanced"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image', required=True,
        help='Path to an image or a directory of images to be enhanced')

    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to previous checkpoint')

    parser.add_argument(
        '--config', default=None, type=str,
        help="Path to configurations file for the RCTNet model")

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Number of samples per minibatch')

    parser.add_argument('--device', default=None, choices=["cpu", "cuda"],
                        type=str, help='Device to use for training')

    args = parser.parse_args()

    main(args)
