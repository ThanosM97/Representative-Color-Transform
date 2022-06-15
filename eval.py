"""This module implements the evaluation of RCTNet."""
import argparse
from pathlib import Path
import random
import json

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import RCTDataset
from RCTNet.model import RCTNet


def density(scores: list[float], filename: str, xlabel: str) -> None:
    """Plot a density figure for `scores` list of values.

    Arguments:
        - scores (list) : List of values to plot density figure for
        - filename (str) : Filename to use to save the figure
        - xlabel (st) : The x-axis label for the figure
    """
    bins = 10 if len(scores) < 50 else 50
    density = stats.gaussian_kde(scores)
    plt.figure()
    _, x, _ = plt.hist(scores, bins=bins,
                       histtype=u'step', density=True)
    plt.plot(x, density(x))
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.savefig(filename)


def log(scores: list[float], filename: str, metric: str) -> None:
    """Create or append to the log file of the evaluation.

    The log file consists of the following metrics:
        - Mean
        - Standard deviation
        - Minimum
        - Maximum
    of the `scores`, for the selected `metric`. 

    Arguments:
        - scores (list) : List of values to calculate metrics for
        - filename (str) : Filename to use to save the figure
        - metric (st) : The metric that corresponds to the list of scores
    """
    with open(filename, 'a') as f:
        f.write(f"{metric}\n")
        f.write("-"*(len(metric)+2))
        f.write("\n")
        f.write(f"Mean: {np.mean(scores)}\n")
        f.write(f"Std: {np.std(scores)}\n")
        f.write(f"Min: {np.min(scores)}\n")
        f.write(f"Max: {np.max(scores)}\n\n")


def main(args):
    # Set path for checkpoints
    root = Path(args.save)
    root.mkdir(parents=True, exist_ok=True)

    # Unless otherwise specified, model runs on CUDA if available
    if args.device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Initialize dataloader
    dataset = RCTDataset(args.images, args.targets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    # Load model's weights if checkpoint is given
    if args.checkpoint:
        model.load_state_dict(torch.load(
            args.checkpoint, map_location=torch.device(device)))

    random_seeds = random.sample(range(0, 1000), args.nseeds)
    with torch.no_grad():
        total_psnr = []
        total_ssim = []
        for i, seed in enumerate(random_seeds):
            torch.manual_seed(seed=seed)

            psnr_scores = []
            ssim_scores = []
            for x, target in dataloader:

                y = torch.clamp(model(x), max=255.0)
                target = torch.clamp(target, max=255.0)

                for img_true, img_test in zip(target, y):
                    img_true = img_true.numpy()
                    img_test = img_test.numpy()

                    psnr_scores.append(
                        psnr(img_true, img_test, data_range=255))
                    ssim_scores.append(ssim(
                        img_true, img_test,
                        channel_axis=0,
                        data_range=255))

            total_psnr.append(np.mean(psnr_scores))
            total_ssim.append(np.mean(ssim_scores))
            print(f"[{i+1}/{args.nseeds}]", end="\r")

    density(total_psnr, root / Path("psnr"), "PSNR (dB)")
    density(total_ssim, root / Path("ssim"), "SSIM")
    log(total_psnr, root / Path("log.txt"), "PSNR")
    log(total_ssim, root / Path("log.txt"), "SSIM")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--images', required=True,
                        help='Path to the directory of images to be enhanced')

    parser.add_argument(
        '--targets', required=True,
        help='Path to the directory of groundtruth enhanced images')

    parser.add_argument(
        '--save', required=True,
        help='Path to the save plots and log file with metrics')

    parser.add_argument(
        '--config', default=None, type=str,
        help="Path to configurations file for the RCTNet model")

    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to previous checkpoint')

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Number of samples per minibatch')

    parser.add_argument(
        '--nseeds', default=100, type=int,
        help='Number of seeds to run evaluation for, in range [0 .. 1000]')

    parser.add_argument('--device', default=None, choices=["cpu", "cuda"],
                        type=str, help='Device to use for training')

    args = parser.parse_args()

    main(args)
