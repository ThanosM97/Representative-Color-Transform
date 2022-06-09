"""This module implements the training of our model."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import RCTDataset
from RCTNet.loss import Loss
from RCTNet.model import RCTNet


def save_checkpoint(model: RCTNet, root: Path, epoch: int) -> None:
    """Save checkpoint for the model.

    This method is called at the end of each training epoch in order to
    save a checkpoint for the model's weights.

    Args:
        - model (RCTNet): The trained RCTNet model
        - root (Path): Root path for the checkpoints
        - epoch (int): The current epoch of training
    """
    save_path = root / Path(f"epoch-{epoch}/")
    save_path.mkdir()

    # Save model weights
    torch.save(
        model.state_dict(),
        save_path / "checkpoint.pt"
    )


def main(args):
    # Set path for checkpoints
    root = Path(f"checkpoints/checkpoint-"
                f"{datetime.today().strftime('%Y-%m-%d-%H-%M')}/")
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

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10)

    # Initialize loss
    loss = Loss(device=device)

    # Training loop
    print('Training...')
    losses = np.empty([])
    for epoch in range(args.epochs):
        l = 0.0

        # Loop through batches
        for i, (x, target) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward pass
            enhanced = model(x)

            # Calculate loss
            batch_loss = loss.estimate_loss(enhanced, target)

            # Backward pass
            batch_loss.backward()

            # Update weights
            optimizer.step()

            # Update scheduler
            scheduler.step()

            l += batch_loss.detach().cpu().item()
            print(
                (f"Epoch: {epoch+1}/{args.epochs}, "
                 f"Iter: {i+1}/{len(dataloader)}, Loss: {l/(i+1)}"),
                end='\r')

        # Save checkpoint
        np.append(losses, l/len(dataloader))
        if ((epoch+1) % args.checkpoint_interval == 0):
            save_checkpoint(model=model, root=root, epoch=epoch)

    np.save(root / "losses.npy", losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--images', required=True,
                        help='Path to the directory of images to be enhanced')

    parser.add_argument('--targets', required=True,
                        help='Path to the directory of enhanced images')

    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs')

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Number of samples per minibatch')

    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Initial Learning rate of Adam optimizer')

    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay of Adam optimizer')

    parser.add_argument('--config', default=None, type=str,
                        help="Path to configurations file for RCTNet model")

    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to previous checkpoint')

    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Interval for saving checkpoints')

    parser.add_argument('--device', default=None, choices=["cpu", "cuda"],
                        type=str, help='Device to use for training')

    args = parser.parse_args()

    main(args)
