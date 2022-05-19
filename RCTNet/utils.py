"""This module implements utility functions used for the implementation
of Kim et al. (2021) https://ieeexplore.ieee.org/document/9710400"""
import torch
import torch.nn.functional as F


def padding(x: torch.Tensor, grid_size: int, mode: str = "reflect") -> tuple[
        torch.Tensor, int, int, list[int]]:
    """Pad input `x` for it to be divisible by `grid_size`.

    This function implements the paddding of input `x`, so it can be divisible
    by the `grid_size`. If the required padding is not even, then the paddings
    on the right and bottom are larger (by 1).

    Args:
        - x (torch.Tensor) : input Tensor to be padded
        - grid_size (int) : size of meshgrid
        - mode (string) : mode used to pad the input `x`. 
                          (Options: 'constant', 'reflect', 'replicate' 
                           or 'circular'.  Default: 'reflect')

    Returns:
        A tuple that consists of:
        - x_padded (torch.Tensor)
        - h_new (int) : the height of x_padded
        - w_new (int) : the width of x_padded
        - paddings (list) : a list of the paddings [left, right, top, bottom]

    """
    h, w = x.shape[2], x.shape[3]
    w_mod = w % grid_size
    if w_mod != 0:
        w_pad_len = grid_size - w_mod
        w_new = w + w_pad_len
        w_pad = [w_pad_len // 2, w_pad_len // 2 + w_pad_len % 2]

    h_mod = h % grid_size
    if h_mod != 0:
        h_pad_len = grid_size - h_mod
        h_new = h + h_pad_len
        h_pad = [h_pad_len // 2, h_pad_len // 2 + h_pad_len % 2]

    paddings = h_pad + w_pad

    return F.pad(x, w_pad + h_pad, mode=mode), h_new, w_new, paddings


def remove_padding(x: torch.Tensor, paddings: list[int]) -> torch.Tensor:
    """Remove `paddings` from input `x`.

    This function removes the `padddings` of input `x`, to retun it to its
    original size. 

    Args:
        - x (torch.Tensor) : input Tensor to be padded
        - paddings (list) : a list of the paddings [left, right, top, bottom]

    Returns:
        The un-padded input `x`.
    """
    h, w = x.shape[2], x.shape[3]
    return x[:, :, paddings[0]:h-paddings[1],
             paddings[2]:w-paddings[3]]
