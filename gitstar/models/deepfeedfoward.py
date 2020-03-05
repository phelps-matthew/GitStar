"""Deep feedforward NN model, with variable activation functions and layer
    size/depth. Inlcudes helper functions for logging,
    plotting, and storing loss data.

"""

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import numpy as np
import logging
import re
import matplotlib.pyplot as plt
import arrow
# from gitstar.models.dataload import (
#     GitStarDataset,
#     WrappedDataLoader,
#     split_csv,
#     get_data,
# )


class DFF(nn.Module):
    """Construct basic deep FF net with len(D_hid) hidden layers.

    Args:
        D_in, D_out : int
            Input, output dimension.
        D_hid : list or int
            List of hidden layer dimensions, sequential.
        a_fn, optional : torch.nn.functional, default F.relu
            Activation function on hidden layers.
    """

    def __init__(self, D_in, D_hid, D_out, a_fn=F.relu):

        # Module must be initialized, many hidden attributes
        super().__init__()
        self.a_fn = a_fn

        # Must be list, cannot be None or other iterable
        assert isinstance(D_hid, (int, list))

        # Compose list of DFF dimensions
        if isinstance(D_hid, int):
            dim_list = [D_in] + [D_hid] + [D_out]
        else:
            dim_list = [D_in] + D_hid + [D_out]

        self.layers = []

        # Construct interlaced dims for self.layers ModuleList
        for dim in range(len(dim_list) - 1):
            self.layers.append(nn.Linear(dim_list[dim], dim_list[dim + 1]))

        # Separate output layer for different activation
        self.out = self.layers.pop()

        # Modules w/i ModuleList are properly inherited
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """Execute feedforward with input x
        Args:
            x : torch.tensor
                Input from DataLoader
        """
        # ReLU on hidden layers
        for layer in self.layers:
            x = self.a_fn(layer(x))
        # No activation on output layer (linear)
        return self.out(x)


def print_gpu():
    """Print GPU torch cuda status"""
    try:
        print("torch.cuda.device(0): {}".format(torch.cuda.device(0)))
        print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
        print(
            "torch.cuda.get_device_name(0): {}".format(
                torch.cuda.get_device_name(0)
            )
        )
        print("torch.cuda_is_available: {}".format(torch.cuda.is_available()))
        print(
            "torch.cuda.current_device(): {}".format(
                torch.cuda.current_device()
            )
        )
    except:
        print("Some torch.cuda functionality unavailable")


def set_logger(filepath):
    """Intialize root logger here.
    Args:
        filepath : str or Path
    """
    logging.basicConfig(
        filename=str(filepath),
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def hyper_str(h_layers, lr, opt, a_fn, bs, epochs, prefix=None, suffix=None):
    """Generate str for DFF model for path names

    Args:
        h_layers : int or list of int
        lr : float
        opt : torch.optim
        a_fn : F.functional
        bs, epochs : int
        prefix, suffix : str, default None
    Returns:
        full_str : str
    """
    prefix = "" if prefix is None else prefix
    suffix = "" if suffix is None else suffix

    # e.g. 16x16 hidden dims
    h_layers_str = "x".join(list(map(str, h_layers)))

    # regex search patterns based on fn str
    a_fn_sub = re.search("^<\w+\s(\w+)\w.*$", str(a_fn))
    a_fn_str = a_fn_sub.group(1)
    opt_sub = re.search("^(\w+)\s.*", str(opt))
    opt_str = opt_sub.group(1)

    param_str = "{}_lr_{}_{}_{}_bs_{}_epochs_{}".format(
        h_layers_str, lr, opt_str, a_fn_str, bs, epochs
    )
    full_str = prefix + param_str + suffix
    return full_str


def plot_loss(
    loss_array, path=None, title="Loss", ylabel="MSE Loss", ylim=(0, 2)
):
    """Simple plot of 1d array.

    Args:
        loss_array : list or nd.array
        path : str or Path. default None
            Image path.
        ylabel : str
        ylim : tuple of int or float
    """
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_ylim(ylim)
    ax.set(xlabel="batch number", ylabel=ylabel, title=title)
    ax.grid()
    if path:
        fig.savefig(
            str(path), transparent=False, dpi=300, bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def loss_batch(model, loss_func, xb, yb, opt=None):
    """Computes batch loss for training (with opt) and validation.
        
    Args:
        model : DFF
        loss_func : torch.nn.functional
        xb, yb : torch.tensor
        opt : torch.optim

    Returns:
        loss.item() : float
        len(xb) : int
    """
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Capture loss
        # logging.info(loss)
    # loss returns torch.tensor
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, path, hyper_str):
    """Iterates feedforward and validation loops.

    Args:
        epochs : int
        model : DFF
        loss_func : torch.nn.functional
        opt : torch.optim
        train_dl, valid_dl : torch.utils.data.DataLoader
    Returns:
        batch_loss : list of float))
            One dimensional.
    """
    batch_loss, valid_loss, valid_rs = [], [], []
    for epoch in range(epochs):
        model.train()  # Good habit. Relevant for Dropout, BatchNorm layers

        for xb, yb in train_dl:
            train_loss, _ = loss_batch(model, loss_func, xb, yb, opt)
            # Store for plotting
            batch_loss.append(train_loss)

        model.eval()  # Good habit. Relevant for Dropout, BatchNorm layers

        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )

        # Weighted sum of mean loss per batch. Batches may not be identical.
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        valid_loss.append(val_loss)

        # R^2 = valid_loss/var(valid_dl)
        val_var = torch.cat([yb for xb, yb in valid_dl]).var().item()
        val_rs = 1 - (val_loss / val_var)
        valid_rs.append(val_rs)

        print(
            "[{}] Epoch: {:02d}  MSE: {:8.7f}  R^2: {: 8.7f}".format(
                arrow.now(), epoch, val_loss, val_rs
            )
        )
    # Log losses
    np.savetxt(path / ("train_bloss_" + hyper_str + ".csv"), batch_loss)
    np.savetxt(path / ("valid_loss_" + hyper_str + ".csv"), valid_loss)
    np.savetxt(path / ("valid_rs_" + hyper_str + ".csv"), valid_rs)
    return batch_loss, valid_loss, valid_rs
