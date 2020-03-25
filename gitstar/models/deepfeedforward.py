"""
Deep feedforward NN model, with variable activation functions and layer
size/depth. Inlcudes helper functions for logging, plotting, and storing
loss data.
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


class DFF(nn.Module):
    """
    Construct deep feedfoward net with backprop and arbitrary hidden layers

    Parameters
    ----------
    D_in, D_out : int
        input, output dimension of net
    D_hid : list or int
        list of hidden layer dimensions, sequential
    a_fn : torch.nn.functional, default F.relu
        activation function on hidden layers

    Attributes
    ----------
    a_fn : torch.nn.functional
    layers : list of nn.Module
        does not include ouput layer
    out : nn.Module
        output layer
    """

    def __init__(self, D_in, D_hid, D_out, a_fn=F.relu):

        # nn.Module must be initialized, many class attributes
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

        # Construct interlaced dims for nn.ModuleList
        for dim in range(len(dim_list) - 1):
            self.layers.append(nn.Linear(dim_list[dim], dim_list[dim + 1]))

        # Separate output layer for different activation
        self.out = self.layers.pop()

        # ModuleList allows optimizer to properly handle Module weights
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """
        Execute feedforward from input

        Parameters
        ----------
        x : torch.tensor

        Returns
        -------
        out(x) : torch.tensor
        """
        # ReLU on hidden layers
        for layer in self.layers:
            x = self.a_fn(layer(x))
        # No activation on output layer (linear)
        return self.out(x)


def print_gpu_status():
    """Print GPU torch cuda status"""
    try:
        cuda_status = [
            "torch.cuda.device(0): {}".format(torch.cuda.device(0)),
            "torch.cuda.device_count(): {}".format(torch.cuda.device_count()),
            "torch.cuda.get_device_name(0): {}".format(
                torch.cuda.get_device_name(0)
            ),
            "torch.cuda_is_available: {}".format(torch.cuda.is_available()),
            "torch.cuda.current_device: {}".format(
                torch.cuda.current_device()
            ),
        ]
        print(cuda_status, sep="\n")
    except:
        print("Some torch.cuda functionality unavailable")


def set_logger(filepath):
    """
    Intialize basic root logger

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    None
    """
    logging.basicConfig(
        filename=str(filepath),
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def hyper_str(h_layers, lr, opt, a_fn, bs, epochs, prefix=None, suffix=None):
    """
    Generate DFF model string for path and plot naming

    Parameters
    ----------
    h_layers : int or list of int
        hidden layer dimensions, e.g. [16,16]
    lr : float
        learning rate.
    opt : torch.optim
        optimizer, e.g. torch.optim.SGD(..)
    a_fn : F.functional
        activation function applied to hidden layers
    bs, epochs : int
        batch size.
    prefix, suffix : str, optional
        consider appending with '_'

    Returns
    -------
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

    # form the model string
    param_str = "{}_lr_{}_{}_{}_bs_{}_epochs_{}".format(
        h_layers_str, lr, opt_str, a_fn_str, bs, epochs
    )

    # insert any prefixes or suffixes
    full_str = prefix + param_str + suffix
    return full_str


def plot_loss(
    loss_array, path=None, title="Loss", ylabel="MSE Loss", ylim=(0, 2)
):
    """
    Simple plot of 1d array, defaulted for MSE Loss

    Parameters
    ----------
    loss_array : list or ndarray
    path : str or Path, default None
        Path to store image.
    ylabel : str
    ylim : tuple of int or float

    Returns
    -------
    None
    """
    # Plot the array; add labels, limits, grid
    fig, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_ylim(ylim)
    ax.set(xlabel="batch number", ylabel=ylabel, title=title)
    ax.grid()
    # Store as png if path given, otherwise show plot UI
    if path:
        fig.savefig(
            str(path), transparent=False, dpi=300, bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    Computes batch loss for training and validation

    Parameters
    ----------
    model : nn.Module
        E.g. DFF
    loss_func : torch.nn.functional
    xb, yb : torch.tensor
    opt : torch.optim

    Returns
    -------
    float
        loss of batch
    int
        latch size

    Notes
    -----
    Does not update weights if optimizer not provided
    """
    # Compute batch loss
    loss = loss_func(model(xb), yb)
    
    # Peform backprop if training
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Log traning loss only
        logging.info(loss)
    # extract loss float from torch.tensor
    return loss.item(), len(xb)


def inv_loss_batch(model, loss_func, xb, yb, t_scaler):
    """
    Computes unscaled batch loss from validation set

    Parameters
    ----------
    model : DFF
    loss_func : torch.nn.functional
    xb, yb : torch.tensor
    t_scaler : tsklearn.preprocessing.scaler()

    Returns
    -------
    float
        Loss of batch.
    int
        Batch size.

    Notes
    -----
    If inverse fn not provided, passes dummy returns. See fit().
    """
    # Determine unscaled MSE
    if t_scaler is not None:
        # numpy -> inverse transformation -> torch.tensor
        inv_y_pred = t_scaler.inverse_transform(model(xb).numpy())
        inv_y = t_scaler.inverse_transform(yb.numpy())

        # Delete possible NaNs from sklearn scaler inverse
        if np.any(np.isnan(inv_y_pred)):
            bool_array = np.isnan(inv_y_pred)
            # Pass only True non-NaNs
            inv_y_pred = inv_y_pred[~bool_array]
            inv_y = inv_y[~bool_array]

        # Convert back to torch.tensor
        model_yb_unorm = torch.from_numpy(inv_y_pred)
        yb_unorm = torch.from_numpy(inv_y)
        unorm_loss = loss_func(model_yb_unorm, yb_unorm).item()
        size = len(yb_unorm)
    else:
        unorm_loss = 0
        size = 1
    return unorm_loss, size


def fit(
    epochs,
    model,
    loss_func,
    opt,
    train_dl,
    valid_dl,
    path,
    hyper_str,
    t_scaler=None,
):
    """
    Iterates feedforward and validation loops.

    Parameters
    ----------
    epochs : int
    model : DFF
    loss_func : torch.nn.functional
    opt : torch.optim
    train_dl, valid_dl : torch.utils.data.DataLoader
    t_scaler : tsklearn.preprocessing.scaler(), optional

    Returns
    _______
    batch_loss, valid_loss, valid_inv_loss : tuple of float
        One dimensional.
    """
    batch_loss, valid_loss, valid_inv_loss, valid_rs, valid_inv_rs = (
        [],
        [],
        [],
        [],
        [],
    )
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
            inv_losses, inv_nums = zip(
                *[
                    inv_loss_batch(model, loss_func, xb, yb, t_scaler)
                    for xb, yb in valid_dl
                ]
            )

        # Weighted sum of mean loss per batch. Batches may not be identical.
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        valid_loss.append(val_loss)

        # R^2 = valid_loss/var(valid_dl)
        val_var = torch.cat([yb for xb, yb in valid_dl]).var().item()
        val_rs = 1 - (val_loss / val_var)
        valid_rs.append(val_rs)

        # Weighted sum of mean unscaled loss per batch.
        # Batches may not be identical.
        val_inv_loss = np.sum(np.multiply(inv_losses, inv_nums)) / np.sum(
            inv_nums
        )
        valid_inv_loss.append(val_inv_loss)

        # Unscaled R^2
        if t_scaler is not None:
            to_inv = lambda x: torch.from_numpy(
                t_scaler.inverse_transform(x.numpy())
            )
            val_inv_var = (
                torch.cat([to_inv(yb) for xb, yb in valid_dl]).var().item()
            )
            val_inv_rs = 1 - (val_inv_loss / val_inv_var)
            valid_inv_rs.append(val_inv_rs)
        else:
            val_inv_rs = 0

        print(
            (
                "[{}]\n Epoch: {:02d}  MSE: {:8.7f}  R^2: {: 8.7f} "
                + "uMSE: {:2.7f}  uR^2: {: 2.7f}"
            ).format(
                arrow.now(), epoch, val_loss, val_rs, val_inv_loss, val_inv_rs
            )
        )
    # Log losses
    np.savetxt(path / ("train_bloss_" + hyper_str + ".csv"), batch_loss)
    np.savetxt(path / ("valid_loss_" + hyper_str + ".csv"), valid_loss)
    np.savetxt(path / ("valid_rs_" + hyper_str + ".csv"), valid_rs)
    np.savetxt(path / ("valid_inv_loss_" + hyper_str + ".csv"), valid_inv_loss)
    np.savetxt(path / ("valid_inv_rs_" + hyper_str + ".csv"), valid_inv_rs)
    return (batch_loss, valid_loss, valid_inv_loss)
