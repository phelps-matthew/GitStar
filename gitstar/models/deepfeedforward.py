"""
Constructs deep feedforward NN model with backprop
* Presents NN class with adjustable hidden layer size/depth and act. functions
* Provides training and validation functions
* Validation performed on scaled and unscaled data
* Computes and stores loss function and R^2 model stats
* Inlcudes many helper functions for logging, plotting, printing, and storing
  loss and validation data
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

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "hyperparams"
LOG_PATH = BASE_DIR / "logs"


class DFF(nn.Module):
    """
    Construct deep feedfoward net with arbitrary hidden layers and act. fn.

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
    layers : nn.ModuleList
        does not include ouput layer
    out : nn.Module
        output layer
    """

    def __init__(self, D_in, D_hid, D_out, a_fn=F.relu):

        # nn.Module must be initialized, many class attributes
        super().__init__()
        self.a_fn = a_fn

        # Check for list or int; cannot be None or other iterable
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


def fit(
    epochs,
    model,
    loss_func,
    opt,
    train_dl,
    valid_dl,
    t_scaler=None,
    path=LOG_PATH,
    hyper_str="test_model",
):
    """
    Iterates training and validation over mulitple epochs

    Parameters
    ----------
    epochs : int
    model : DFF
    loss_func : torch.nn.functional
    opt : torch.optim
    train_dl, valid_dl : torch.utils.data.DataLoader
    t_scaler : sklearn.preprocessing.scaler(), default None
        Target inverse scale transformer
    path : str or Path
        Directory to store loss data
    hyper_str : str
        Base filename for logging/storage (do not include filetype)

    Returns
    -------
    train_losses : ndarray
    """
    fit_list = []
    train_losses = []
    for epoch in range(epochs):
        # Train and validate
        val_list, train_loss = fit_epoch(
            model, loss_func, opt, train_dl, valid_dl, t_scaler
        )
        # Store validation data
        fit_list.append(val_list)
        # Store training losses
        train_losses.append(train_loss)
        # Print table of validation status
        print_stats(epoch, *val_list)

    # Flatten the list of lists
    train_losses = np.array(train_losses).flatten()
    # Save files
    store_losses(path, hyper_str, *zip(*fit_list), train_losses)
    return train_losses


def fit_epoch(model, loss_func, opt, train_dl, valid_dl, t_scaler=None):
    """
    Train and validate over one epoch

    Parameters
    ----------
    model : DFF
    loss_func : torch.nn.functional
    opt : torch.optim
    epoch : int
    train_dl, valid_dl : torch.utils.data.DataLoader
    path : str or Path
        Directory to store loss data
    t_scaler : sklearn.preprocessing.scaler(), default None
        Target inverse scale transformer

    Returns
    _______
    list of float
        [val_loss, val_rs, val_inv_loss, val_inv_rs]
    list of float
        train losses
    """
    # -- Training --
    model.train()  # good habit; relevant for Dropout, BatchNorm layers
    train_losses = []
    for xb, yb in train_dl:
        train_loss, _ = loss_batch(model, loss_func, xb, yb, opt)
        train_losses.append(train_loss)

    # -- Validation --
    model.eval()  # good habit; relevant for Dropout, BatchNorm layers

    # Do not track gradient on validation operations
    with torch.no_grad():
        # Create lists of validation loss and batch sizes
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        )
        # Create lists of unscaled validation loss and batch sizes
        inv_losses, inv_nums = zip(
            *[
                inv_loss_batch(model, loss_func, xb, yb, t_scaler)
                for xb, yb in valid_dl
            ]
        )
        # Compute valid. loss and R^2
        val_loss, val_rs = compute_stats(losses, nums, valid_dl)
        # Compute unscaled valid. loss and R^2
        val_inv_loss, val_inv_rs = compute_inv_stats(
            inv_losses, inv_nums, valid_dl, t_scaler
        )
    return [val_loss, val_rs, val_inv_loss, val_inv_rs], train_losses


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    Computes batch loss performs backprop on training data

    Also computes batch loss on validation data

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
        batch size

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
    # extract loss float from torch.tensor
    return loss.item(), len(xb)


def inv_loss_batch(model, loss_func, xb, yb, t_scaler):
    """
    Computes unscaled batch loss from validation set

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
        batch size

    Notes
    -----
    If inverse fn not provided, passes dummy returns of zero loss and unity
    size
    Some sklearn scaler functions may diverge on input range; NaNs are omitted
    from batch loss
    """
    # If an inverse function is provided, use it
    if t_scaler is not None:
        # numpy -> inverse transformation -> torch.tensor
        inv_y_pred = t_scaler.inverse_transform(model(xb).numpy())
        inv_y = t_scaler.inverse_transform(yb.numpy())

        # Delete possible NaNs from divergent scaling transforms
        if np.any(np.isnan(inv_y_pred)):
            logging.warning("NaNs found in inverting target prediction")
            # Find all NaNs
            bool_array = np.isnan(inv_y_pred)
            # Pass only True non-NaNs
            inv_y_pred = inv_y_pred[~bool_array]
            inv_y = inv_y[~bool_array]

        # Convert back to torch.tensor
        model_yb_unorm = torch.from_numpy(inv_y_pred)
        yb_unorm = torch.from_numpy(inv_y)
        unorm_loss = loss_func(model_yb_unorm, yb_unorm).item()
        size = len(yb_unorm)

    # If no inverse function, pass dummy values; log warning
    else:
        logging.warning(
            "No target inverse function provided: unscaled loss = scaled loss"
        )
        unorm_loss = 0
        size = 1
    return unorm_loss, size


def compute_stats(losses, batch_sizes, valid_dl):
    """
    Compute batch scaled validation loss and R^2 per epoch

    Parameters
    ----------
    losses : list of float
        MSE losses
    batch_sizes : list of int
        Not all batches are of identical size
    valid_dl : WrappedDataLoader or torch.utils.DataLoader
        Used to compute variance

    Returns
    -------
    val_loss, val_rs : float
       Weighted average of loss over batches; R^2
    """
    # Weighted sum of mean loss per batch; batches may not be identical
    val_loss = np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes)
    # Compute variance
    val_var = torch.cat([yb for xb, yb in valid_dl]).var().item()
    # R^2 = valid_loss/var(valid_dl)
    val_rs = 1 - (val_loss / val_var)
    return val_loss, val_rs


def compute_inv_stats(losses, batch_sizes, valid_dl, t_scaler):
    """
    Compute batch scaled validation loss and R^2 for unscaled validation data
    per epoch

    Parameters
    ----------
    losses : list of float
        MSE losses
    batch_sizes : list of int
        Not all batches are of identical size
    valid_dl : WrappedDataLoader or torch.utils.DataLoader
        Used to compute variance
    t_scaler : sklearn.preprocessing.scaler
        Target inverse scale transformer

    Returns
    -------
    val_loss, val_rs : float
       Weighted average of loss over batches; R^2
    """
    # Weighted sum of mean unscaled loss per batch; batches may not be identical
    val_loss = np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes)

    # If an inverse transform is provided, use it
    if t_scaler is not None:
        # Construct inverse function torch -> np -> torch
        def to_inv(x):
            return torch.from_numpy(t_scaler.inverse_transform(x.numpy()))

        # Compute variance
        val_var = torch.cat([to_inv(yb) for xb, yb in valid_dl]).var().item()
        # R^2 = valid_loss/var(valid_dl)
        val_rs = 1 - (val_loss / val_var)

    # If no inverse transform, use compute_stats
    else:
        _, val_rs = compute_stats(losses, batch_sizes, valid_dl)
    return val_loss, val_rs


def store_losses(
    path,
    filename,
    valid_loss,
    valid_rs,
    valid_inv_loss,
    valid_inv_rs,
    train_loss,
):
    """
    Store csv's of training and validation losses and R^2 values

    Parameters
    ----------
    path : str or Path
        Output directory destination
    filename : str
        Output filename (do not include filetype)
    train_loss : list of float
    valid_loss, valid_rs, valid_inv_loss, valid_inv_rs : list of float

    Returns
    -------
    None
    """
    np.savetxt(path / ("train_bloss_" + filename + ".csv"), train_loss)
    np.savetxt(path / ("valid_loss_" + filename + ".csv"), valid_loss)
    np.savetxt(path / ("valid_rs_" + filename + ".csv"), valid_rs)
    np.savetxt(path / ("valid_inv_loss_" + filename + ".csv"), valid_inv_loss)
    np.savetxt(path / ("valid_inv_rs_" + filename + ".csv"), valid_inv_rs)


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


def print_stats(epoch, *args):
    """Print table of validation loss, R^2, and epoch (scaled and unscaled)"""
    table_str = (
        "[{}]\n Epoch: {:02d}  MSE: {:8.7f}  R^2: {: 8.7f} "
        + "uMSE: {:2.7f}  uR^2: {: 2.7f}"
    )
    print(table_str.format(arrow.now(), epoch, *args))


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
            "torch.cuda.current_device: {}".format(torch.cuda.current_device()),
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
