"""Deep feedforward model

    ToDo:
        Hyperparameter tuning
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
from gitstar.models.dataload import (
    GitStarDataset,
    WrappedDataLoader,
    rand_split_rel,
    get_data,
)


class DFF(nn.Module):
    """Construct basic deep FF net with len(D_hid) hidden layers.
        Args:
            D_in (int): Input dimension
            D_hid (list or int): list of hidden layer dimensions, sequential
            D_out (int): Output dimension
            [a_fn=F.relu] (torch.nn.functional) : Activation function on hidden
                layers
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
                x (torch.tensor): Input from DataLoader
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
            filepath (str, Path)
    """
    logging.basicConfig(
        filename=filepath,
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def plot_loss(loss_array, path=None, ylabel="MSE Loss", ylim=(0, 2)):
    """Simple plot of 1d array.

        Args:
            loss_array (list, nd.array)
            path=None (str, Path): Image path
            ylabel (str)
            ylim (tuple)
    """
    fig, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_ylim(ylim)
    ax.set(xlabel="batch number", ylabel=ylabel)
    ax.grid()
    if path:
        fig.savefig(
            path, transparent=False, dpi=300, bbox_inches="tight",
        )
    else:
        plt.show()


def error(y_pred, y):
    """Computes |y-y_pred|/y. Returns scalar representing average error
        
        Args:
            y_pred (torch.tensor 1d)
            y (torch.tensor 1d)
    """
    err = torch.div(torch.abs(y - y_pred), torch.abs(y))
    avg_err = torch.div(torch.sum(err), len(err)).item()
    # Returns float, int
    return avg_err, len(err)


def loss_batch(model, loss_func, xb, yb, opt=None):
    """Computes batch loss for training (with opt) and validation.
        
        Args:
            model (DFF)
            loss_func (torch.nn.functional)
            xb (torch.tensor)
            yb (torch.tensor)
            opt (torch.optim)
        Return:
            loss.item() (float)
            len(xb) (int)
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
            epocs (int)
            model (DFF)
            loss_func (torch.nn.functional)
            opt (torch.optim)
            train_dl (torch.utils.data.DataLoader)
            valid_dl (torch.utils.data.DataLoader)
        Return:
            batch_loss (list(float)): 1d
    """
    batch_loss, valid_loss, valid_error = [], [], []
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
            errors, _ = zip(*[error(model(xb), yb) for xb, yb in valid_dl])
        # Weighted sum of mean loss or error per batch.
        # Batches may not be identical.
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_error = np.sum(np.multiply(errors, nums)) / np.sum(nums)
        valid_loss.append(val_loss)
        valid_error.append(val_error)
        print(
            "Epoch: {}  Loss: {}  Error: {}".format(epoch, val_loss, val_error)
        )
    # Log losses
    np.savetxt(path / ("train_bloss_" + hyper_str + ".csv"), batch_loss)
    np.savetxt(path / ("valid_loss_" + hyper_str + ".csv"), valid_loss)
    np.savetxt(path / ("valid_error_" + hyper_str + ".csv"), valid_error)
    return batch_loss, valid_loss, valid_error


def hyper_str(h_layers, lr, opt, a_fn, bs, epochs):
    """Generate str for DFF model for path names"""
    h_layers_str = "x".join(list(map(str, h_layers)))
    a_fn_sub = re.search("^<\w+\s(\w+)\w.*$", str(a_fn))
    a_fn_str = a_fn_sub.group(1)
    opt_sub = re.search("^(\w+)\s.*", str(opt))
    opt_str = opt_sub.group(1)
    full_str = "{}_lr_{}_{}_{}_bs_{}_epochs_{}".format(
        h_layers_str, lr, opt_str, a_fn_str, bs, epochs
    )
    return full_str


def main():
    """Test class implementations"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    IMG_PATH = BASE_DIR / "hyperparams"
    LOG_PATH = BASE_DIR / "logs"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    # Enable GPU support
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print_gpu()

    def preprocess(x, y):
        return x.to(dev), y.to(dev)

    # Load data
    batch_size = 64
    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE, sample_frac=0.1)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=batch_size)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)

    # Hyperparameters
    lr = 10 ** (-5)
    h_layers = [15, 15]
    epochs = 1
    a_fn = F.rrelu

    # Intialize model, optimization method, and loss function
    model = DFF(21, h_layers, 1, a_fn=a_fn)
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss

    # Generate descriptive string
    model_str = hyper_str(h_layers, lr, opt, a_fn, batch_size, epochs)

    # Train, validate, save loss
    train_loss, _, _ = fit(
        epochs, model, loss_func, opt, train_dl, valid_dl, LOG_PATH, model_str
    )
    np.savetxt(LOG_PATH / "Adam_lr_{}_DFF_21_rrelu.csv".format(lr), train_loss)
    plot_loss(train_loss, path=IMG_PATH / (model_str + ".png"))

    # ########################################################################
    # for lr in rates:
    #    # Train DFF. Validate. Print validation loss and error.
    #    opt = optim.Adam(model.parameters(), lr=lr)
    #    train_loss = fit(epochs, model, loss_func, opt, train_dl, valid_dl)

    #    # Export training loss
    # csv_paths = [pth for pth in LOG_PATH.iterdir() if pth.suffix == ".csv"]
    # fig, *ax = plt.subplots(2,3, sharey=True, sharex=True)
    # ax_list = [ax_n for row in ax[0] for ax_n in row]

    # for path, ax_n in zip(csv_paths, ax_list):
    #     df = pd.read_csv(path, usecols=[1])
    #     ax_n.plot(df.values)
    #     ax_n.set(xlabel="batch number", ylabel="MSE", title=df.columns[0])
    #     ax_n.set_ylim(0,2)
    # plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
