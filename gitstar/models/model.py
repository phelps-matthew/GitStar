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
    print("torch.cuda.device(0): {}".format(torch.cuda.device(0)))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print(
        "torch.cuda.get_device_name(0): {}".format(
            torch.cuda.get_device_name(0)
        )
    )
    print("torch.cuda_is_available: {}".format(torch.cuda.is_available()))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))


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


def plot_loss(loss_array, ylabel="MSE Loss", ylim=(0, 2)):
    """Simple plot of 1d array.

        Args:
            loss_array (list, nd.array)
            ylabel (str)
            ylim (tuple)
    """
    fig, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_ylim(ylim)
    ax.set(xlabel="batch number", ylabel=ylabel)
    ax.grid()
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


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
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
    batch_loss = []
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
        print(
            "Epoch: {}  Loss: {}  Error: {}".format(epoch, val_loss, val_error)
        )
    return batch_loss


def main():
    """Test class implementations"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    LOG_PATH = BASE_DIR / "logs"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    # Enable GPU support
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    def preprocess(x, y):
        return x.to(dev), y.to(dev)

    # Initialize logger
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
    set_logger(str(LOG_PATH / "model.log"))

    # Load data
    batch_size = 64
    dataset = GitStarDataset(DATA_PATH / FILE)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=batch_size)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)

    # Hyperparameters
    h_layers = [21]
    lr = 10**(-5)
    epochs = 20

    # Intialize model, optimization method, and loss function
    model = DFF(21, h_layers, 1, a_fn=F.rrelu)
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss

    train_loss = fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    loss_df = pd.DataFrame(train_loss, columns=["lr={}".format(lr)])
    loss_df.to_csv(LOG_PATH / "Adam_lr_{}_DFF_21_ReLU.csv".format(lr))


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
