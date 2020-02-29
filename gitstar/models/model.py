"""Deep feedforward model

    ToDo:
        Hyperparameter tuning
"""

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import logging
import matplotlib.pyplot as plt
from gitstar.models.dataload import GitStarDataset, rand_split_rel, get_data


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


def plot_loss(loss_array, ylabel="MSE Loss", ylim=(0,2)):
    """Simple plot of 1d array.

        Args:
            loss_array (list, nd.array)
            ylabel (str)
            ylim (tuple)
    """
    fig, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_ylim(ylim)
    ax.set(xlabel="batch", ylabel=ylabel)
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
        logging.info(loss)
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

    # Initialize logger
    set_logger(LOG_PATH / "model.log")

    # Model params
    bs = 5
    lr = 0.001
    epochs = 1
    h_layers = [16, 16]

    # Load data
    dataset = GitStarDataset(
        DATA_PATH / SAMPLE_FILE, sample_frac=1, transform=True, shuffle=False
    )
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)

    # Intialize model, optimization method, and loss function
    model = DFF(21, h_layers, 1, a_fn=F.rrelu)
    # opt = optim.SGD(model.parameters(), lr=lr, momentum=0)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_func = F.mse_loss

    train_loss = fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    plot_loss(train_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
