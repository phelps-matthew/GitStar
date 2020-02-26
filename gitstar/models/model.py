"""Deep feedforward model

    ToDo:
        Normalize inputs/output
        BS, lr, model dims. Separate affects
"""

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from gitstar.models.dataload import GitStarDataset, rand_split_rel, get_data
import numpy as np


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
        # We shall try ReLU on hidden layers
        for layer in self.layers:
            x = self.a_fn(layer(x))
        # Relu activation on output layer (since target strictly > 0)
        return F.relu(self.out(x))


def loss_batch(model, loss_func, xb, yb, opt=None):
    """Computes batch loss for training (with opt) and validation"""
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    # loss returns torch.tensor
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        # Weighted sum of mean loss per batch. Batches may not be identical.
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


def main():
    """Test class implementations"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    # Model params
    bs = 8
    lr = 0.001
    epochs = 2
    h_layers = [21]

    # Load data
    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE, sample_frac=1)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)

    # Intialize model, optimization method, and loss function
    model = DFF(21, h_layers, 1)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = F.mse_loss

    # Train model
    for epoch in range(epochs):
        model.train()
        i = 0
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)
            # Print initial train_dl loss
            if i == 0:
                print(loss)

            loss.backward()
            opt.step()
            opt.zero_grad()
            i += 1

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        # Print loss from valid_dl
        print("Epoch: {}  Validation Loss: {}".format(epoch, valid_loss / len(valid_dl)))

    # fit(epochs, model, loss_func, opt, train_dl, valid_dl)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
