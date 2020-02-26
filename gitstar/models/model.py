"""Deep feedforward NN model"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from gitstar.models.dataload import GitStarDataset, rand_split_rel, get_data
from torch import nn
import torch.nn.functional as F


class DFF(nn.Module):
    """Construct basic FF NN with len(h_sizes) hidden layers.
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
        # We shall try ReLU
        for layer in self.layers:
            x = self.a_fn(layer(x))
        # linear activation on output layer
        return self.out(x)


def main():
    """Test class implementations"""

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "gs_table_v2_sample.csv"

    model = DFF(21, [24, 48], 1)

    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=2)

    for xb, yb in train_dl:
        out = model(xb)
        print(out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()