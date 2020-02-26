"""Generate Multi-Layer Perceptron"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from gitstar.models.dataload import GitStarDataset, rand_split_rel, get_data
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Construct MLP with len(h_sizes) hidden layers of type nn.Linear.
        Args:
            h_sizes (iterable): list of hidden layer dimensions, sequential
            out_size (int): output layer dimension
    """
    def __init__(self, h_sizes, out_size):
        super().__init__()

        # Modules w/i ModuleList are properly registered
        # and visible by all module methods
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Hidden layers
        self.hidden = []
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        output= F.softmax(self.out(x), dim=1)

        return output

def main():
    """Test class implementations"""

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "gs_table_v2_sample.csv"

    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=1)

    for i, elem in enumerate(valid_dl):
        print(i,elem)
        if i == 3:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
