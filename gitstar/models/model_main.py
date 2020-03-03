"""Implements deep feedforward model for consuming the GitStar dataset. Loads
    training and validation datasets, optimizes hyperparameters, and generates
    loss plots/data. Cuda GPU ready.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
from gitstar.models.dataload import (
    GitStarDataset,
    WrappedDataLoader,
    rand_split_rel,
    get_data,
)
import gitstar.models.deepfeedfoward as dff

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "hyperparams"
LOG_PATH = BASE_DIR / "logs"
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"

# Enable GPU support
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dff.print_gpu()
print(dev)


def preprocess(x, y):
    """Cast tensors into GPU/CPU device type.

    Args:
        x,y : torch.tensor

    Returns:
        x.to(dev), y.to(dev) : torch.tensor
    """
    return x.to(dev), y.to(dev)


def main():
    """Train, validate, optimize model."""

    # Load data
    batch_size = 64
    dataset = GitStarDataset(DATA_PATH / FILE, sample_frac=0.4, shuffle=True)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=batch_size)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)

    # Hyperparameters
    lr = 10 ** (-5)
    h_layers = [21]
    epochs = 5
    a_fn = F.rrelu

    # Intialize model (w/ GPU support), optimization method, and loss function
    model = dff.DFF(D_in=21, D_hid=h_layers, D_out=1, a_fn=a_fn)
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_func = F.mse_loss

    # Generate descriptive parameter string (for pngs and csvs)
    model_str = dff.hyper_str(h_layers, lr, opt, a_fn, batch_size, epochs)
    print(model_str)

    # Train, validate, save loss
    train_loss, _, _ = dff.fit(
        epochs, model, loss_func, opt, train_dl, valid_dl, LOG_PATH, model_str
    )
    dff.plot_loss(
        train_loss, path=IMG_PATH / (model_str + ".png"), title=model_str
    )

    # rates = [
    #     10 ** (-6),
    #     10 ** (-5),
    #     10 ** (-4),
    # ]
    # h_layer_ls = [[21], [32]]
    # optims = [
    #     optim.SGD(model.parameters(), lr=lr, momentum=0.9),
    #     optim.Adam(model.parameters(), lr=lr),
    #     optim.SparseAdam(model.parameters(), lr=lr),
    # ]

    # for h_lay in h_layer_ls:
    #     for lr in rates:
    #         for opts in optims:
    #             # Train DFF. Validate. Print validation loss and error.
    #             opt = opts
    #             model = dff.DFF(21, h_lay, 1, a_fn=a_fn)
    #             model.to(dev)
    #             model_str = dff.hyper_str(
    #                 h_lay, lr, opt, a_fn, batch_size, epochs
    #             )
    #             train_loss, _, _ = dff.fit(
    #                 epochs,
    #                 model,
    #                 loss_func,
    #                 opt,
    #                 train_dl,
    #                 valid_dl,
    #                 LOG_PATH,
    #                 model_str,
    #             )
    #             dff.plot_loss(
    #                 train_loss,
    #                 path=IMG_PATH / (model_str + ".png"),
    #                 title=model_str,
    #             )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
