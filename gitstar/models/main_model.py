"""
Implements deep feedforward model for consuming the GitStar dataset. Loads
training and validation datasets, optimizes hyperparameters, and generates
loss plots/data. Cuda GPU ready.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
import pandas as pd
from gitstar.models.dataload import (
    GitStarDataset,
    WrappedDataLoader,
    split_df,
    get_data,
)
import gitstar.models.deepfeedforward as dff
import numpy as np

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "hyperparams"
LOG_PATH = BASE_DIR / "logs"
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"

# Enable GPU support
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dff.print_gpu()
print(dev)


def preprocess(x, y):
    """
    Cast tensors into GPU/CPU device type.

    Parameters
    ----------
    x,y : torch.tensor.

    Returns
    -------
    x.to(dev), y.to(dev) : torch.tensor
    """
    return x.to(dev), y.to(dev)


def main():
    """Train, validate, optimize model."""

    # Load data. Apply scaler transformations to training data. Get DataLoader.
    batch_size = 64
    df = pd.read_csv(DATA_PATH / FILE).astype("float64")
    df = df.loc[df["stargazers"] >= 100]
    train_df, valid_df = split_df(df, sample_frac=1)
    train_ds = GitStarDataset(train_df)
    valid_ds = GitStarDataset(
        valid_df,
        f_scale=train_ds.feature_scalers,
        t_scale=train_ds.target_scaler,
    )
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=batch_size)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)

    # Gather inverse target scaler
    target_inv_scaler = train_ds.target_scaler['stargazers']

    # Hyperparameters
    lr = 10 ** (-5)
    h_layers = [64, 32]
    epochs = 100
    a_fn = F.rrelu

    # Intialize model (w/ GPU support), optimization method, and loss function
    model = dff.DFF(D_in=21, D_hid=h_layers, D_out=1, a_fn=a_fn)
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss

    # Generate descriptive parameter string (for pngs and csvs)
    model_str = dff.hyper_str(
        h_layers, lr, opt, a_fn, batch_size, epochs, prefix="stars_ge_100_"
    )
    print(model_str)

    # Train, validate, save and plot loss
    train_loss, _, _ = dff.fit(
        epochs,
        model,
        loss_func,
        opt,
        train_dl,
        valid_dl,
        LOG_PATH,
        model_str,
        t_scaler=target_inv_scaler,
    )
    # dff.plot_loss(
    #     train_loss, path=IMG_PATH / (model_str + ".png"), title=model_str
    # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
