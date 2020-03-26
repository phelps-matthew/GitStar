"""
Implements deep feedforward model for consuming the GitStar dataset
* Construct dataset and dataloaders
* Execute training and validation sequence
* Log loss and validation statistics; plot training loss
* Cuda GPU ready
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
import gitstar.models.deepfeedforward as dff
from gitstar.models.dataload import form_dataloaders, form_datasets

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "hyperparams"
LOG_PATH = BASE_DIR / "logs"
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"

# Enable GPU support
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dff.print_gpu_status()


def main():
    """Train, validate, and optimize model"""
    # Set hyperparameters: batch size, learning rate, hidden layers, activ. fn
    bs = 64
    epochs = 1000
    lr = 10 ** (-5)
    h_layers = [32, 16]
    a_fn = F.relu

    # Construct Dataset from file; form DataLoaders
    train_ds, valid_ds = form_datasets(DATA_PATH / FILE)
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
    train_dl, valid_dl = form_dataloaders(train_ds, valid_ds, bs, preprocess)

    # Gather target inverse scaler fn
    t_inv_scaler = train_ds.target_scaler["stargazers"]

    # Intialize model (w/ GPU support), optimization method, and loss function
    model = dff.DFF(D_in=21, D_hid=h_layers, D_out=1, a_fn=a_fn)
    model.to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss
    fit_args = (model, loss_func, opt, train_dl, valid_dl, t_inv_scaler)

    # Generate descriptive filename string for csv logs
    prefix = "FINAL_"
    model_str = dff.hyper_str(h_layers, lr, opt, a_fn, bs, epochs, prefix)
    print(model_str)

    # Train, validate, and store loss
    dff.fit(epochs, *fit_args, LOG_PATH, model_str)


def preprocess(x, y):
    """
    Cast tensors into GPU/CPU device type.

    Parameters
    ----------
    x,y : torch.tensor.

    Returns
    -------
    torch.tensor
    """
    return x.to(DEV), y.to(DEV)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
