"""Data handling for DFF model.
"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
from gitstar.models.datanorm import feature_transform, target_transform


class GitStarDataset(Dataset):
    """
    Args:
        csv_file : str or Path

        sample_frac :  float or int
            [0,1]

        transform : boolean, default True
            Apply scale transformations according to datanorm module.

        shuffle : boolean, default False
            Randomize dataframe.

    Attributes:
        df : pd.DataFrame
            Entire dataset

        target_inv_fn : sklearn.preprocessing.scaler()
            Scaler object that holds target fit parameters. Access inv.
            function via target_inv_fn.inverse_transform(X), X : nd.array

        features, target : pd.DataFrame
    """

    def __init__(self, csv_path, sample_frac=1, transform=True, shuffle=False):
        # Load data. Take random subset according to sample_frac.
        if shuffle:
            self.df = (
                pd.read_csv(csv_path).astype("float64").sample(frac=sample_frac)
            )
        else:
            self.df = pd.read_csv(csv_path).astype("float64")
            self.df = self.df.head(int(sample_frac * len(self.df)))

        # Intialize transform related attributes
        self.target_inv_fn = None
        self.transform = transform

        # Apply data scaling
        if self.transform:
            feature_transform(self.df)
            _, self.target_inv_fn = target_transform(self.df)

        # Separate features and target. Drop method returns deep copy of df
        self.features = self.df.drop("stargazers", axis=1)
        self.target = self.df["stargazers"].to_frame()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # pd.df -> np.ndarray with dtype float64
        x_sample = self.features.iloc[idx].values
        y_sample = self.target.iloc[idx].values
        # np.ndarry -> torch.tensor.float() to match weights datatype
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.tensor(y_sample).float()

        return x_sample, y_sample


def rand_split_rel(dataset, frac, **kwargs):
    """Splits dataset as fraction of total. Based on torch random_split.

        Args:
            dataset : torch.utils.data.Dataset

            frac : float or int
                [0,1]
        Returns:
            ds_frac, ds_remainder : tuple of torch.utils.Dataset
    """
    size_1 = int(frac * len(dataset))
    size_2 = len(dataset) - size_1
    return random_split(dataset, [size_1, size_2], **kwargs)


def get_data(train_ds, valid_ds, bs):
    """Create dataloaders based on train/validation datasets and batch size.

        Args:
            train_ds, valid_ds : torch.utils.data.Dataset
            bs : int
        Returns:
            train_dl, valid_dl : torch.utils.data.DataLoader
    """
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2 * bs)
    return train_dl, valid_dl


class WrappedDataLoader:
    """Applied preprocessing function to torch.utils.Data.Dataloader objects

    Args:
        dl : torch.utilts.Data.Dataloader
        func : function()
    """
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def module_test():
    """Test class implementations"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=64)
    for xb, yb in train_dl:
        print(xb, yb)


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
