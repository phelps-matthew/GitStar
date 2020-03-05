"""Data handling for DFF model.

    ToDo:
        Comment + Docstrings
        Possible wrappers for csv -> ds -> dataloader
"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from gitstar.models.datanorm import feature_transform, target_transform


class GitStarDataset(Dataset):
    """
    Implements torch.utils.data.Dataset with optional transformations upon
    the features and/or target. See datanorm for scale transformations.

    Parameters
    ----------
    df : pd.DataFrame
    transform : boolean, optional
        Apply scale transformations according to datanorm module.
    f_scale, t_scale : dict, optional
        Feature or target scalers, e.g.
        {"openissues": PowerTransformer(method="yeo-johnson")}

    Attributes
    ----------
    df : pd.DataFrame
    transform : boolean
        Apply scale transformations according to datanorm module.
    f_scale, t_scale : dict
        Feature or target scalers, e.g.
        {"openissues": PowerTransformer(method="yeo-johnson")}
    """

    def __init__(self, df, transform=True, f_scale=None, t_scale=None):
        # Intialize transform related attributes
        self.df = df
        self.transform = transform

        # Apply data scaling. If scalers are provided, use them.
        if self.transform:
            if f_scale is not None:
                self.feature_scalers = feature_transform(self.df, f_scale)
                self.target_scaler = target_transform(self.df, t_scale)
            else:
                self.feature_scalers = feature_transform(self.df)
                self.target_scaler = target_transform(self.df)
        else:
            self.target_scaler = f_scale
            self.feature_scalers = t_scale

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


def split_df(df, split_frac=0.8, sample_frac=1):
    """
    Random splitting of dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    split_frac : float or int
        [0,1]. Split of train/validation.
    sample_frac : float or int, optional
        [0,1]. Total fraction of data.
    train_ds, valid_ds : torch.utils.data.Dataset
    bs : int

    Returns
    -------
    train_df, valid_df : tuple of pd.DataFrame
    """
    # Collect subset of dataframe if specified. Must use copy here!
    new_df = df.sample(sample_frac).copy() if sample_frac < 1 else df.copy()
    # Split the df
    train_df, valid_df = train_test_split(new_df, train_size=split_frac)
    return train_df, valid_df


def get_data(train_ds, valid_ds, bs):
    """
    Create dataloaders based on train/validation datasets and batch size.

    Parameters
    ----------
    train_ds, valid_ds : torch.utils.data.Dataset
    bs : int

    Returns
    -------
    train_dl, valid_dl : torch.utils.data.DataLoader
    """
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2 * bs)
    return train_dl, valid_dl


class WrappedDataLoader:
    """
    Applied preprocessing function to torch.utils.Data.Dataloader objects.

    Parameters
    ----------
    dl : torch.utilts.Data.Dataloader
    func : function()

    Attributes
    ----------
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

    df = pd.read_csv(DATA_PATH / FILE).astype("float64")
    df = df.loc[df["stargazers"] >= 100].reset_index(drop=True)
    train_df, valid_df = split_df(df)
    train_ds = GitStarDataset(train_df)
    valid_ds = GitStarDataset(
        valid_df,
        f_scale=train_ds.feature_scalers,
        t_scale=train_ds.target_scaler,
    )
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=64)
    for xb, yb in train_dl:
        print(xb, yb)


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
