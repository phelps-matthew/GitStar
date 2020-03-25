"""
Data handling for DFF model.

ToDo:
    Comment + Docstrings
    Possible wrappers for csv -> ds -> dataloader
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from gitstar.models.datanorm import (
    scale_cols,
    FEATURE_SCALERS,
    TARGET_SCALER,
)


class GitStarDataset(Dataset):
    """
    Implements torch.utils.data.Dataset with optional transformations upon
    the features and/or target

    Parameters
    ----------
    df : pandas.DataFrame
    transform : boolean, default True
        Apply default scale transformations according to datanorm module
    f_scale, t_scale : dict of sklearn.preprocessing.scaler(), default None
        Follows {"col_name" : MyScaler(), ...}

    Attributes
    ----------
    df : pandas.DataFrame
    transform : boolean
        Apply default scale transformations according to datanorm module
    f_scale, t_scale : dict of sklearn.preprocessing.scaler()
        Scalers contain inverse function, accessible via
        scaler().inverse_transform(ndarray)
    """

    def __init__(self, df, transform=True, f_scale=None, t_scale=None):
        self.df = df
        self.transform = transform

        # If scalers are provided, use them; otherwise use datanorm defaults
        if self.transform:
            self.feature_scalers = scale_cols(self.df, FEATURE_SCALERS)
            self.target_scaler = scale_cols(self.df, TARGET_SCALER)
        else:
            self.feature_scalers = scale_cols(self.df, f_scale)
            self.target_scaler = scale_cols(self.df, t_scale)

        # Separate features and target; drop method returns deep copy of df
        self.features = self.df.drop("stargazers", axis=1)
        self.target = self.df["stargazers"].to_frame()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Indexes feature and target arrays

        Parameters
        ----------
        idx : int, Index

        Returns
        -------
        x_sample, y_sample : torch.tensor.float()
        """
        # pd.df -> np.ndarray with dtype float64
        x_sample = self.features.iloc[idx].values
        y_sample = self.target.iloc[idx].values
        # np.ndarry -> torch.tensor.float() to match weight params datatype
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.tensor(y_sample).float()
        return x_sample, y_sample


class WrappedDataLoader:
    """
    Wrap torch.utils.Data.Dataloader with additional preprocessing function

    E.g. set torch device as GPU.

    Parameters
    ----------
    dl : torch.utilts.Data.Dataloader
    func : function
        Preprocessing function, e.g. GPU support

    Attributes
    ----------
    dl : torch.utilts.Data.Dataloader
    func : function

    Notes
    -----
    To serve as dataloader, must provide __len__ and __iter__ methods
    """

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = self.dl
        # generator naturally provides next method for iterating
        for b in batches:
            # unpack b, apply function; e.g. preprocess
            yield self.func(*b)


def form_dataloaders(train_ds, valid_ds, bs, preprocess=lambda x, y: (x, y)):
    """
    Create dataloaders based on train/validation datasets and batch size.

    Parameters
    ----------
    train_ds, valid_ds : GitStarDataset or torch.utils.data.Dataset
    bs : int
    preprocess : function, default lambda x, y: (x, y)
        e.g. GPU support, (x.to(dev), y.to(dev))

    Returns
    -------
    train_dl, valid_dl : torch.utils.data.DataLoader
    """
    # Form the torch DataLoaders
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2 * bs)

    # Apply preprocessing function
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    return train_dl, valid_dl


def form_datasets(path, sample_frac=1):
    """
    Form training and validation datasets from file

    Parameters
    ----------
    path : str or Path
        Filepath to csv data
    sample_frac : float or int, default 1
        sample_size/total_data_size

    Returns
    -------
    train_dl, valid_dl : torch.utils.data.DataLoader

    Notes
    -----
    Performs canonical GitStar data filtering
    """
    # Load the file into DataFrame
    df = pd.read_csv(path).astype("float64")
    # Filter data based on canonical GitStar criteria
    df = df.loc[
        (df["stargazers"] >= 10)
        & (df["closedissues"] > 0)
        & (df["commitnum"] > 1)
        & (df["readme_bytes"] > 0)
    ]
    # Split DataFrame into training/validation
    train_df, valid_df = split_df(df, sample_frac)
    # Form training Dataset object
    train_ds = GitStarDataset(train_df)
    # Form validation Dataset object; use scaling params from training Dataset
    valid_ds = GitStarDataset(
        valid_df,
        f_scale=train_ds.feature_scalers,
        t_scale=train_ds.target_scaler,
    )
    return train_ds, valid_ds


def split_df(df, split_frac=0.8, sample_frac=1):
    """
    Random splitting of DataFrame; generate train and validation DataFrames

    Parameters
    ----------
    df : pd.DataFrame
    split_frac : float or int
        (train)/(train+validation)
    sample_frac : float or int, optional
        sample_len/df_len

    Returns
    -------
    train_df, valid_df : tuple of pd.DataFrame
    """
    # Collect subset of dataframe if specified. Must use copy here!
    new_df = df.sample(sample_frac).copy() if sample_frac < 1 else df.copy()
    # Split the df
    train_df, valid_df = train_test_split(new_df, train_size=split_frac)
    return train_df, valid_df


def module_test():
    """Test functions and class implementations"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    SAMPLE_FILE = "10ksample.csv"

    df = pd.read_csv(DATA_PATH / SAMPLE_FILE).astype("float64")
    train_df, valid_df = split_df(df)
    train_ds = GitStarDataset(train_df)
    valid_ds = GitStarDataset(
        valid_df,
        f_scale=train_ds.feature_scalers,
        t_scale=train_ds.target_scaler,
    )
    train_dl, valid_dl = form_dataloaders(train_ds, valid_ds, bs=64)
    for xb, yb in train_dl:
        print(xb, yb)
        input("Press any key to continue, ctrl+z to exit")


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
