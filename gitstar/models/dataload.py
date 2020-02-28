"""Data handling for DFF NN"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
from gitstar.models.datanorm import feature_transform, target_transform


class GitStarDataset(Dataset):
    """
        Args:
            csv_file (str, Path): Path to the csv file.

            transform=True (boolean): Apply scale transformations according to
            datanorm module.

        Attributes:
            data (pd.DataFrame): Entire dataset

            transform (boolean): Apply scale transformation

            target_inv_fn (sklearn.preprocessing.scaler()): Scaler object that holds
            target fit parameters. Access inv. function via
            target_inv_fn.inverse_transform(X), X (nd.array)

            features (pd.DataFrame):

            target (pd.DataFrame):
    """

    def __init__(self, csv_path, sample_frac=1, transform=True):
        # Load data, initialize attributes
        self.data = pd.read_csv(csv_path).astype("float64")
        self.target_inv_fn = None
        self.transform = transform

        # Slice data frame according to sample_frac
        sample_size = int(sample_frac * len(self.data))
        self.data = self.data.iloc[:sample_size, :]

        # Apply data scaling
        if self.transform:
            feature_transform(self.data)
            _, self.target_inv_fn = target_transform(self.data)

        # Separate features and target. Drop method returns deep copy of df
        self.features = self.data.drop("stargazers", axis=1)
        self.target = self.data["stargazers"].to_frame()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # pd.df -> np.ndarray with dtype float64
        x_sample = self.features.iloc[idx].values
        y_sample = self.target.iloc[idx].values
        # np.ndarry -> torch.tensor.float() to match weights datatype
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.tensor([y_sample]).float()

        return x_sample, y_sample


def rand_split_rel(dataset, frac, **kwargs):
    """Splits dataset as fraction of total. Based on torch random_split.
        Args:
            dataset (torch.utils.data.Dataset)
            frac (float): [0,1]
        Return:
            (ds_frac, ds_remainder) (tuple)
    """
    size_1 = int(frac * len(dataset))
    size_2 = len(dataset) - size_1
    return random_split(dataset, [size_1, size_2], **kwargs)


def get_data(train_ds, valid_ds, bs):
    """Create dataloaders based on train/test datasets and batch size"""
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2 * bs)
    return train_dl, valid_dl


def module_test():
    """Test class implementations"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    data = pd.read_csv(DATA_PATH / SAMPLE_FILE).astype("float64")

    # data.iloc[:10000, :].to_csv(DATA_PATH / SAMPLE_FILE, index=False)

    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=1)


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
