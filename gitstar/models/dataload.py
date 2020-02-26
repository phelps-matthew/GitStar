"""Data handling for NN"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch


class GitStarDataset(Dataset):
    """GitStar Dataset.
    Args:
        csv_file (str, Path): Path to the csv file. Target must be col. 0.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path).astype("float64").iloc[:10, :]
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # pd.df -> np.ndarray with dtype float64
        x_sample = self.data_frame.iloc[idx, 1:].values
        y_sample = self.data_frame.iloc[idx, 0]
        # np.ndarry -> torch.tensor. float() to match default weights
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.tensor([y_sample]).float()

        if self.transform:
            x_sample = self.transform(x_sample)

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
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True)
    return train_dl, valid_dl



def main():
    """Test class implementations"""

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "gs_table_v2_sample.csv"

    dataset = GitStarDataset(DATA_PATH / SAMPLE_FILE)
    train_ds, valid_ds = rand_split_rel(dataset, 0.8)
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
