"""Data handling for NN"""
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class GitStarDataset(Dataset):
    """GitStar dataset."""

    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_file (str, Path): Path to the csv file. Target must be col. 0.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_path).astype('float64')
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # pd.df -> np.ndarray with dtype float64
        x_sample = self.data_frame.iloc[idx, 1:].values
        y_sample = self.data_frame.iloc[idx, 0]
        # np.ndarry -> torch.tensor
        x_sample = torch.from_numpy(x_sample)
        y_sample = torch.tensor(y_sample)

        if self.transform:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample


def main():
    """Test class implementations"""

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILENAME = "gs_table_v2.csv"
    CSV_PATH = DATA_PATH / FILENAME

    dataset = GitStarDataset(CSV_PATH)
    print(dataset[1])
    train_frac = 0.7
    train_len = int(train_frac * len(dataset))
    trainset, valset = random_split(
        dataset, [train_len, len(dataset) - train_len]
    )
    print(trainset)

    train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
    val_loader = DataLoader(valset, batch_size=10, shuffle=True)

    for i, batch in enumerate(train_loader):
        print(i, batch)

    for i, batch in enumerate(val_loader):
        print(i, batch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
