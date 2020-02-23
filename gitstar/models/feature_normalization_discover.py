from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import time

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILENAME = "query1.csv"


def main():
    data = pd.read_csv(DATA_PATH / FILENAME)

    # print histograms of columns
    for col in data.iloc[:, :4]:
        ax = data[col].plot.hist(bins=100, title=col)
        fig = ax.get_figure()
        # plt.show()
        fig.savefig(
            IMG_PATH / ("{}_hist.png".format(col)),
            transparent=False,
            dpi=100,
            bbox_inches="tight",  # fit bounds of figure to plot
        )
        ax = data.plot.scatter(x=col, y="stargazers")
        fig = ax.get_figure()
        fig.savefig(
            IMG_PATH / ("{}_star.png".format(col)),
            transparent=False,
            dpi=100,
            bbox_inches="tight",  # fit bounds of figure to plot
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
