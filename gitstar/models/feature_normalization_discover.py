from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import arrow

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILENAME_test = "query1.csv"
FILENAME = "gs_table_v2.csv"


def main():
    data = pd.read_csv(DATA_PATH / FILENAME)
    ti = int(arrow.get("2017-06-01").format("X"))
    tf = int(arrow.get("2018-01-01").format("X"))
    data_sub = data[data["created"].between(ti, tf)]
    # print histograms of columns
    for col in data_sub.iloc[:, 1:]:
        qtl = data_sub[col].quantile(0.9)
        ax = data_sub[col].plot.hist(bins=100, title=col, range=(0, qtl))
        fig = ax.get_figure()
        # plt.show()
        fig.savefig(
            IMG_PATH / ("{}_hist.png".format(col)),
            transparent=False,
            dpi=300,
            bbox_inches="tight",  # fit bounds of figure to plot
        )
        plt.close(fig)
    # print scatter plot of col vs stargazers
    for col in data_sub.iloc[:, 1:]:
        ax = data_sub.plot.scatter(x=col, y="stargazers", s=5, alpha=0.5)
        qtl = data_sub[col].quantile(0.9)
        ax.set_xlim([-0.1 * qtl, 1.1 * qtl])
        fig = ax.get_figure()
        fig.savefig(
            IMG_PATH / ("{}_star.png".format(col)),
            transparent=False,
            dpi=300,
            bbox_inches="tight",  # fit bounds of figure to plot
        )
        plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
