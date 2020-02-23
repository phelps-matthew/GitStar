from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import arrow
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILENAME = "gs_table_v2.csv"


def gen_hist(img_path, data, qtl=None):
    """Generate histograms of columns"""
    img_path.mkdir(parents=True, exist_ok=True)
    for col in data:
        if qtl:
            qrange = (data[col].min(), 1.1*data[col].quantile(qtl))
        else:
            qrange = (data[col].min(), data[col].max())
        ax = data[col].plot.hist(bins=2000, title=col, range=qrange, align='mid')
        fig = ax.get_figure()
        input("Press Return to continue")
        print(data[col].describe())
        plt.show()
        # fig.savefig(
        #     img_path / ("{}_hist.png".format(col)),
        #     transparent=False,
        #     dpi=300,
        #     bbox_inches="tight",  # fit bounds of figure to plot
        # )
        plt.close(fig)
        os.system('clear')


def gen_scatter(img_path, data, qtl=None):
    """Generate scatter plots of columns vs stargazers"""
    img_path.mkdir(parents=True, exist_ok=True)
    for col in data.iloc[:,1:]:
        ax = data.plot.scatter(x=col, y="stargazers", s=5, alpha=0.5)
        if qtl:
            ax.set_xlim(left=data[col].min(), right=data[col].quantile(qtl))
            ax.margins(x=0.05)
            ax.autoscale(True)
            # print("{}:".format(col)+str(1.1*data[col].quantile(qtl)))
        ax.set_ylim(top=100000)  # outlier 996.ICU
        fig = ax.get_figure()
        fig.savefig(
            img_path / ("{}_star.png".format(col)),
            transparent=False,
            dpi=300,
            bbox_inches="tight",  # fit bounds of figure to plot
        )
        plt.close(fig)


def main():
    data = pd.read_csv(DATA_PATH / FILENAME)
    ti = int(arrow.get("2017-06-01").format("X"))
    tf = int(arrow.get("2018-01-01").format("X"))
    star_min = 100
    # data_sub = data[data["created"].between(ti, tf)]
    #data = data[data["stargazers"] >= star_min]
    mypath = "full_qtl_90"
    qtl = None
    gen_hist(IMG_PATH/mypath, data, qtl=qtl)
    #gen_scatter(IMG_PATH/mypath, data, qtl=qtl)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
