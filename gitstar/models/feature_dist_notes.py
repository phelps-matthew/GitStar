"""This script explores the distributions and properties of input features, as
    well as the output. Most importantly, we explore various
    normalization/standardization tranforms from sklearn.preprocessing.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import arrow
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILENAME = "gs_table_v2.csv"
DISTS = {
    "standard scaling": StandardScaler(),
    "min-max scaling": MinMaxScaler(),
    "max-abs scaling": MaxAbsScaler(),
    "robust scaling": RobustScaler(quantile_range=(25, 75)),
    "power transformation (Yeo-Johnson)": PowerTransformer(
        method="yeo-johnson"
    ),
    "power transformation (Box-Cox)": PowerTransformer(method="box-cox"),
    "quantile transformation (gaussian pdf)": QuantileTransformer(
        output_distribution="normal"
    ),
    "quantile transformation (uniform pdf)": QuantileTransformer(
        output_distribution="uniform"
    ),
    "sample-wise L2 normalizing": Normalizer(),
}


def gen_hist(img_path, data, qtl=None):
    """Generate histograms of columns"""
    img_path.mkdir(parents=True, exist_ok=True)
    for col in data:
        if qtl:
            qrange = (data[col].min(), 1.1 * data[col].quantile(qtl))
        else:
            qrange = (data[col].min(), data[col].max())
        ax = data[col].plot.hist(
            bins=2000, title=col, range=qrange, align="mid"
        )
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
        os.system("clear")


def gen_scatter(img_path, data, qtl=None):
    """Generate scatter plots of columns vs stargazers"""
    img_path.mkdir(parents=True, exist_ok=True)
    for col in data.iloc[:, 1:]:
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


def input_menu(data):
    """Input menu for gathering dataframe column and scale trans. type.

        Args:
            data (pandas dataframe or string iterable)
        Return:
            col name (str), scaler type (str)
    """
    # User input column
    print("Data Columns:")
    for col in data:
        print("[{}] {}".format(data.columns.get_loc(col), col))
    col_in = int(input("Select Column: "))

    # User input scale transformation
    print("\nScale Transformations:")
    dist_keys = list(DISTS.keys())
    for i in range(len(dist_keys)):
        print("[{}] {}".format(i, dist_keys[i]))
    scale_in = int(input("Select Scaler: "))
    return data.columns[col_in], dist_keys[scale_in]


def scale_hist(img_path, data, col, scaler):
    """Generate histograms of given single column

        Args:
            img_path (Path, str)
            data (pd.df)
            col (str): Single df col name
            scaler (str): based on DISTS dict
    """
    img_path.mkdir(parents=True, exist_ok=True)
    transformer = DISTS[scaler]
    # deep df copy
    tdata = data[["hasUrl", col]].copy()
    # apply scaler to all cols
    tdata[tdata.columns] = transformer.fit_transform(tdata[tdata.columns])
    tplt = plt.figure(1)
    t_ax = tdata[col].plot.hist(bins=2000, title="{}: {}".format(col, scaler))
    print("\nStatistics: \n{}\n".format(data[col].describe()))
    tplt.show()
    rplt = plt.figure(2)
    ax = data[col].plot.hist(bins=2000, title=col)
    rplt.show()
    # Method for continutation w/ matplotlib
    input("Press any key to continue, ctrl+z to exit.")
    plt.close()
    os.system("clear")


def main():
    mypath = "transformed/full"
    data = pd.read_csv(DATA_PATH / FILENAME)
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
    while True:
        col, scaler = input_menu(data)
        scale_hist(IMG_PATH / mypath, data, col, scaler)

    ###############################################
    # ti = int(arrow.get("2017-06-01").format("X"))
    # tf = int(arrow.get("2018-01-01").format("X"))
    # star_min = 100
    # qtl = None
    # data_sub = data[data["created"].between(ti, tf)]
    # data = data[data["stargazers"] >= star_min]
    # gen_hist(IMG_PATH/mypath, data, qtl=qtl)
    # gen_scatter(IMG_PATH/mypath, data, qtl=qtl)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()