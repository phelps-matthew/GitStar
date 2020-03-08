"""
This script explores the distributions and properties of input features, as
well as the output. Most importantly, we explore various
normalization/standardization transforms from sklearn.preprocessing.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import arrow
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from gitstar.models.dataload import GitStarDataset

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"
DISTS = {
    "no scaling": None,
    "standard scaling": StandardScaler(),
    "min-max scaling": MinMaxScaler(),
    # "max-abs scaling": MaxAbsScaler(),
    # "robust scaling": RobustScaler(quantile_range=(25, 75)),
    "log scaling": np.log,
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
    # "sample-wise L2 normalizing": Normalizer(),
}


def gen_hist(img_path, data, qtl=None):
    """
    Generate histograms of columns.

    Parameters
    ----------
    img_path : str or Path
    data : DataFrame
    qtl : float, optional
        [0,1]

    Returns
    -------
    None
    """
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


def gen_scatter(x, y, path=None, xlabel="x", ylabel="y", title=None):
    """
    Generate scatter plot of x vs y.

    Parameters
    ----------
    x, y : nd.array
    path : str or Path, optional
    xlabel, ylabel, title : str, optional

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="o", s=4, alpha=0.2)
    # ax.set_ylim(ylim)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if path:
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(path), transparent=False, dpi=300, bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        input("Press any key to continue")
        plt.close()


def col_menu(data):
    """
    Convenient input menu for fetching gitstar dataframe column name.

    Parameters
    ----------
    data : pd.Dataframe or iterable of str

    Returns
    _______
    column : str
    """
    # User input column
    print("Data Columns:")
    for col in data:
        print("[{}] {}".format(data.columns.get_loc(col), col))
    col_in = int(input("Select Column: "))
    return data.columns[col_in]


def scaler_menu():
    """
    Input menu for selecting scaler type.

    Returns
    -------
    str
        scaler key.
    """
    # User input scale transformation
    print("\nScale Transformations:")
    dist_keys = list(DISTS.keys())
    for i in range(len(dist_keys)):
        print("[{}] {}".format(i, dist_keys[i]))
    scale_in = int(input("Select Scaler: "))
    return dist_keys[scale_in]


def scatter_menu_loop(df):
    """
    Input menu for selecting scaler type.

    Parameters
    ----------
    data : pd.Dataframe

    Returns
    -------
    col_x, col_y : str
        Selected columns
    """

    # Run menu for selecting two df columns
    while True:
        input("Press any key to continue, ctrl+z to exit.")
        col_x = col_menu(df)
        col_y = col_menu(df)
        print(col_x, col_y)
        array_x = df.loc[:, col_x].values
        array_y = df.loc[:, col_y].values
        gen_scatter(array_x, array_y, xlabel=col_x, ylabel=col_y)


def scale_hist(data, col, scaler):
    """
    Generate histograms of given single column.

    Parameters
    ----------
    data : DataFrame
    col : str
        Single DataFrame col name.
    scaler : str
        based on DISTS dict.

    Returns
    -------
    None
    """
    transformer = DISTS[scaler]
    # deep df copy. transformer needs at least 2 cols
    tdata = data[col].to_frame()
    # make column transformer
    ct = make_column_transformer((transformer, [col]))
    # nd array
    new_data = ct.fit_transform(tdata)
    # make new df
    newdf = pd.DataFrame(new_data, columns=[col])
    # apply scaler to all cols
    # if isinstance(transformer, np.ufunc):
    #    tdata[tdata.columns] = np.log(tdata[tdata.columns])
    # elif transformer is not None:
    #    tdata[tdata.columns] = transformer.fit_transform(tdata[tdata.columns])
    # tplt = plt.figure()
    # ax = tdata[col].plot.hist(bins=2000, title="{}: {}".format(col, scaler))
    print("\nStatistics - {}: \n{}\n".format(scaler, newdf[col].describe()))
    # fig = ax.get_figure()
    # fig.savefig(
    #    IMG_PATH / "transformed/full/{}_{}_hist.png".format(col, scaler),
    #    transparent=False,
    #    dpi=300,
    #    bbox_inches="tight",  # fit bounds of figure to plot
    # )
    # plt.close()
    # tplt.show()


def main():
    df = pd.read_csv(DATA_PATH / FILE).astype("float64")
    # df = df.loc[df["stargazers"] >= 100]
    trans_df = GitStarDataset(df).df
    scatter_menu_loop(trans_df)

    # qtl=0.9
    # qrange = (data[col].min(), 1.1 * data[col].quantile(qtl))
    # ax = mydata[col].plot.hist(bins=2000, range=qrange)
    # plt.show()

    # while True:
    #     col = col_menu(df)
    #     scaler = scaler_menu()
    #     print(col, scaler)
    #     scale_hist(df, col, scaler)
    #     input("Press any key to continue, ctrl+z to exit.")

    # Method for continutation w/ matplotlib
    # input("Press any key to continue, ctrl+z to exit.")
    # plt.close()
    # os.system("clear")

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
