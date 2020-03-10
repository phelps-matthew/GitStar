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
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from gitstar.models.dataload import GitStarDataset
from gitstar.models.datanorm import (
    Log10Transformer,
    IdentityTransformer,
    feature_transform,
)
from sklearn.preprocessing import MinMaxScaler

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
    ax.scatter(x, y, marker="o", s=4, alpha=0.4)
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
    f_logs = {
        "repositoryTopics": Log10Transformer(),
        "openissues": Log10Transformer(),
        "closedissues": Log10Transformer(),
        "forkCount": Log10Transformer(),
        "pullRequests": Log10Transformer(),
        "commitnum": Log10Transformer(),
        "watchers": Log10Transformer(),
        "readme_bytes": Log10Transformer(),
        "deployments": Log10Transformer(),
        "descr_len": Log10Transformer(),
        "diskUsage_kb": Log10Transformer(),
        "projects": Log10Transformer(),
        "milestones": Log10Transformer(),
        "releases": Log10Transformer(),
        "issuelabels": MinMaxScaler(),
        "created": MinMaxScaler(),
        "updated": MinMaxScaler(),
    }
    t_logs = {"stargazers": Log10Transformer()}

    plotvars = (
        "stargazers",
        "openissues",
        "closedissues",
        "forkCount",
        "pullRequests",
        "commitnum",
        "watchers",
        "readme_bytes",
        "diskUsage_kb",
        "created",
        "updated",
    )

    df = pd.read_csv(DATA_PATH / FILE).astype("float64")
    df = df.loc[
        (df["stargazers"] >= 10)
        & (df["closedissues"] > 0)
        & (df["commitnum"] > 1)
        & (df["readme_bytes"] > 0)
    ]
    trans_df = df.copy()
    trans_df = GitStarDataset(trans_df).df
    # Plotting
    sns.set()
    # g = sns.jointplot("stargazers", "forkCount", trans_df, kind="reg")
    g = sns.PairGrid(trans_df, vars=("stargazers", "commitnum"))
    g = g.map_diag(sns.distplot)
    g = g.map_offdiag(sns.scatterplot, alpha=0.5)

    xlabels, ylabels = [], []
    for ax in g.axes[-1, :]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)
    for ax in g.axes[:, 0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)
    ylabels = [a if a else " " for a in ylabels]

    for j in range(len(xlabels)):
        for i in range(len(ylabels)):
            g.axes[j, i].set(xlabel=xlabels[i], ylabel=ylabels[j])

    # multi_scatter = sns.pairplot
    #    data=trans_df,
    #    vars=("stargazers", "closedissues", "updated", "created",),
    #    kind="reg",
    #    diag_kind="kde",
    #    diag_kws=dict(shade=True),
    #    plot_kws={
    #        "scatter_kws": {"alpha": 0.5},
    #        "line_kws": {"color": "orange"},
    #    },
    # )
    plt.tight_layout()
    #plt.show()
    g.fig.savefig(
        str(IMG_PATH / "full_seaborn/log_canonical_scatter.pdf"),
        transparent=False,
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
