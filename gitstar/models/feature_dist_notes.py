"""
This script explores the distributions and properties of input features, as
well as the output. Most importantly, we explore various
normalization/standardization transforms from sklearn.preprocessing.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
import arrow
import os
import numpy as np
import seaborn as sns
from scipy import stats

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

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features" / "full_seaborn"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"

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


def main():

    df = pd.read_csv(DATA_PATH / FILE).astype("float64")
    data = canonical_data(df)

    # Plotting
    x = "stargazers"
    y = "forkCount"
    linreg_print(x, y, data)

    def log_label(x, pos):
        """The two args are the value and tick position"""
        return r"$10^{%d}$" % (x)

    formatter = FuncFormatter(log_label)

    sns.set()
    g = sns.jointplot(x, y, data, kind="reg", scatter=True)
    # g = sns.jointplot(x, y, data, kind="hex", ax_joint=g.ax_joint)
    g.ax_joint.xaxis.set_major_formatter(formatter)
    g.ax_joint.yaxis.set_major_formatter(formatter)
    # g = sns.regplot(x, y, data=data, ax=g.ax_joint, scatter=False)
    plt.tight_layout()
    plt.show()

    save_fig(g.fig, IMG_PATH/"log_canonical_scatter2.pdf")

    # multi_scatter = sns.pairplot
    #    kind="reg",
    #    diag_kind="kde",
    #    diag_kws=dict(shade=True),
    #    plot_kws={
    #        "scatter_kws": {"alpha": 0.5},
    #        "line_kws": {"color": "orange"},
    #    },
    # )


def save_fig(fig, path):
    """
    Generate figure.

    Parameters
    ----------
    fig : matplotlib.plt.fig
    path : str or Path
    
    Returns
    -------
    None
    """
    fig.savefig(
        str(path),
        transparent=False,
        dpi=300,
        bbox_inches="tight",
    )


def linreg_print(x, y, data):
    """
    Print linear regression stats.

    Parameters
    ----------
    x, y : str
    data : pandas.DataFrame
    
    Returns
    -------
    None
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data[x], data[y]
    )
    print(
        "slope: {}\nintercept: {}\nr_value: {}\np_value: {}\nstd_err: {}".format(
            slope, intercept, r_value, p_value, std_err
        )
    )


def canonical_data(data):
    """
    Procure canonical transformed dataset from full dataset.

    Parameters
    ----------
    data : pandas:DataFrame

    Returns
    -------
    trans_df : pandas.DataFrame
    """
    c_data = data.loc[
        (data["stargazers"] >= 1000)
        & (data["closedissues"] > 0)
        & (data["commitnum"] > 1)
        & (data["readme_bytes"] > 0)
    ].copy()
    trans_df = GitStarDataset(c_data).df
    return trans_df


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
