"""
This script explores the distributions and properties of input features, as
well as the output. Most importantly, we explore various
normalization/standardization transforms from sklearn.preprocessing.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

from gitstar.models.dataload import GitStarDataset

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features" / "full_seaborn"
IMG_PATH.mkdir(parents=True, exist_ok=True)
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"

# Custom color map for hex bin plot
cmap = LinearSegmentedColormap.from_list(
    "mycmap", ["#718ccc", "#3a4869", "#2e3a54", "#1f2638"]
)
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
    """Workspace for generating plots."""

    df = pd.read_csv(DATA_PATH / FILE).astype("float64")
    data = canonical_data(df)
    data_time = data.copy()
    data_time.loc[:,'updated'] = 10**(-6)*(data.loc[:,'updated'].values)
    data_time.loc[:,'created'] = 10**(-12)*(data.loc[:,'created'].values)

    # Pair wise plotting columns
    x = "updated"
    y = "stargazers"
    linreg_print(x, y, data)

    # Initialize plot setup
    formatter = FuncFormatter(log_label)
    sns.set()
    # sns.set_context("talk")

    # Plots
    p = sns.JointGrid(x, y, data_time)
    p = p.plot_joint(
        plt.hexbin, mincnt=1, cmap=cmap, gridsize=60, edgecolors="white",
    )
    p.plot_marginals(sns.distplot)

    # Scale to be square, plot regression
    hex_xlim = p.ax_joint.get_xlim()
    hex_ylim = p.ax_joint.get_ylim()
    p = p.plot_joint(sns.regplot, scatter=False)
    reg_xlim = p.ax_joint.get_xlim()
    reg_ylim = p.ax_joint.get_ylim()
    y2max = get_xylims(hex_xlim, hex_ylim, reg_xlim, reg_ylim)
    p.ax_joint.set_ylim(reg_ylim[0], y2max)

    # Retitle if desired
    p.set_axis_labels(xlabel="Last Pushed", ylabel="Stars")

    # For timelike data
    xticks = p.ax_joint.get_xticks()
    p.ax_joint.set_xticklabels(
        [
            pd.to_datetime(10**6*tm, unit="s").strftime("%Y-%m-%d")
            for tm in xticks
        ],
        rotation=50,
    )

    # Format log style axes. Annotate stats.
    # p.ax_joint.xaxis.set_major_formatter(formatter)
    p.ax_joint.yaxis.set_major_formatter(formatter)
    p.annotate(stats.pearsonr)
    plt.tight_layout()
    #plt.show()

    # Save figure
    save_fig(p.fig, IMG_PATH / "stars_updated_hex_reg.png")


def get_xylims(axes1_x, axes1_y, axes2_x, axes2_y):
    w1 = abs(axes1_x[1] - axes1_x[0])
    h1 = abs(axes1_y[1] - axes1_y[0])
    w2 = abs(axes2_x[1] - axes2_x[0])
    h2 = abs(axes2_y[1] - axes2_y[0])
    y2_max = axes2_y[1]
    aspect1 = h1 / w1
    aspect2 = h2 / w2
    # scale x2
    if aspect2 < aspect1:
        h2 = aspect1 * w2
        y2_max = axes2_y[0] + h2
    return y2_max


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


def log_label(x, pos):
    """Formats axis labels to power of 10.
        Follows matplotlib example.
    """
    if x.is_integer():
        return r"$10^{%d}$" % (x)
    else:
        return ""


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
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path), transparent=False, dpi=300, bbox_inches="tight",
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


def canonical_data(data, transform=True):
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
        (data["stargazers"] >= 10)
        & (data["closedissues"] > 0)
        & (data["commitnum"] > 1)
        & (data["readme_bytes"] > 0)
        & (data["watchers"] > 0)
        & (data["forkCount"] > 0)
        & (data["diskUsage_kb"] > 0)
        & (data["readme_bytes"] > 0)
        & (data["pullRequests"] > 0)
    ].copy()
    trans_df = GitStarDataset(c_data, transform=transform).df
    return trans_df


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
