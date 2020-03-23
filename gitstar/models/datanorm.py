"""
Normalizes/Transforms/Scales GitStar features and target. Provides necessary
function for inverse target transformation.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# from gitstar.models.dataload import GitStarDataset



class Log10Transformer():
    def __init__(self):
        pass

    def fit_transform(self, array):
        # Set lower bound
        array = np.clip(array, 1e-1, None)
        return np.log10(array)

    def inverse_transform(self, array):
        array = 10**array
        array[array <= 0.1] = 0
        return array

class IdentityTransformer():
    def __init__(self):
        pass

    def fit_transform(self, array):
        return array

    def inverse_transform(self, array):
        return array


FEATURE_SCALERS = {
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
    "releases": Log10Transformer(),
    "milestones": Log10Transformer(),
    "issuelabels": MinMaxScaler(),
    "created": MinMaxScaler(),
    "updated": MinMaxScaler(),
}
TARGET_SCALER = {"stargazers": Log10Transformer()}

def col_transform(df, col, scaler):
    """
    Scale single pandas dataframe column based on sklearn scaler.
    Warning: Transforms in-place.

    Parameters
    ----------
    data : pd.DataFrame
        Must have one or more named columns
    col : str
        Column name within DataFrame
    scaler : sklearn.preprocessing.scaler
        E.g. MinMaxScaler(), PowerTransformer(method="box-cox"),
        PowerTransformer(method="yeo-johnson"),
        QuantileTransformer(output_distribution="normal")

    Returns
    -------
    data : pd.DataFrame
        Transformed dataframe.
    scaler : sklearn.preprocessing.scaler
        Object holds fit attributes, e.g. lambdas_ for PowerTransformer()
    """
    # Extract column data
    col_data = df.loc[:, col].values.reshape(-1, 1)
    # Apply transformation. Returns nd.array
    newdata = scaler.fit_transform(col_data)
    # Transform new column in place
    df.loc[:, col] = newdata
    return df, scaler


def feature_transform(df, col_scaler=FEATURE_SCALERS):
    """
    Scale pandas DataFrame according to FEATURE_SCALERS.
    Warning: Transforms in-place.

    Parameters
    ----------
    data : pd.DataFrame
    col_scaler : dict {"col_name" : sklearn.preprocessing.scaler()}, optional

    Returns
    -------
    scaler_dict : dict of sklearn.preprocessing.scaler()
        Holds inverse function, accessible via
        scaler().inverse_transform(X), X : nd.array.
    """
    scaler_dict = {}
    for key in col_scaler:
        _, scaler = col_transform(df, key, col_scaler[key])
        scaler_dict[key] = scaler
    return scaler_dict


def target_transform(df, col_scaler=TARGET_SCALER):
    """
    Scale pandas DataFrame according to TARGET_SCALER.
    Warning: Transforms in-place.

    Parameters
    ----------
    data : pd.DataFrame
    col_scaler : dict {"col_name" : sklearn.preprocessing.scaler()}, optional

    Returns
    -------
    scaler_dict : dict of sklearn.preprocessing.scaler()
        Holds inverse function, accessible via
        scaler().inverse_transform(X), X : nd.array.
    """
    scaler_dict = {}
    for key in col_scaler:
        _, scaler = col_transform(df, key, col_scaler[key])
        scaler_dict[key] = scaler
    return scaler_dict


def module_test():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    df = pd.read_csv(DATA_PATH / SAMPLE_FILE)
    dfa = df.copy()
    target_transform(dfa)
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on

    # tdata = data.copy()
    # tdata, scaler = target_transform(tdata)
    # ndata = tdata.copy()


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
