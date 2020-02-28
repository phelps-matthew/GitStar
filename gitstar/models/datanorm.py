"""Normalizes/Transforms/Scales GitStar features and target. Provides necessary
    function for inverse target transformation.
"""
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
#from gitstar.models.dataload import GitStarDataset


FEATURE_SCALERS = [
    ("repositoryTopics", MinMaxScaler()),
    ("openissues", PowerTransformer(method="yeo-johnson")),
    ("closedissues", PowerTransformer(method="yeo-johnson")),
    ("forkCount", PowerTransformer(method="yeo-johnson")),
    ("pullRequests", PowerTransformer(method="yeo-johnson")),
    ("commitnum", PowerTransformer(method="yeo-johnson")),
    ("watchers", PowerTransformer(method="yeo-johnson")),
    ("readme_bytes", PowerTransformer(method="yeo-johnson")),
    ("deployments", PowerTransformer(method="yeo-johnson")),
    ("descr_len", PowerTransformer(method="yeo-johnson")),
    ("diskUsage_kb", QuantileTransformer(output_distribution="normal")),
    ("projects", QuantileTransformer(output_distribution="normal")),
    ("milestones", QuantileTransformer(output_distribution="normal")),
    ("issuelabels", QuantileTransformer(output_distribution="normal")),
    ("created", QuantileTransformer(output_distribution="normal")),
    ("updated", QuantileTransformer(output_distribution="normal")),
]
TARGET_SCALER = ("stargazers", PowerTransformer(method="box-cox"))


def col_transform(df, col, scaler):
    """Scale single pandas dataframe column based on sklearn scaler.
        Warning: Transforms in-place.
        Args:
            data (pd.DataFrame): Must have one or more named columns

            col (str): Column name within DataFrame

            scaler (sklearn.preprocessing.scaler):
            E.g. MinMaxScaler(), PowerTransformer(method="box-cox"),
            PowerTransformer(method="yeo-johnson"),
            QuantileTransformer(output_distribution="normal")

        Return:
            data (pd.DataFrame): Transformed dataframe
            scaler (sklearn.preprocessing.scaler): Object holds fit attributes,
            e.g. lambdas_ for PowerTransformer
    """
    # Extract column data
    col_data = df[col].values.reshape(-1, 1)
    # Apply transformation. Returns nd.array
    newdata = scaler.fit_transform(col_data)
    # Transform new column in place
    df[col] = newdata
    return df, scaler


def feature_transform(df, col_scaler_list=FEATURE_SCALERS):
    """Scale pandas DataFrame according to FEATURE_SCALERS. Does not return fit
        objects. Warning: Transforms in-place.
        Args:
            data (pd.DataFrame): Must adhere to FEATURE_SCALERS format.
            col_scaler_list (list of tuples): Default FEATURE_SCALERS.

        Return:
            data (pd.DataFrame)
    """
    for feature in col_scaler_list:
        col_transform(df, *feature)
    return df


def target_transform(df, col_scaler=TARGET_SCALER):
    """Scale pandas DataFrame target column according to TARGET_SCALER. Provides
        inverse fit function. Warning: Transforms in-place.
        Args:
            data (pd.DataFrame): Must adhere to FEATURE_SCALERS format.

            col_scaler_list (list of tuples): Default FEATURE_SCALERS.

        Return:
            data (pd.DataFrame)

            scaler (sklearn.preprocessing.scaler()): Holds inverse function,
            accessible via target_inv_fn.inverse_transform(X), X (nd.array)

    """
    df, scaler = col_transform(df, *col_scaler)
    return df, scaler


def module_test():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    #data = GitStarDataset(DATA_PATH / SAMPLE_FILE, 1).df
    # feature_transform(data)

    #tdata = data.copy()
    #tdata, scaler = target_transform(tdata)
    #ndata = tdata.copy()


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
