"""
Normalizes/Transforms/Scales GitStar features and target. Provides necessary
function for inverse target transformation.
"""
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# from gitstar.models.dataload import GitStarDataset


FEATURE_SCALERS = {
    "repositoryTopics": MinMaxScaler(),
    "openissues": PowerTransformer(method="yeo-johnson"),
    "closedissues": PowerTransformer(method="yeo-johnson"),
    "forkCount": PowerTransformer(method="yeo-johnson"),
    "pullRequests": PowerTransformer(method="yeo-johnson"),
    "commitnum": PowerTransformer(method="yeo-johnson"),
    "watchers": PowerTransformer(method="yeo-johnson"),
    "readme_bytes": PowerTransformer(method="yeo-johnson"),
    "deployments": PowerTransformer(method="yeo-johnson"),
    "descr_len": PowerTransformer(method="yeo-johnson"),
    "diskUsage_kb": QuantileTransformer(output_distribution="normal"),
    "projects": QuantileTransformer(output_distribution="normal"),
    "milestones": QuantileTransformer(output_distribution="normal"),
    "issuelabels": QuantileTransformer(output_distribution="normal"),
    "created": QuantileTransformer(output_distribution="normal"),
    "updated": QuantileTransformer(output_distribution="normal"),
}
TARGET_SCALER = {"stargazers": PowerTransformer(method="box-cox")}


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

    data = pd.read_csv(DATA_PATH / SAMPLE_FILE)
    feature_transform(data)

    # tdata = data.copy()
    # tdata, scaler = target_transform(tdata)
    # ndata = tdata.copy()


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
