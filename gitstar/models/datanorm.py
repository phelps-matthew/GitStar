"""
Scales GitStar features and target variables.
* Implements sklearn.preprocessing scaling functions
* Capable of storing fit parameters for allow inverse transformations
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def scale_cols(df, scaler_dict=None):
    """
    Scale multiple pandas DataFrame columns

    Warning: Transforms in-place

    Parameters
    ----------
    data : pandas DataFrame
    scaler_dict : dict of sklearn.preprocessing.scaler(), default None
        Follows {"col_name" : MyScaler(), ...}

    Returns
    -------
    fitted_dict : dict of sklearn.preprocessing.scaler()
        Scalers contain inverse function, accessible via
        scaler().inverse_transform(ndarray)
    """
    fitted_dict = {}
    for col in scaler_dict:
        # If supplied scalers, apply them
        if scaler_dict is not None:
            # transform in place
            _, scaler = scale_col(df, col, scaler_dict[col])
            # Construct dict of fitted scalers; useful for inv. params
            fitted_dict[col] = scaler
        # Otherwise apply identity transformation
        else:
            fitted_dict[col] = FunctionTransformer()
    return fitted_dict


def scale_col(df, col, scaler):
    """
    Scale single pandas dataframe column based on sklearn scaler.
    Warning: Transforms in-place.

    Parameters
    ----------
    data : pandas DataFrame
        Must have one or more named columns
    col : str
        Column name within DataFrame
    scaler : sklearn.preprocessing.scaler
        E.g. MinMaxScaler(), Log10Transformer,
        QuantileTransformer(output_distribution="normal"),
        PowerTransformer(method="yeo-johnson"),

    Returns
    -------
    data : pandas DataFrame
        Scale transformed dataframe
    scaler : sklearn.preprocessing.scaler
        Object may hold fit attributes, e.g. lambdas_ for PowerTransformer()
    """
    # Extract column data, reshape
    col_data = df.loc[:, col].values.reshape(-1, 1)
    # Apply scale transformation; returns ndarray
    newdata = scaler.fit_transform(col_data)
    # Transform new column in place
    df.loc[:, col] = newdata
    return df, scaler


# Define custom log scaler to be passed to FunctionTransformer()
# This allows user to switch generically between custom and sklearn scalers


def log10_map(array):
    """
    Apply scale transformation log10(max(x, 0.1))

    Parameters
    ----------
    array : ndarray

    Returns
    -------
    ndarray
    """
    # Apply max(x, 0.1)
    array = np.clip(array, 1e-1, None)
    # Apply log10
    return np.log10(array)


def log10_inv_map(array):
    """
    Apply inverse scale transformation

    y=10^x; for y in [0,0.1], min(y,0)

    Parameters
    ----------
    array : ndarray

    Returns
    -------
    array : ndarray
    """
    # Invert log10
    array = 10 ** array
    # For x in [0,0.1], apply min(x, 0)
    array[array <= 0.1] = 0
    return array


# Compose custom log-based scale transformer
Log10Transformer = FunctionTransformer(
    log10_map, log10_inv_map, check_inverse=False
)
# Construct useful defaults for feature and target scalers
FEATURE_SCALERS = {
    "repositoryTopics": Log10Transformer,
    "openissues": Log10Transformer,
    "closedissues": Log10Transformer,
    "forkCount": Log10Transformer,
    "pullRequests": Log10Transformer,
    "commitnum": Log10Transformer,
    "watchers": Log10Transformer,
    "readme_bytes": Log10Transformer,
    "deployments": Log10Transformer,
    "descr_len": Log10Transformer,
    "diskUsage_kb": Log10Transformer,
    "projects": Log10Transformer,
    "releases": Log10Transformer,
    "milestones": Log10Transformer,
    "issuelabels": MinMaxScaler(),
    "created": MinMaxScaler(),
    "updated": MinMaxScaler(),
}
TARGET_SCALER = {"stargazers": Log10Transformer}


def module_test():
    """Test the transformer functions"""

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataset"
    FILE = "gs_table_v2.csv"
    SAMPLE_FILE = "10ksample.csv"

    df = pd.read_csv(DATA_PATH / SAMPLE_FILE)
    dfa = df.copy()
    scale_cols(dfa, FEATURE_SCALERS)


if __name__ == "__main__":
    try:
        module_test()
    except KeyboardInterrupt:
        exit()
