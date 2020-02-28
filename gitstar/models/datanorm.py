from pathlib import Path
import pandas as pd
from itertools import chain

from gitstar.models.dataload import GitStarDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import make_column_transformer, ColumnTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
FILE = "gs_table_v2.csv"
SAMPLE_FILE = "10ksample.csv"
SCALER_COLS = {
    "MinMax": ["repositoryTopics"],
    "Box_Cox": ["stargazers"],
    "Yeo_Johnson": [
        "openissues",
        "closedissues",
        "forkCount",
        "pullRequests",
        "commitnum",
        "watchers",
        "readme_bytes",
        "releases",
        "deployments",
        "descr_len",
    ],
    "Quantile_Gaussian": [
        "diskUsage_kb",
        "projects",
        "milestones",
        "issuelabels",
        "created",
        "updated",
    ],
    "Identity": ["hasUrl", "hasLicense", "hasWiki", "hasIssue"],
}


def col_transform(df, col, scaler):
    """Scale single pandas dataframe column based on sklearn scaler.
        Warning: Transforms in-place.
        Args:
            df (pd.DataFrame): Must have one or more named columns

            col (str): Column name within DataFrame

            scaler (sklearn.preprocessing.scaler):
            E.g. MinMaxScaler(), PowerTransformer(method="box-cox"),
            PowerTransformer(method="yeo-johnson"),
            QuantileTransformer(output_distribution="normal")

        Return:
            df (pd.DataFrame): Transformed dataframe
            scaler (sklearn.preprocessing.scaler): Object holds fit attributes,
            e.g. lambdas_ for PowerTransformer
    """
    # Extract column data
    col_data = data[col].values.reshape(-1, 1)
    # Apply transformation. Returns nd.array
    newdata = scaler.fit_transform(col_data)
    # Transform new column in place
    df[col] = newdata
    return df, scaler


col_list = [col for key in SCALER_COLS for col in SCALER_COLS[key]]
GitStarDataset

data = GitStarDataset(DATA_PATH / SAMPLE_FILE, 1).df
newdata, fit = col_transform(data, "stargazers", PowerTransformer(method="box-cox"))

preprocess = make_column_transformer(
    (MinMaxScaler(), SCALER_COLS["MinMax"]),
    (PowerTransformer(method="box-cox"), SCALER_COLS["Box_Cox"]),
    (PowerTransformer(method="yeo-johnson"), SCALER_COLS["Yeo_Johnson"]),
    (
        QuantileTransformer(output_distribution="normal"),
        SCALER_COLS["Quantile_Gaussian"],
    ),
    remainder="passthrough",
)

new_array = preprocess.fit_transform(data)
newdf = pd.DataFrame(new_array, columns=col_list)
