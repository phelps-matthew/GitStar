from pathlib import Path
import pandas as pd

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
    "Identity": ["hasUrl", "hasLicense", "hasWiki", "hasIssue"],
    "MinMax": ["repositoryTopics"],
    "Quantile_Gaussian": [
        "diskUsage_kb",
        "projects",
        "milestones",
        "issuelabels",
        "created",
        "updated",
    ],
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
    "Box_Cox": ["stargazers"],
}

df = GitStarDataset(DATA_PATH / SAMPLE_FILE, 0.1).data_frame
df = df[['watchers', 'forkCount']]

ct_features = make_column_transformer(
    (MinMaxScaler(), SCALER_COLS["MinMax"]),
    (
        QuantileTransformer(output_distribution="normal"),
        SCALER_COLS["Quantile_Gaussian"],
    ),
    (PowerTransformer(method="yeo-johnson"), SCALER_COLS["Yeo_Johnson"]),
    #    (PowerTransformer(method="box-cox"), SCALER_COLS["Box_Cox"]),
    remainder="passthrough",
)

test1 = ColumnTransformer(
    [
        ("yj_watchers", PowerTransformer(method="yeo-johnson"), ["watchers"]),
    ],
    remainder="passthrough",
)
test2 = ColumnTransformer(
    [
        (
            "yj_watchers_forkCount",
            PowerTransformer(method="yeo-johnson"),
            ["watchers", "forkCount"],
        )
    ],
    remainder="passthrough",
)
out1 = test1.fit_transform(df)
out2 = test2.fit_transform(df)
# fmt: off
import ipdb,os; ipdb.set_trace(context=5)  # noqa
# fmt: on
print(df)
new_array = preprocess.fit_transform(df)
print(new_array)
