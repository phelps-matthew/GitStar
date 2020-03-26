# &#10031; GitStar 

## What is GitStar?
GitStar is a pipeline used to construct a neural network that analyzes a given public GitHub repository and attempts to predict its star number. Implemented in PyTorch, GitStar uses a large variety of GitHub heuristics to make star number estimates.

## Table of Contents
  * [Installation](#installation)
  * [End-to-end Overview](#end-to-end-overview)
  * [Usage](#usage)
    + [ETL (Extract, Transform, Load)](#etl)
      - [GraphQL](#graphql)
      - [gqlquery](#gqlquery)
      - [gstransform](#gstransform)
      - [main_etl](#main_etl)
      - [config_sample](#config_sample)
    + [models](#models)
      - [datanorm](#datanorm)
      - [dataload](#dataload)
      - [deepfeedforward](#deepfeedforward)
      - [main_model](#main_model)
  * [Results](#results)
    + [Model Performance](#model-performance)
    + [Feature Correlations](#feature-correlations)
    + [Considerations](#considerations)

## Installation

* Clone into directory `<my_dir>`
```
git clone https://github.com/phelps-matthew/GitStar.git
```
* Enable virtual environment (optional)
```
python -m venv myvenv
<my_dir> myvenv/bin/activate
```
* Install package
``` 
pip install <my_dir>

# Or, to make package editable
pip install -e <my_dir>
```
## End-to-end Overview
Here is a representation of the processes executed in Gitstar, as we go from collecting the data to training the final network.

![Flowchart](/presentation/mermaid-diagram-svg2.svg)

## Usage
The package is divided into two stages: data extract, transform, load (`ETL`) and model training/optimization (`models`). The file tree below highlights the location of the the public API as well as suggested application of the modules (i.e., `main_etl` and `main_model`).
```
├── gitstar
│   ├── ETL
│   │   ├── gqlquery.py
│   │   ├── gstransform.py
│   │   └── main_etl.py
│   ├── models
│   │   ├── dataload.py
│   │   ├── datanorm.py
│   │   ├── deepfeedforward.py
│   │   └── main_model.py
│   └── config_sample.py
├── README.md
└── setup.py
```

### ETL
The GitHub heuristics that are to serve as inputs for the NN are obtained by querying GitHub's [API](https://developer.github.com/v4/), which is based on the GraphQL query language. 

#### GraphQL
Here is a sample of a graphQL query that uses specific criteria (stored in `myq`) to search through public repositories. This simple example returns, among other things, nameWithOwner, readme_size, and stargazers. The use of GraphQL features such as inline fragments, variables, and aliases prove to helpful (and at times are necessary, see [graphql.org/learn](https://graphql.org/learn/)). The full query of 20+ features used to train the NN is located in `gitstar/ETL/GQL_QUERIES/QUERY`.
```graphql
query GitStarSearch($myq: String!, $maxItems: Int, $cursor: String) {
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
  search(query: $myq, type: REPOSITORY, first: $maxItems, after: $cursor) {
    pageInfo {
      endCursor
      hasNextPage
    }
    repositoryCount
    edges {
      node {
        ... on Repository {
          nameWithOwner
          readme_size: object(expression: "master:README.md") {
            ... on Blob {
              byteSize
            }
          }
          stargazers {
            totalCount
          }
        }
      }
    }
  }
}
# ---------------
# Query Variables 
# ---------------
{
   "myq" : "archived:false mirror:false stars:>0 created: >=2015-01-01 pushed:>=2019-01-01 fork:true",
   "maxItems" : 5
}
```
#### gqlquery

Once a GraphQL query is formed, we must send it to the GitHub API enpoint and capture the response. This is implemented through HTTP POST queries from the `requests` package. The class `GraphQLQuery` serves as a constructor for implementingGraphQL specific requests through the `gql_response` method. It is simply a wrapper around `requests.post` to accept `query` and `variable` arguments.  For example,
```python
QUERY = """\
query {
  viewer {
    login
    name
  }
}
"""
ENDPOINT_URL = "https://api.github.com/graphql"
PAT = "<A PERSONAL ACCESS TOKEN>"
headers={"Authorization": "token {}".format(PAT)},

my_request = GraphQLQuery(headers, ENDPOINT_URL, QUERY)
my_response = my_request.gql_response()
```
Also of importance, `gql_response` handles GraphQL based response errors. If a timeout from GitHub's API is detected, the function will wait 60 seconds before reattempting. Based on the current state of GitHub's API, timeouts are inevitable.

The class `GitHubGraphQLQuery` simply intializes `GraphQLQuery` with the GitHub API endpoint and passes a supplied OAuthtoken into the header.

```python
QUERY = """\
query {
  viewer {
    login
    name
  }
}
"""
PAT = "<A PERSONAL ACCESS TOKEN>"

my_request = GraphQLQuery(PAT, QUERY)
my_response = my_request.gql_response()
```

Class `GitStarSearchQuery` now gets very specific and implements a GitStar specific query stored in `gistar/ETL/GQL_QUERIES/QUERY`. In addition, it imposes search criteria `["archived:false", "mirror:false", "fork:true"]`. In future versions, it may make sense to present a `GitHubSearchQuery` class for more flexibility in the query and query variables.
`GitStarSearchQuery` is initialized with dates in the form of `arrow` objects, which serve as an alternative to datetime.
The search query is designed around ranges of stars, push dates, and created dates.
```python
PAT = "<A PERSONAL ACCESS TOKEN>"
CREATED_START = arrow.get("2018-09-21")
CREATED_END = arrow.get("2019-12-31")
PUSH_START = arrow.get("2020-01-01")
MAXITEMS = 50
MINSTARS = 1
MAXSTARS = None

gitstar_response = gqlquery.GitStarSearchQuery(
    PAT,
    created_start=CREATED_START,
    created_end=CREATED_END,
    pushed_start=PUSH_START,
    maxitems=MAXITEMS,
    minstars=MINSTARS,
    maxstars=MAXSTARS,
)
```
Within GraphQL, the entire response corresponding to a query is partitioned into pages that must be accessed by iterating over multiple sub-queries until an end condition is met. This process, called pagination, is accomplished through the `gql_generator` method which returns a generator function. When iterated, the generator calls `self.gql_response()` and obtains the relevant pagination variables from the response.
Based on the pagination variables, API rate limit conditions are checked. If the limits are exceeded, an appropriate sleep time is calculated and executed.

You might be wondering, why a generator function? The answer is twofold. First, we need to pause the pagination in order to take the response and store it in a database. Secondly, the generator allows the user at the public level to control the execution flow based on any conditions involving the query response. 

There is one more important limitation of GitHubs API that is handled in the `gql_generator`. Though not stated anywhere in the GitHub API, for any search query that returns more than 1000 matching repositories, the pagination process will only return the first 1000. Such an event will throw an error belonging to class `RepoCountError`.
Handling this error is left up to the end-user. For example, in `main_etl` this error is used to slice the search space into smaller parititions based on push date.

Continuing the above example, the generator may be utilized as follows. 
```python
import json
gitstar_gen = gitstar_response.gql_generator()

while True:
    try:
        response = next(gistar_gen)
        print(json.dumps(response, indent=4)) # pretty printing
    except StopIteration:
        print("End of pagination")
	break
    except gqlquery.RepoCountError:
        print("Exceeded repo count")
	break
```

#### gstransform
Here we perform a round of data cleaning and transformation from the direct output of the GitStarSearchQuery generator (i.e. the GraphQL response). The `transform` function discards extraneous headers and utilizes uses `normalize` to flatten nested dictionaries corresponding to a single node (repo) into dictionaries of depth=1. 

Next, two key:value pairs are added to each node dictionary that represent the repository creation and last pushed to date in UTC integer timestamp (secs) format.

Application of `transform` is simple. Taking the last example, we may apply the transform as
```python
...

while True:
    try:
    	# Now we transform the response
        response = transform(next(gistar_gen))
        print(json.dumps(response, indent=4)) # pretty printing
    except StopIteration:
    	...
    ...
```
Additionally, the function `repocount` allows one to collect the total number of repositories returned from the raw GraphQL response.
#### main_etl
Putting the `gqlquery` and `gstransform` modules together, we finally construct an application to execute the ETL process. Within `main_etl`, this is achieved through looping over date ranges and iterating the `GitStarSearchQuery` generator.

Some important notes:
* The Azure MSSQL database chosen is accessed through the `pyodbc` module; different db connections may be necessary if database type differs; for pyodbc usage see [pyodbc](https://github.com/mkleehammer/pyodbc)
* The database configurations, GitHub PAT, and SQL insertion query are convienently grouped and stored within the module `config.py`. An example `config_sample.py` has been provided as a template; again, db connection implementation may vary
* Consider using online SQL transformation operations to further clean and process data. See `config_sample.py` for examples of this
* Consider use of `gqlquery.RepoCountError` to log GQL queries that yield repo counts exceeding 1000; alternatively, use conditional statement to redirect loop over push date slices (see function `special_etl()`)
* 50 items/(http request) was reasonable fetching param; higher items/request yields more http timeouts

Omitting detail on logging and pyodbc specifics, the main ETL process can be implemented as
```python
CREATED_START = arrow.get("2018-09-21")
CREATED_END = arrow.get("2019-12-31")
PUSH_START = arrow.get("2020-01-01")
MAXITEMS = 50
MINSTARS = 1
MAXSTARS = None

# Intialize db connection and GraphQL response generator
dbcnxn = dbconnection()
gql_gen = gql_generator(CREATED_START, PUSH_START)

# Loop until end date
delta = (CREATED_END - CREATED_START).total_seconds()
day = CREATED_START
while delta >= 0:
    try:
        # Iterate generator, transform response
        clean_data = transform(next(gql_gen))

        # Construct generator of dict values (db rows)
        value_list = (list(node.values()) for node in clean_data)

        # Load into db
        dbload(dbcnxn, value_list)

    # Catch pagination end condition or repo count overflow
    except (StopIteration, gqlquery.RepoCountError):
        # Increment over date range
        day = day.shift(days=+1)

        # Initialize new generator
        gql_gen = gql_generator(day, PUSH_START)

        # Update delta date range
        delta = (created_end - day).total_seconds()
```
The result is a pipeline that inserts 50 rows of respository features at a time, executing until search query reaches end condition.
#### config_sample
Here is a sub-sample of `config_sample.py` to illustrate an example of pyodbc config and query insertion. Online SQL transformation operations highly recommended. (the '?'s represent slots to pass variables from `value_list` above).
```python
# GitHub PERSONAL ACCESS TOKEN
PAT = "<PERSONAL ACCESS TOKEN>"

# Azure SQL database config
SERVER = "mydb.database.windows.net"
DATABASE = "my_db"
USERNAME = "username"
PASSWORD = "pass"

# INSERT SQL pyodbc query. When using LEFT, NVARCHARS are interpreted as ntext.
# Must cast to NVARCHAR for truncation
INSERT_QUERY = """\
INSERT INTO my_table_v1
(
createdAt,
updatedAt,
watchers,
diskUsage_kb,
readme_bytes,
...
isLocked,
createdAt_sec,
updatedAt_sec
)
VALUES
(
LEFT(CAST(? AS NVARCHAR(200)), 200),
?,
...
?,
LEFT(CAST(? AS NVARCHAR(200)), 200),
LEFT(CAST(? AS NVARCHAR(2000)), 2000),
?
);
"""
```
### models
After completing ETL process, the data is used to train the neural network. This process goes as follows:
1) Use scale functions to scale the feature and target variables (`datanorm`)
2) Form datasets and dataloaders that properly interface with network training and validation (`dataload`)
3) Design the NN architecture; construct training and validation sequences; create helper functions to store model performance diagnostics (`deepfeedforward`)
4) Execute training and validation; optimize model based on performance (`main_model`)
#### datanorm
While scaling the data is not strictly necessary, it was found that it immensely improved model convergence and stability; in fact, all models tested with unscaled data failed to converge toward any minimal mean square error.

The scaling of the data is enacted by the function `scale_cols` which scales selected columns, e.g. features or target, within the dataset. Specifically, a dictionary of column names and scale transformers is passed to `scale_cols`, allowing one to use different scalers on different features. The dataset should be cast into a DataFrame before passing to `scale_cols`. 

The scalers themselves are taken from the [sklearn preprocessing](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py) module. The sklearn scalers are capable of storing fit parameters and provide convienent inverse transformation methods.

To demonstrate the use of `scale_cols` and sklearn, we will transform the "created" and "updated" columns within the dataset using `MinMaxScaler()`.
```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load dataset into DataFrame from specified path
df = pd.read_csv(DATA_PATH / SAMPLE_FILE)

feature_scalers = {"created":MinMaxScaler(), "updated":MinMaxScaler()}

# scale_cols transforms the DataFrame in place
scale_cols(df, feature_scalers)
```
If one would like to use a custom scale transformation, the sklearn preprocessing `FunctionTransformer` readily handles this. To make a log10 scale transformer, for example
```python
my_log10 = lambda x: np.log10(x)
my_inv_log10 = lambda x: 10 ** x
myTransformer = FunctionTransformer(my_log10, my_inv_log10)
```
By this method, `myTransformer` follows the same call signature as all other sklearn preprocessing scalers and thus interfaces with `scale_cols` seamlessly.

The `datanorm` module also provides pre-configured feature and target scalers that have shown to yield good model performance. These are stored in the globals `FEATURE_SCALERS` and `TARGET_SCALER`, with many based on a quasi log10 scaling transformation (see `log10_map` and `log10_inv_map` for more details).
#### dataload
To facilitate use of the dataset within traning and validation of the model, the `dataload` module inherits the `Dataset` and `DataLoader` classes from `torch.utils.data`. These torch data classes provide conveinent methods for automatic batching, dataset iteration, and preprocessing. 

Class `GitStarDataset` provides a dataset object (inherited from torch.utils.data.Dataset) that provides additional functionality for passing scale transformers and storing the target inverse scale transformer. It is important to provide the inverse target scaler in order to convert scaled predictions (e.g. stars) to unscaled predictions. 

Class `WrappedDataLoader` inherits torch.utils.data.DataLoader and incorporates an additional, user defined preprocessing function. In the main application, we use this to cast torch.tensors into GPU device types.

In addition, this module provides functions for splitting a dataframe into train and validation sets (`split_df`), as well as a filtering function (`canonical_data`) to pass a dataset that follows the canonical GitStar criteria (e.g. constraints on features, such as commitnum > 1).

The above classes and functions are wrapped into the two main public functions `form_datasets` and `form_dataloaders`.
`form_datasets` reads a csv path, filters data to canonical form, splits the data into training/valdiation sets, and passes each set to construct instances of `GitStarDataset`. In addition, the function provides flexibility in imposing scale transformations from `GitStarDataset` kwargs.
`form_dataloaders` then takes in the `GitStarDataset`s and forms dataloaders according to the user specified batch size and preprocessing function.
Using the default scaling transforms, we may, for example, construct the datasets and dataloaders as
```python
# Form datasets from csv path
train_ds, valid_ds = form_datasets(DATA_PATH / FILE)

# Define GPU casting preprocess function
def preprocess(x, y):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return x.to(dev), y.to(dev)

# Form dataloaders based on batch size
batch_size = 64
train_dl, valid_dl = form_dataloaders(train_ds, valid_ds, bs=batch_size, preprocess=preprocess)
```



#### deepfeedforward
In Progress...
#### main_model
In Progress...

## Results
### Model Performance
In Progress...
### Feature Correlations
In Progress...
### Considerations
In Progress...
