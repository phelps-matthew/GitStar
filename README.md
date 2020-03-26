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
stargazers,
watchers,
readme_bytes,
licenseInfo
...
)
VALUES
(
?,
?,
?,
LEFT(CAST(? AS NVARCHAR(2000)), 2000),
...
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
Here we create the deep feedforward NN model and incorporate methods of backpropogation, training and validation. Performance diagnostics are printed and saved for later analysis or plotting.

The NN itself is constructed in the class `DFF` which inherits the torch.nn.Module. It provides the user a means to impose the number and dimensionality of the hidden layers as well as the choice of hidden layer activation function. An overridden forward method is provided that executes a foward pass. For example, going from 21 input features to 1 output with 3 hidden layers of size 16, 16, and 8 respectively, we may form the network model as
```python
import torch.nn.functional as F

h_layers = [16, 16, 8]
a_fn = F.relu
model = dff.DFF(D_in=21, D_hid=h_layers, D_out=1, a_fn=a_fn)
```
Once a model is constructed, we need methods to execute the training and validation. This is achieved through the `fit` function which loops over a number of epochs, calling `fit_epoch` on each iteration and printing current validation status. Once the training/validation is complete, performance data related to training loss, validation loss (scaled and unscaled), and other statistics are stored as csv's by use of the helper function `store_losses`. 

Representing the complete transfer learning process, the `fit` function is intialized by providing the number of epochs, model (DFF), loss function (torch.nn.Functional), optimizer (torch.optim), dataloaders and path strings for storing performance results.

Within each iteration of `fit`, function `fit_epoch` computes the batch loss for scaled and unscaled data via `loss_batch` and `inv_loss_batch`. If batch loss is called within the training phase, weights are backpropogated according to the provided optimization method. In a call to `inv_loss_batch`, the target inverse scaler is utilized to convert scaled loss to unscaled loss. After losses are computed, relevant performance stats are computed for scaled and unscaled data by `compute_stats` and `compute_inv_stats`.

Finally, some helper functions are provided. `set_logger` initializes the root logger for debugging, `hyper_str` generates conveinent descriptive strings for saving loss data and generating pngs (useful in hyperparameter searches), and `print_gpu_status` prints the availablity of Cuda GPU computation on the local machine.

With `fit` serving as the primary public function, here is an example of its usage. Confer `dataload` for constructing `train_dl` and `valid_dl`.
```python
import gitstar.models.deepfeedforward as dff

# Path to store performance diagnostic files
LOG_PATH = BASE_DIR / "logs"

# Set hyperparameters: batch size, learning rate, hidden layers, activ. fn
bs = 64
epochs = 1000
lr = 10 ** (-5)
h_layers = [16, 16, 8]
a_fn = F.relu

# Intialize model, optimization method, and loss function
model = dff.DFF(D_in=21, D_hid=h_layers, D_out=1, a_fn=a_fn)
opt = optim.Adam(model.parameters(), lr=lr)
loss_func = F.mse_loss

# Generate descriptive filename string for csv logs
model_str = dff.hyper_str(h_layers, lr, opt, a_fn, bs, epochs)

# Train, validate, and store loss
fit_args = (model, loss_func, opt, train_dl, valid_dl)
dff.fit(epochs, *fit_args, LOG_PATH, model_str)
```
#### main_model
Putting all the modules together, we may form a main application that implemements the deep feedforward model by constructing GitStar datasets dataloaders, executing model training and validation, and logging loss and validation diagnostics. To facilitate hardware acceleration, all torch tensors and model parameters are cast to Cuda device type for GPU computation.
```python
import torch
import torch.nn.functional as F
from torch import optim
import gitstar.models.deepfeedforward as dff
from gitstar.models.dataload import form_dataloaders, form_datasets

# Path Globals
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
LOG_PATH = BASE_DIR / "logs"
FILE = "gs_table_v2.csv"

# Enable GPU support and initialize logger
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dff.print_gpu_status()
def preprocess(x, y):
    return x.to(DEV), y.to(DEV)

# Set hyperparameters: batch size, learning rate, hidden layers, activ. fn
bs = 64
epochs = 10
lr = 10 ** (-5)
h_layers = [32, 16]
a_fn = F.relu

# Construct Dataset from file; form DataLoaders
train_ds, valid_ds = form_datasets(DATA_PATH / FILE)
train_dl, valid_dl = form_dataloaders(train_ds, valid_ds, bs, preprocess)

# Gather target inverse scaler fn
t_inv_scaler = train_ds.target_scaler["stargazers"]

# Intialize model (w/ GPU support), optimization method, and loss function
model = dff.DFF(D_in=21, D_hid=h_layers, D_out=1, a_fn=a_fn)
model.to(DEV)
opt = optim.Adam(model.parameters(), lr=lr)
loss_func = F.mse_loss
fit_args = (model, loss_func, opt, train_dl, valid_dl, t_inv_scaler)

# Generate descriptive filename string for csv logs
model_str = dff.hyper_str(h_layers, lr, opt, a_fn, bs, epochs)

# Train, validate, and store loss
dff.fit(epochs, *fit_args, LOG_PATH, model_str)
```
Comments:
* Default scaling methods are applied; must apply target inverse scaler for proper unscaled batch loss
* Learning rate, hidden layers, and activation function shown yielded best performance in optimization tests thus far
* Computing adapative learning rates for each parameter (e.g. Adam) highly suggested due to large feature-target variability
* Consider methods for parallelizing fit function for faster computation
## Results
The neural network was trained from a dataset spanning over 450k public repositories. For each repository, 21 different features were captured in order to predict its star number.

Features:
* Open Issues 
* Closed Issues
* Fork Count
* Pull Requests
* Commits
* Watchers
* Disk Usage (kb)
* README.md size (bytes)
* Releases 
* Projects
* Milestones
* Deployments
* Issue Labels (total number)
* Repository Topics (total number)
* Description Length
* Url (boolean)
* Liscense (boolean)
* Wiki (boolean)
* Issues Enabled (boolean)
* Created Date
* Last Push Date

Target:
* Stars
### Model Performance
The model itself is based on a multi-layer feedforward network with backpropogation. Here is a description of its architecture:

* Dimensions: 21 (in) x 32 x 16 x 1 (out) 
* Activation Function: ReLU
* Optimizer: Adam
  + Learning Rate: 10^(-5)
* Scaling Transforms
  + MinMaxScaler (sklearn.preprocessing)
  + "Quasi" Log10Transformer (see `datanorm`)
* Batch Size: 64
* Epochs: 1000
* Public Repository Filter Criteria
  + Stars > 1
  + Closed Issues > 0
  + Commits > 0
  + Readme Size (bytes) > 0
  + Watchers > 0
  + Fork Count > 0
  + Disk Usage (kb) > 0
  + Pull Requests > 0

To assess model performance, the mean squared error (MSE) and coefficient of determination were calculated with respect to the unormalized target predictions and output. The maximum R^2 reached is 0.79.
<img src="/presentation/unscaled_validation_err_50_FINAL.png"  width="450"><img src="/presentation/unscaled_validation_err_full_FINAL.png"  width="450">

### Feature Correlations
One method to gain insight into how features might affect the star number (like readme size or commit number), is to form a correlation matrix. Here the numbers represent linear correlation coefficients (pearson's). A value of 0 suggests the relationship between x and y is linearly uncorrelated, whereas as +/- 1 implies that the data is perfectly described by a linear relationship.
<p align="center"><img src="/presentation/correlation_matrix.png"  width="2000"></p>
To capture the essence of the actual dataset, and to further explore interesting correlations, here we plot the data directly. We perform a linear regression on inter-feature and feature-target variables, plot frequency distributions, and illustrate density with a hexbin plot. Pearson's correlation coefficient and p value statistics are overlayed. (Two sided p-value from [scipy.stats.pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) - roughly indicates probability of uncorrelated data producing the actuareadme dataset (or one that is even more correlated). Assumes normal distributions of features and or target).


*Note: The colorbar indicates number of repositories within a single hexbin of specified color*

First are the strongest correlations.
* Stars vs Fork Count : Strong correlation, as we would probably expect. Note skew in star distribution.
<p align="center"> <img src="/presentation/features/canonical_stargazers_forkCount.png"  width="600"> </p>
* Stars vs Watchers : Strong correlation. Fork counts are better indicator than watchers interestingly. Might think that forkers might forget to star as they are busy going about their business, but not true!
<p align="center"> <img src="/presentation/features/canonical_stargazers_watchers.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_stargazers_readme_bytes.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_stargazers_watchers.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_stargazers_diskUsage_kb.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_forkCount_watchers.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_closedissues_openissues.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_commitnum_pullRequests.png"  width="600"> </p>
<p align="center"> <img src="/presentation/features/canonical_closedissues_pullRequests.png"  width="600"> </p>


### Considerations
In Progress...
