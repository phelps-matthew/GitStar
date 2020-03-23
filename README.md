# &#10031; GitStar 

## What is GitStar?
GitStar is a pipeline used to construct a neural network that analyzes a given public GitHub repository and attempts to predict its star number. Implemented in pytorch, GitStar uses a large variety of GitHub heuristics to make star number estimates.

## Table of Contents
  * [Installation](#installation)
  * [End-to-end Overview](#end-to-end-overview)
  * [Usage](#usage)
    + [ETL (Extract, Transform, Load)](#etl)
      - [GraphQL](#graphql)
      - [gqlquery](#gqlquery)
      - [gstransform](#gstransform)
      - [main_etl](#main-etl)
      - [config_sample](#config-sample)
    + [models](#models)
      - [datanorm](#datanorm)
      - [dataload](#dataload)
      - [deepfeedforward](#deepfeedforward)
      - [main_model](#main-model)
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
<my_dir> venv/bin/activate
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
The package is divided into two stages: data `ETL` (extract, transfrom, load) and model training/optimization (`models`). The file tree below highlights the location of the the public modules as well as applications of the modules (`main_etl` and `main_model`).
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

Application of `transform` is simple. Taking the last example, we could apply the transform as
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
In Progress...
#### config_sample
In Progress...
### models
In Progress
#### datanorm
In Progress...
#### dataload
In Progress...
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
