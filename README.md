# &#10031; GitStar 

## What is GitStar?
GitStar is a trained neural network that analyzes a given public GitHub repository and attempts to predict its star number. Implemented in pytroch, GitStar uses a large variety of GitHub heuristics to make star number estimates.

Here is a representation of the elements contained in GitStar.

[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbmYxKGZlYXR1cmUgMSkgLS0-IEFcbmYyKGZlYXR1cmUgMikgLS0-IEFcbmYzKGZlYXR1cmUgLi4pIC0tPiBBXG5mNChmZWF1dHJlIG4pIC0tPiBBXG5BKEdyYXBoUUwgUXVlcnkpIC0tPiBCKEdyYXBoUUwgUmVzcG9uc2UpXG5CIC0tPiBDKGNsZWFuLCB0cmFuc2Zvcm0pXG5DIC0tPiBEKEF6dXJlIE1TU1FMKVxuRCAtLT4gfFNRTCB0cmFuc2Zvcm1hdGlvbnN8RFxuRCAtLT4gRShTY2FsaW5nIFRyYW5zZm9ybXMpXG5FIC0tPiBGKERhdGFzZXQgTG9hZGVycylcbkYgLS0-IEcoQXp1cmUgVk0pXG5HIC0tPiBHMihDb3JyZWxhdGlvbiBNYXRyaXgpXG5HMiAtLT4gRzMoUGxvdCBJbnRlcmVzdGluZyBSZWxhdGlvbnNoaXBzKVxuRyAtLT4gSChOZXVyYWwgTmV0KVxuSShUcmFpbmluZywgT3B0aW1pemF0aW9uLCBWYWxpZGF0aW9uKSAtLT4gSFxuSCAtLT4gSVxuSCAtLT4gSihBcHBsaWNhdGlvbilcblxuXHRcdFx0XHRcdCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggTFJcbmYxKGZlYXR1cmUgMSkgLS0-IEFcbmYyKGZlYXR1cmUgMikgLS0-IEFcbmYzKGZlYXR1cmUgLi4pIC0tPiBBXG5mNChmZWF1dHJlIG4pIC0tPiBBXG5BKEdyYXBoUUwgUXVlcnkpIC0tPiBCKEdyYXBoUUwgUmVzcG9uc2UpXG5CIC0tPiBDKGNsZWFuLCB0cmFuc2Zvcm0pXG5DIC0tPiBEKEF6dXJlIE1TU1FMKVxuRCAtLT4gfFNRTCB0cmFuc2Zvcm1hdGlvbnN8RFxuRCAtLT4gRShTY2FsaW5nIFRyYW5zZm9ybXMpXG5FIC0tPiBGKERhdGFzZXQgTG9hZGVycylcbkYgLS0-IEcoQXp1cmUgVk0pXG5HIC0tPiBHMihDb3JyZWxhdGlvbiBNYXRyaXgpXG5HMiAtLT4gRzMoUGxvdCBJbnRlcmVzdGluZyBSZWxhdGlvbnNoaXBzKVxuRyAtLT4gSChOZXVyYWwgTmV0KVxuSShUcmFpbmluZywgT3B0aW1pemF0aW9uLCBWYWxpZGF0aW9uKSAtLT4gSFxuSCAtLT4gSVxuSCAtLT4gSihBcHBsaWNhdGlvbilcblxuXHRcdFx0XHRcdCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)

## Table of Contents
  * [Installation](#installation)
  * [Usage](#usage)
    + [ETL (Extract, Transform, Load)](#etl--extract--transform--load-)
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
## Usage
The package is divided into two stages: data collection (extract, transfrom, load - ETL) and model training/optimization (models). The file tree below contains the public modules
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
### ETL (Extract, Transform, Load)
The GitHub heuristics that are to serve as inputs for the NN are obtained by querying GitHub's [API](https://developer.github.com/v4/), which is based on the GraphQL query language. 
#### GraphQL
Here is a subsample of a graphQL query that uses specific criteria (stored in `myq`) to search through public repositories and returns, among other things, nameWithOwner, readme_size, and stargazers. The use of inline fragments, variables, and aliases prove to helpful and even necessary for some features (see [graphql.org/learn](https://graphql.org/learn/)). The full query used to train the NN is located in `gitstar/ETL/GQL_QUERIES/QUERY`.
```graphql
query searchmp($myq: String!, $maxItems: Int, $cursor: String) {
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
adsf
#### gstransform
asdf
#### main_etl
asdf
#### config_sample
asdf
### models
asdf
#### datanorm
asdf
#### dataload
asdf
#### deepfeedforward
asdf
#### main_model
asdf

## Results
### Model Performance
asdf
### Feature Correlations
asdf
### Considerations
asdf
