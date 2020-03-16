# GitStar

## What is GitStar?
GitStar is a trained neural network that analyzes a given public GitHub repository and attempts to predict its star number. Implemented in pytroch, GitStar uses a large variety of GitHub heuristics to make star number estimates.


## Installation:

* Clone into desired directory (denoted as `.`)
```
git clone https://github.com/phelps-matthew/GitStar.git
```
* Enable virtual environment (optional)
```
python -m venv myvenv
. venv/bin/activate
```
* Install package
``` 
pip install .

# Or, to make package editable
pip install -e .
```
## Usage
The package is divided into two stages: data collection (extract, transfrom, load - ETL) and model training/optimization (models). Confer the file tree below.
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
###ETL (Extract, Transform, Load)
The features that the NN uses are obtained by querying GitHubs API. Specifically, 
