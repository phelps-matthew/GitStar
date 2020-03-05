# GitStar
Star predictor based on GitHub heuristics.

Suggested installation method:

* Clone into desired directory
* Enable virtual environement
```
# Cloned dir is denoted as .
python -m venv myvenv
. venv/bin/activate
```

* Install package to virtual environement
```
pip install .

# If you would like to edit the package contents, use
pip install -e .
```
Public modules/API: 
```
.
├── gitstar
│   ├── ETL
│   │   ├── gqlquery.py
│   │   ├── gstransform.py
│   │   └── main_etl.py
│   ├── models
│   │   ├── dataload.py
│   │   ├── datanorm.py
│   │   ├── deepfeedfoward.py
│   │   └── main_model.py
│   └── config_sample.py
├── README.md
└── setup.py

20 directories, 739 files
```
