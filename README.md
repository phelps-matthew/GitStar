# GitStar
Star predictor based on GitHub heuristics

To install:

* Clone into desired directory
* Enable virtual environement
```
python -m myvenv venv
. venv/bin/activate
```

* Within GitStar repository directory:
```
pip install .

# For editable
pip install -e .
```

.
├── gitstar
│   ├── ETL
│   │   ├── GQL_QUERIES
│   │   │   ├── QUERY
│   │   │   └── ...
│   │   ├── SQL_SCHEMA
│   │   │   ├── SCHEMA_v0
│   │   │   └── ...
│   │   ├── data
│   │   │   ├── ...
│   │   ├── logs
│   │   │   ├── ETL.log
│   │   │   ├── ETL_special.log
│   │   │   └── discover_params.log
│   │   ├── __init__.py
│   │   ├── discover_params.py
│   │   ├── gqlquery.py
│   │   ├── gstransform.py
│   │   └── main_etl.py
│   ├── models
│   │   ├── dataset
│   │   │   ├── gs_table_v2.csv
│   │   │   └── ...
│   │   ├── features
│   │   │   ├── ...
│   │   │   ├── transformed
│   │   │   │   └── full
│   │   │   └── plot_all_scaling.py
│   │   ├── __init__.py
│   │   ├── dataload.py
│   │   ├── feature_dist_notes.py
│   │   └── model.py
│   ├── __init__.py
│   ├── config.py
│   └── config_sample.py
├── tests
│   ├── __init__.py
│   ├── btest.py
│   ├── error_test.py
│   ├── gs_table_v2.csv
│   ├── loadsql.py
│   ├── main_etl_bak.py
│   ├── scratch.py
│   ├── simple_model1.py
│   └── sql_test.py
├── README.md
└── setup.py

18 directories, 310 files
```
