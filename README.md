# GitStar
Star predictor based on GitHub heuristics.

### Suggested installation method:

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

# Or, make package editable
pip install -e .
```
### Public modules/API: 
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
│   │   ├── deepfeedforward.py
│   │   └── main_model.py
│   └── config_sample.py
├── README.md
└── setup.py

20 directories, 739 files
```
