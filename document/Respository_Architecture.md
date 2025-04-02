# Respository Architecture

We you clone the project from github, your file structure will be looked as follows:

```
4420
├── README.md
├── document
│   ├── 4420_proposal.pdf
│   ├── 4420_report.pdf
│   ├── Environment_Setup.md
│   ├── Respository_Architecture.md (You are reading this)
│   └── assets
└── src
    ├── data_loader
    │   ├── __init__.py
    │   └── data_utils.py
    ├── datasets
    ├── graphs
        ├── dif_dim_graphs
        ├── dif_size_graphs
    ├── results
        ├── Final_Results
    ├── scripts
    ├── test_cases
    └── tree
        ├── BruteForce.py
        ├── GeometricDataStructure.py
        ├── KDTree.py
        ├── __init__.py
        └── utils.py

```

### `document`

Contain the documentation of project including instructions on how to set up the environment, our proposal and our report.

### `src/dataloader/data_utils.py`

Load and preprocess benchmark datasets for experiment

### `datasets`

To run experiments, datasets are needed to download in this folder. All datasets used in this project can be found and downloaded from [here](https://github.com/DBAIWangGroup/nns_benchmark/tree/master/data).

### `scripts`

Directory contains `python `scripts to run experiments

### `test_cases`

Directory contains simple test cases to verify our implementation

### `tree`

Implementation of K-D tree, Ball\* Tree and Bruteforce approach can be found in `KDTree.py`,`BallTree.py` , `BruteForce.py` respectively.
