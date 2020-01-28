# Delayed Feedback Problem

## Important scripts

- train_model.py

Use to learn a selected model and get a average score.
It includes the implementation of the estimation for FSIW.

- setting.json

In this json, the hyperparameters are written.

- predicted_model/dfm.pyc

The implementation code for DFM written in Cython

## Requirement
### Software except for Python
- [vowpal wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki)

### Python Package
- pandas 0.23.0
- numpy 1.16.1
- sklearn 0.20.2
- Cython 0.28.2
- scipy 1.2.0
- lightGBM 2.2.2

## Setup
- make required directories
`$ mkdir data model`

- compile Cython
`$ cd prediction_model`
`$ python setup.py build_ext --inplace`

- Run the script
`$ cd ..`
`$ python train_model.py`
