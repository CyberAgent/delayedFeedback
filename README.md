# Implementation of Delayed Feedback Model (DFM) for Conversion Rate Prediction

This method is introduced in [1].

We implemented it in Cython to compare our method[2] to DFM.

## Important scripts

- train_model.py

Use to learn DFM and get a average score.

- predicted_model/dfm.pyc

The implementation code for DFM written in Cython.

## Requirement

### Python Package
- pandas 0.23.0
- numpy 1.16.1
- sklearn 0.20.2
- Cython 0.28.2
- scipy 1.2.0

## Setup
Run `setup.sh`.


## Reference
Chapelle, Olivier. “Modeling delayed feedback in display advertising.” KDD '14 (2014).

Yasui Shota, Gota Morishita, Komei Fujita, Masashi Shibata "A Feedback Shift Correction in Display Advertising under the Delayed Feedback." WWW '20 (2020).
