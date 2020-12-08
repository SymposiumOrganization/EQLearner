# EQLearner
This repo contains the dataset generator library for creating symbolic regression datasets presented in the workshop paper "A Seq2Seq approach to Symbolic Regression" at NIPS 2020

Two types of datasets are available:
* Univariate datasets with combinations of polynomial and composition fuctions, with a fixed set of domain points (Used for our publication).
* Multivariate datasets of polynomial functions.

## Dataset features
We try to reduce to a minimum ambiguities in the dataset, by introducing two key features:
* There is one to one correspondence between mathematical expression and realization set. (i.e. if sin(x)*cos(x) is in the dataset, cos(x)*sin(x) would be not)
* Expressions are always in a precise order

## How to use 
Create a python virtual enviroment with ```python -m venv env``` and activate it with ```source env/bin/activate```
Install the library with ```pip install ./lib/``` or ```pip install -e ./lib/``` if you want the editable mode.
Eventually, run ```py.test test/``` for checking that everything went well
Then check the two jupyters in the how tos


