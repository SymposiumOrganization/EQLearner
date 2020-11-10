import pytest
from eq_learner.evaluation import eqs_dataset_finder
from collections import namedtuple
import numpy as np
import torch

@pytest.fixture(scope="package")
def training_list(training_dataset):
    training_list = eqs_dataset_finder.training_list(training_dataset)
    return training_list

@pytest.fixture(scope="package")
def tokens(training_list):
    tokens = eqs_dataset_finder.dataset_without_padding(training_list[1])
    return tokens

@pytest.fixture(scope="package")
def extracted_text(training_list):
    seq = eqs_dataset_finder.list_all_equations_in_dataset(training_list[1])
    return seq
    
@pytest.fixture(scope="package")
def train():
    train = namedtuple('train', "tensors")
    train.tensors = list()
    train.tensors.append( torch.tensor([np.array([ 4.309691  ,  4.93819314,  5.75472605,  6.83242364,  8.27271756,
       10.21224355, 12.82483951, 16.31046521, 20.85900948, 26.57722573,
       33.37863887, 40.86473492, 48.26353605, 54.50838934, 58.49547549,
       59.44940996, 57.2221209 , 52.35312468, 45.85433261, 38.84778934,
       32.24732774, 26.60491566, 22.12299535, 18.75934632, 16.34640505,
       14.68094406, 13.57449923, 12.87331227, 12.46057734, 12.25092788]),
        np.array([-6.8025e+00, -4.6068e+00, -3.2634e+00, -2.2640e+00, -1.4508e+00,
            -7.5499e-01, -1.4062e-01,  4.1305e-01,  9.1910e-01,  1.3863e+00,
            1.8209e+00,  2.2274e+00,  2.6095e+00,  2.9701e+00,  3.3113e+00,
            3.6353e+00,  3.9435e+00,  4.2375e+00,  4.5185e+00,  4.7875e+00,
            5.0455e+00,  5.2933e+00,  5.5316e+00,  5.7612e+00,  5.9826e+00,
            6.1965e+00,  6.4031e+00,  6.6031e+00,  6.7969e+00,  6.9847e+00])]))
    train.tensors.append(torch.tensor([np.array([12,  4,  5,  3,  5, 16,  8,  1,  6,  9,  3,  5, 15,  8,  1,  6,  9, 14,
          6,  9,  3,  5,  2,  5,  1,  6,  7, 16,  9,  2,  5,  1,  6,  7, 15,  9,
          2,  5,  1,  6,  9, 14,  6, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0]),
        np.array([12,  4,  5,  1,  7, 16,  6,  9,  4,  5,  1,  7, 16,  9,  1,  7, 15,  9,
          1,  9, 14,  6, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0])]))
            
    return train

    
