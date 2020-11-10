
from eq_learner.evaluation import statistics
import pytest
from eq_learner.evaluation import eqs_dataset_finder
import numpy as np
from collections import namedtuple

@pytest.fixture(scope="module")
def results_example():
    a =    [(0.06333800052900065, 0.21957464837866034, False),
        'log(exp(3*x)+exp(2*x)+1)+exp(sin(x)**3+sin(x)**2+sin(x)+1)',
        'sin(x)**6+sin(x)**5+sin(x)**4+sin(x)**3+1']
    b =    [(0.02290454875247764, 0.023228930813278086, True),
        'log(x**3)+log(x**3+x**2+x+1)',
        'log(x**3+x**2+x)']
    return [a,b]

@pytest.fixture(scope="module")
def tensor_pred_example():
    inp_gt = np.array([ 4.309691  ,  4.93819314,  5.75472605,  6.83242364,  8.27271756,
       10.21224355, 12.82483951, 16.31046521, 20.85900948, 26.57722573,
       33.37863887, 40.86473492, 48.26353605, 54.50838934, 58.49547549,
       59.44940996, 57.2221209 , 52.35312468, 45.85433261, 38.84778934,
       32.24732774, 26.60491566, 22.12299535, 18.75934632, 16.34640505,
       14.68094406, 13.57449923, 12.87331227, 12.46057734, 12.25092788])
    out_gt = np.array([12,  4,  5,  3,  5, 16,  8,  1,  6,  9,  3,  5, 15,  8,  1,  6,  9,
       14,  6,  9,  3,  5,  2,  5,  1,  6,  7, 16,  9,  2,  5,  1,  6,  7,
       15,  9,  2,  5,  1,  6,  9, 14,  6, 13,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0], dtype=np.int32) 
    out_prd = [12, 2, 5, 1, 6, 7, 19, 9, 2, 5, 1, 6, 7, 18, 9, 2, 5, 1, 6, 7, 17, 9, 2, 5, 1, 6, 7, 16, 9, 14, 13]
    return inp_gt,out_gt,out_prd





def test_count_occurences_extracted_text(extracted_text):
    seq = statistics.count_occurences(extracted_text)
    return seq

def test_count_occurences_tokens(tokens):
    tokens = statistics.count_number_occurences(tokens)

def test_is_correct_with_symbols(tokens):
    assert statistics.is_correct_with_symbols(tokens[0],tokens[0])
    assert not statistics.is_correct_with_symbols(tokens[0],tokens[1])

def test_rmse_calculator(tensor_pred_example, supports):
    inp_gt,out_gt,out_prd = tensor_pred_example
    out_gt = eqs_dataset_finder.remove_padding(out_gt)
    interpolation, extrapolation = supports
    rmse, rmse_extra = statistics.rmse_calculator(out_gt,out_prd,interpolation,extrapolation)
    assert rmse>10e-6
    assert rmse_extra>10e-6


@pytest.mark.skip(reason="I do not know how to load the model")
def test_evaluation_pipeline(train, supports, model):
    i = 0
    interpolation, extrapolation = supports
    rmse, rmse_extra = statistics.evaluation_pipeline(train.tensors[0][i].numpy(),train.tensors[1][i].numpy(),interpolation, extrapolation, model)
    assert rmse>10e-6
    assert rmse_extra>10e-6

@pytest.mark.skip(reason="I do not know how to load the model")
def test_dict_creator(train, supports, model):
    pass
     
def test_total_rmse_calculator(results_example):
    statistics.total_rmse_calculator(results_example)