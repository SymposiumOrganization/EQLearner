import pytest
from sympy import sin, Symbol, log, exp, zoo
from eqlearner.dataset.univariate.datasetcreator import DatasetCreator
from eqlearner.dataset.univariate.datasetcreator  import utils
import numpy as np

@pytest.mark.skip(reason="Not implemented yet")
def test_forecast_len():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, dictionary =  fun_generator.generate_fun()
    utils.forecast_len(dictionary)


@pytest.mark.parametrize("intervals", [[(-10,-1),(1,5)],[(1,1),(1,1)]])
def test_random_interval(intervals):
    counter = 0
    total_runs =10000
    for i in range(total_runs):
        res = utils.random_from_intervals(intervals)
        if res >= 0:
            counter = counter + 1
    if sum(end-start for start,end in intervals) > 0:
        assert counter/total_runs > (intervals[1][1]-intervals[1][0])/sum(end-start for start,end in intervals) - 0.3
        assert counter/total_runs < (intervals[1][1]-intervals[1][0])/sum(end-start for start,end in intervals) + 0.3


@pytest.mark.parametrize("intervals", [[(1,3)]])
def test_random_interval_single(intervals):
    counter = 0
    total_runs =10000
    for i in range(total_runs):
        res = utils.random_from_intervals(intervals)
        if intervals[0][0] <= res <= intervals[0][1]:
            counter = counter + 1
    if sum(end-start for start,end in intervals) > 0:
        for interal in intervals:
            assert counter/total_runs == 1

# @pytest.fixture
# def datasets():
#     from eqlearner.dataset.univariate.datasetcreator import DatasetCreator
#     import numpy as np
#     x = Symbol('x')
#     basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
#     generator = DatasetCreator(basis_functions,max_linear_terms=2, max_compositions=2, constants_enabled=True, random_terms=True)
#     support = support = np.arange(0.1,3.1,0.1)
#     train_dataset, info_training = generator.generate_set(support,5,isTraining=True)
#     test_dataset, info_testing = generator.generate_set(support,1,isTraining=False)
#     return train_dataset, info_training, test_dataset, info_testing

# def test_save_and_dataset(datasets):
#     utils.save_dataset(*datasets)
#     fin = utils.load_dataset()
#     for k in fin[1].keys():
#         if type(datasets[1][k]) == np.ndarray:
#             assert all(datasets[1][k] == fin[1][k])
#         else:
#             assert datasets[1][k] == fin[1][k]
#     for k in fin[3].keys():
#         if type(datasets[3][k]) == np.ndarray:
#             assert all(datasets[3][k] == fin[3][k])
#         else: 
#             assert datasets[3][k] == fin[3][k]
   
