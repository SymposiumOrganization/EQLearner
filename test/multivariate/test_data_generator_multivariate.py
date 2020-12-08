import pytest
from eqlearner.dataset.multivariate.datasetcreator import DatasetCreator,utils_dataclasses
from eqlearner.dataset.processing import tokenization
import numpy as np
from sympy import sin, Symbol, log, exp, zoo, Id


num_elements = utils_dataclasses.NumElements(3,2)
num_elements = utils_dataclasses.NumElements(5,4)
no_costs = utils_dataclasses.Constants([(1,1)],[(1,1)])
consts = utils_dataclasses.Constants([(10,10)],[(5,5)])


@pytest.mark.parametrize("num_elements,constats_enabled", [(num_elements,no_costs), (num_elements,consts)], ids=(1,2))
def test_single_function_generation(intialize_values_multivariate,num_elements,constats_enabled):
    basis_functions, symbols = intialize_values_multivariate
    fun_generator = DatasetCreator(basis_functions,symbols,num_elements=num_elements,constants=constats_enabled)
    fun_obj =  fun_generator.generate_fun()
    
@pytest.mark.parametrize("num_elements,constats_enabled", [(num_elements,no_costs), (num_elements,consts)])
def test_generate_X_and_Y(intialize_values_multivariate,num_elements,constats_enabled):
    basis_functions, symbols = intialize_values_multivariate
    fun_generator = DatasetCreator(basis_functions,symbols,num_elements=num_elements,constants=constats_enabled)
    fun_obj =  fun_generator.generate_fun()
    support = []
    for x in symbols:
        support.append(np.arange(1,20))
    y = fun_generator.evaluate_function(support,fun_obj.elem_with_constant)

@pytest.mark.parametrize("num_elements,constats_enabled", [(num_elements,no_costs), (num_elements,consts)])
def test_generate_batch(intialize_values_multivariate,num_elements,constats_enabled):
    basis_functions, symbols = intialize_values_multivariate
    fun_generator = DatasetCreator(basis_functions,symbols,num_elements=num_elements,constants=constats_enabled)
    number_to_generate = 1
    support = []
    for x in symbols:
        support.append(np.arange(1,20))
    eq_storer = fun_generator.generate_batch(support, number_to_generate, X_noise=0.01, Y_noise=1)
    print(eq_storer)

@pytest.mark.parametrize("num_elements,constats_enabled", [(num_elements,no_costs), (num_elements,consts)])
def test_no_terms_are_zoo(intialize_values_multivariate,num_elements,constats_enabled):
    basis_functions, symbols = intialize_values_multivariate
    fun_generator = DatasetCreator(basis_functions,symbols,num_elements=num_elements,constants=constats_enabled)
    for i in range(100):
        x = Symbol('x')
        #basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
        fun  =  fun_generator.generate_fun()
        assert fun.elem_without_constant != zoo