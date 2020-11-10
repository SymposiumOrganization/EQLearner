import pytest
from sympy import sin, Symbol, log, exp, zoo
from eq_learner.DatasetCreator.DatasetCreator import DatasetCreator
from eq_learner.processing import tokenization
import numpy as np


def test_assert_exception_to_many_terms():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=5, random_terms=False)
    with pytest.raises(IndexError):
        for i in range(10):
            fun_generator.generate_fun()

def test_assert_assertion():
    x = Symbol
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    with pytest.raises(TypeError):
        fun_generator = DatasetCreator(basis_functions)
    
@pytest.mark.parametrize("constats_enabled", [False, True])
def test_single_function_generation(constats_enabled):
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4, constants_enabled=constats_enabled,max_compositions=4)
    string, dictionary, dictionary_cleaned =  fun_generator.generate_fun()

def test_generate_X_and_Y():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, _, _ =  fun_generator.generate_fun()
    support = np.arange(1,20)
    y = fun_generator.evaluate_function(support,string)

@pytest.mark.parametrize("X_noise,Y_noise", [(0,0), (0.1,0.1), (1,1), (10,10)])
def test_generate_batch(X_noise,Y_noise):
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=1, max_compositions=2)
    number_to_generate = 1
    support = np.arange(-20,20,0.1)
    inp, out, real_dict = fun_generator.generate_batch(support, number_to_generate, X_noise=X_noise, Y_noise=Y_noise, return_real_dict=True)
    print(out)
    print(real_dict)

def test_no_terms_are_zoo():
    for i in range(100):
        x = Symbol('x')
        basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
        fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
        string, dictionary, _ =  fun_generator.generate_fun()
        for terms in dictionary.values():
            for term in terms:
                assert term != zoo

@pytest.mark.skip(reason="Not implemented yet")
def test_domain_calculator():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, dictionary, _ =  fun_generator.generate_fun()
    fun_generator.compute_admissible_domain(string)
    

@pytest.mark.parametrize("constats_enabled", [False, True])
def test_luca_dataset_creator(constats_enabled):
    support = np.arange(0.1,3.1,0.1)
    x = Symbol('x')
    basis_functions = [x,sin,log,exp]
    creator_object = DatasetCreator(basis_functions,max_linear_terms=2, max_binomial_terms=1, max_compositions=1,
                    max_N_terms=0,division_on=False, constants_enabled=constats_enabled, random_terms=False)
    res = creator_object.generate_set(support,25)
    print(res)

@pytest.mark.parametrize("constats_enabled", [False, True])
def test_compare_speed_with_constants(constats_enabled):
    for i in range(100):
        support = np.arange(0.1,3.1,0.1)
        x = Symbol('x')
        basis_functions = [x,sin,log,exp]
        creator_object = DatasetCreator(basis_functions,max_linear_terms=1, max_binomial_terms=1, max_compositions=1,
                        max_N_terms=1,division_on=False, constants_enabled=constats_enabled, random_terms=True)
        numerical, dictionary = creator_object.generate_batch(support,1)

# @pytest.mark.parametrize("constats_enabled", [True])
# def test_luca_dataset_creator_II(constats_enabled):
#     for i in range(10):
#         support = np.arange(0.1,3.1,0.1)
#         x = Symbol('x')
#         basis_functions = [x,sin,log,exp]
#         creator_object = DatasetCreator(basis_functions,max_linear_terms=1, max_binomial_terms=1, max_compositions=0,
#                         max_N_terms=0,division_on=False, constants_enabled=constats_enabled, random_terms=True, constant_intervals_ext=[(-10,1),(1,10)], constant_intervals_int = [(1,3)])
#         res = creator_object.generate_set(support,500)