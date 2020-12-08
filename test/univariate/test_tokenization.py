import pytest
from sympy import sin, Symbol, log, exp, zoo
from eqlearner.dataset.univariate.datasetcreator import DatasetCreator
from eqlearner.dataset.processing import tokenization
from sympy.utilities.lambdify import lambdify
import sympy
import pdb
import numpy as np


def test_separator():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, dictionary, _ =  fun_generator.generate_fun()
    separated_dict = tokenization.extract_terms(dictionary)
    print(separated_dict['Single'])

def test_numberizer():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, dictionary, _ =  fun_generator.generate_fun()
    separated_dict = tokenization.extract_terms(dictionary)
    numberized_dict = tokenization.numberize_terms(separated_dict)
    print(numberized_dict)
    print("hello")

def test_joiner():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, dictionary, _ =  fun_generator.generate_fun()
    separated_dict = tokenization.extract_terms(dictionary)
    numberized_dict = tokenization.numberize_terms(separated_dict)
    print(numberized_dict)
    print("hello")


def test_final():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
    string, dictionary, _ =  fun_generator.generate_fun()
    separated_dict = tokenization.extract_terms(dictionary)
    numberized_dict, mapping = tokenization.numberize_terms(separated_dict)
    final_seq = tokenization.flatten_seq(numberized_dict, mapping=mapping)
    print(final_seq)
    print("hello")

def test_check_for_plus():
    support = np.arange(0.1,3.1,0.1)
    for i in range(100):
        x = Symbol('x')
        basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
        fun_generator = DatasetCreator(basis_functions,max_linear_terms=1, max_binomial_terms=1,max_compositions=1,max_N_terms=0,division_on=False)
        string, dictionary  = fun_generator.generate_batch(support,1)
        res = tokenization.get_string(tokenization.pipeline(dictionary)[0])
        if res:
            assert res[-1] != "+"

def test_check_for_none():
    support = np.arange(0.1,3.1,0.1)
    counter = 0
    for i in range(100):
        x = Symbol('x')
        basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
        fun_generator = DatasetCreator(basis_functions,max_linear_terms=1, max_binomial_terms=1,max_compositions=1,max_N_terms=0,division_on=False)
        string, dictionary = fun_generator.generate_batch(support,1)
        res = tokenization.get_string(tokenization.pipeline(dictionary)[0])
        if not res:
            counter = counter + 1
    assert counter < 100
def test_batch():
    try:
        x = Symbol('x')
        basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
        fun_generator = DatasetCreator(basis_functions,max_linear_terms=4)
        support = np.arange(-3,3,0.1)
        string, dictionary = fun_generator.generate_batch(support,20)
        result = tokenization.pipeline(dictionary)
    except:
        pdb.set_trace()

@pytest.mark.parametrize("constats_enabled", [False, True])
def test_get_back(constats_enabled):
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    for i in range(10):
        fun_generator = DatasetCreator(basis_functions,max_linear_terms=4,constants_enabled=constats_enabled)
        simpy_output, dictionary, dictionary_clean =  fun_generator.generate_fun()
        separated_dict = tokenization.extract_terms(dictionary_clean)
        numberized_dict, mapping = tokenization.numberize_terms(separated_dict)
        final_seq = tokenization.flatten_seq(numberized_dict,mapping=mapping)
        ori_fun = lambdify(x, simpy_output, 'numpy')
        try:
            get_back = lambdify(x, tokenization.get_string(final_seq), 'numpy')
        except:
            pdb.set_trace()
        input_x = np.arange(-3,3,0.1)
        if not constats_enabled:
            ori_y = np.nan_to_num(fun_generator.handling_nan_evaluation(input_x,ori_fun))
            new_y = np.nan_to_num(fun_generator.handling_nan_evaluation(input_x,get_back))
            assert np.all(ori_y == new_y)
