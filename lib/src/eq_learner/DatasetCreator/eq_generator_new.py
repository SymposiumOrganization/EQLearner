from sympy import *
import numpy as np
import itertools
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
import sympy
from sympy.utilities.lambdify import lambdastr
import gzip
import json
import os
import pdb
import copy
import graphviz
import random
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
import bisect
from . import utils
from .EquationStructure import EquationStructures
import logging
from typing import List, Tuple

#LOGGIN_DIR = 'logs/data_creations.log'
#print("Logging enabled: Log can found at", os.path.join(os.getcwd(),LOGGIN_DIR))
#logger = logging.getLogger('data_creation')
#logger.setLevel(logging.DEBUG)
#fh = logging.FileHandler(LOGGIN_DIR)
#fh.setLevel(logging.DEBUG)
#logger.addHandler(fh)

def Const_term():
    return 1, 1

def basis_and_symbol_joiner(basis_function,symbol, constant_interval=[(1,1)]): 
    if type(basis_function) == sympy.core.symbol.Symbol:
            symbol = basis_function
            return basis_function
    else:
        if type(basis_function) != sympy.core.function.FunctionClass:
            raise(TypeError, "Basis functions must be func or symbol")
        else: 
            try:
                c = utils.random_from_intervals(constant_interval)
                res = basis_function(symbol*c)
                return res
            except:
                print("Something wrong happended")
                pdb.set_trace()

def constant_adder_binomial(elements: List,symbol, constant_interval=[(1,1)]):
    assert len(elements) == 2
    with_constants = []
    for basis_function in elements:
        with_constants.append(basis_and_symbol_joiner(basis_function,symbol,constant_interval))
    return with_constants[0]*with_constants[1]
def order_assigner(list_of_terms):
        return { k:i for k, i in  enumerate(list_of_terms)}

def polynomial_single(tracker,expression: List, raw: List,symbol, constant_interval=[(1,1)]):
    # Add a linear function to the current set. It returns two:
    # One-List: the ordered set of basis functions
    # Second-List: number that represents the priority of the basis function
    # curr = set(priority_list)
    # candidate_keys = set(total_combinations.keys())
    # try:
    #     chosen = random.choice(list(candidate_keys-curr))
    # except:
    #     raise ValueError('The number of linear functions is bigger than the number of linear terms')
    # i = bisect.bisect(priority_list, chosen)
    # priority_list.append(chosen)
    chosen = tracker.get_equation(drop=0)
    raw.append(chosen)
    elem_with_constant = EquationStructures.polynomial_joiner(chosen, symbol, constant_interval, constant_interval)
    expression.append(elem_with_constant)
    return expression, raw

def Binomial_single(total_combinations,expression: List, priority_list, symbol,constant_interval=[(1,1)]):
    curr = set(priority_list)
    candidate_keys = set(total_combinations.keys())
    try:
        chosen = random.choice(list(candidate_keys-curr))
    except:
        raise ValueError('The number of linear functions is bigger than the number of linear terms')
    i = bisect.bisect(priority_list,chosen)
    priority_list.insert(i,chosen)
    elem_with_constant = constant_adder_binomial(total_combinations[chosen],symbol, constant_interval)
    expression.insert(i,elem_with_constant)
    return expression, priority_list

def N_single(basis_functions,fun_list,priority_list,n,begin=3):
    curr = set(priority_list) #watch out exp(x)^2 == exp(x^2)
    ordered = order_assigner(N_creator(basis_functions,n,begin))
    candidate_keys = set(ordered.keys())
    if list(candidate_keys-curr):
        chosen = random.choice(list(candidate_keys-curr))
        i = bisect.bisect(priority_list,chosen)
        priority_list.insert(i,chosen)
        fun_list.insert(i,ordered[chosen])
    return fun_list, priority_list

def N_creator(basis_functions:list(),n,begin):
    pr_basis = []
    factors = []
    for n in range(begin,n+1):
        if n == 0:
            pr_basis.append(1)  
            continue
        for i in itertools.combinations_with_replacement(basis_functions, n):
            pr_basis.append(reduce(operator.mul, i))
    return pr_basis

def Composite_creator(path_to_composite,expr):
    check_if_exist()
    return Composite_creator

def Join_expression(*args):
    total_coeff = sum([count_depth_dictionary([entry]) for entry in args])
    coeffs = np.random.random_sample(len(total_coeff))
    res = join_expression(args)
    expression = 0
    for idx ,curr_item in enumerate(myprint):
        expression = coeffs[idx]*curr_item
    return expression


def Composition_single(tracker,expression: List,raw: List,symbol,constant_interval=[1,1]):
    #ordered = order_assigner(raw_basis_functions).keys()
    #parent_keys = [key*1000 for key in ordered]
    #parent_key = random.choice(list(ordered)) #Can be also equal, not really imporant
    #child_basis = random.sample(list(set(basis_functions)-{raw_basis_functions[parent_key](symbol)}),2)
    #raw_basis_functions_set = set(raw_basis_functions)
    #raw_basis_functions_set.add(symbol)
    #child_basis = random.sample(list(raw_basis_functions_set-{raw_basis_functions[parent_key]}),2)
    chosen = tracker.get_equation(drop=2)
    raw.append(chosen)
    curr = EquationStructures.composition_joiner(chosen, symbol, constant_interval, constant_interval) 
    expression.append(curr)

    # if 0:
    #     tmp = Binomial_single(EquationStructures(child_basis).binomial,[],[],symbol, constant_interval= [(1,1)])  
    #     res  = tmp[0][0] + res
    #     pr = tmp[1][0] + pr
    #final_keys = parent_key*1000 + pr  
    #composition = raw_basis_functions[parent_key](res)
    #composition, final_keys = eliminate_infity_term(composition,final_keys)
    
    #compositions.append(composition)
    return expression, raw

def Division_single(basis_functions,n):
    #We said that we are just creating a single instane of division. Hence priority not really necessary. 
    candidate = random.choice(basis_functions)
    numerator = []
    denominator = []
    while numerator == denominator:
        numerator = []
        denominator = []
        poly = []
        priority = []
        for i in range(n):
            numerator, priority =  N_single([candidate],numerator, priority,6,begin=0)
            denominator, priority = N_single([candidate],denominator, priority, 6,begin=0)
    symbolic_numerator = expression_creator({"numerator": numerator})
    symbolic_denominator = expression_creator({"denominator": denominator})
    division = symbolic_numerator/symbolic_denominator 
    #logger.info("Create raw Division Term {}".format(str(division)))
    division, priority = eliminate_infity_term(division, 999)
    return [division], priority

def expression_creator(one_dictionary):
    total_coeff = 0
    expression = 0
    for key in one_dictionary.keys():
        total_coeff = len(one_dictionary[key]) + total_coeff
        #coeffs = np.random.random_sample(total_coeff)*0 + 1 #No constant
        for idx ,curr_item in enumerate(one_dictionary[key]):
            expression = curr_item + expression
            #expression = coeffs[idx]*curr_item + expression
    return expression

# def expression_joiner(*dictionaries):
#     final_expression = {dictionaries}
#     return final_expression
def eliminate_infity_term(expression, priority_list):
    if expression == zoo:
        return 0, 0
    else:
        return expression, priority_list

def eliminate_random_terms(expression,probability):
    for key in final_expression:
        to_drop = random.randint(0,10) > probability*10
        key.remove()
    return final_expression

def function_evaluator(support,expression):
    function = lambdify(x, result)
    res_dictionary['GT_Symbolic'] = expression
    res_dictionary['y'] = y
    res_dictionary['coeff'] = coeff
    res_dictionary['basis_fun'] = basis_functions
    res_dictionary['x'] = support['support']
    y = np.array([function(i) for i in support['support']])
    return res_dictionary



def Noise_adder_y(dataset: dict, var = 1.0):
    dataset = copy.deepcopy(dataset)
    for eq in dataset:
        eq['y'] = eq['y'] + np.random.normal(scale=np.sqrt(var), size=len(eq['y']))
    return dataset

def Noise_adder_x(dataset: dict, var = 0.025):
    dataset = copy.deepcopy(dataset)
    for eq in dataset:
        eq['x'] = eq['x'] + np.random.normal(scale=np.sqrt(var), size=len(eq['x']))
        result = np.dot(eq['basis_fun'],eq['coeff'])
        function = lambdify(x, result)
        eq['y'] = np.array([function(i) for i in eq['x']])
    return dataset

def count_depth_dictionary(d):
    return sum([count(v)+1 if isinstance(v, dict) else 1 for v in d.values()])

def entry_returner(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        yield ("{0} : {1}".format(k, v))


def save_generated_data(data,dir = r"C:\Users\lbg\OneDrive - CSEM S.A\Bureau\Pytorch\NEW_EQ_LEARN\Data"):
    name_key = list(data.keys())[0]
    #Save support points 
    path = os.path.join(dir,name_key)
    np.save(path,data)



