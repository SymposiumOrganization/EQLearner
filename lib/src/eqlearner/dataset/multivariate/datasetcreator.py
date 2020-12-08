from .. import utils
from .equationstructures import EquationStructures
from .equationstorer import EquationsStorer
import sympy.abc
from sympy.utilities.lambdify import lambdify
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import warnings
import logging
import pandas as pd
#from .equationstructure import EquationStructures
from ..processing import tokenization
from ..processing.tensordataset import TensorDataset

import typing
import torch
import copy
from sympy.calculus.util import continuous_domain
from .eqgeneratornew import basis_and_symbol_joiner, constant_adder_binomial, polynomial_single
from sympy import Symbol, S
from deprecated import deprecated
from .. import utils_dataclasses
from collections import Counter



# fh = logging.FileHandler('logs/data_creations.log')
# fh.setLevel(logging.WARNING)
class DatasetCreator():
    def __init__(self, raw_basis_functions: list(), symbols : list() = None, num_elements: utils_dataclasses =None,
                 constants: utils_dataclasses.Constants = None, max_num_equations=5):
        """Create a dataset"""
        self.raw_basis_functions =  raw_basis_functions
        self.symbols = symbols
        self.num_elements = num_elements
        self.constants = constants
        self.generated = Counter()
        self.total_combinations = EquationStructures(self.raw_basis_functions, self.symbols, num_elements=num_elements)
        self.max_num_equations = max_num_equations
    
    def number_of_terms(self):
        if self.random_terms:
            linear_terms = random.randint(0,self.max_linear_terms)
            binomial_terms = random.randint(0,self.max_binomial_terms)
            N_terms = random.randint(0,self.max_N_terms)
            compositions = random.randint(0,self.max_compositions)
            return linear_terms, binomial_terms, N_terms, compositions
        return self.max_linear_terms,self.max_binomial_terms, self.max_N_terms, self.max_compositions

    ## FIX ME: PRIORITY LIST MIGHT DEPRECATE
    def generate_fun(self):

        tracker = self.total_combinations.pol
        ##tracker["created_element"] = set()
        FLAG = 1
        while FLAG:
            fun_obj = polynomial_single(tracker, constant_interval=self.constants.constant_intervals_int, num_elements = self.num_elements)
            if fun_obj.elem_without_constant in self.generated:
                if self.generated[fun_obj.elem_without_constant] >= self.max_num_equations:
                    continue
                else:
                    self.generated[fun_obj.elem_without_constant] += 1
                    FLAG = 0
            else:
                self.generated[fun_obj.elem_without_constant] = 1
                FLAG = 0
        return fun_obj #singles, binomial_terms, N_terms, compositions, division)

    
    @staticmethod
    def handling_nan_evaluation(X, lambda_fun, X_noise=0, Y_noise=0):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            y = []
            X_t = list(zip(*X))
            for j in X_t:
                curr_v = list(j)
                for idx, _ in enumerate(curr_v):
                    curr_v[idx] = curr_v[idx]+np.random.normal(scale=X_noise**(1/2))
                try:
                    y.append(lambda_fun(*curr_v) + np.random.normal(scale=Y_noise**(1/2)))
                except RuntimeWarning:
                    y.append(np.nan)
        return y

    def evaluate_function(self,X,sym_function, X_noise=0, Y_noise=0):
        total_symbols = [Symbol(x) for x in self.symbols]
        function = lambdify(total_symbols, sym_function)

        # All the warnings are related to function not correctly evaluated. So we catch them and set a nan.
        y = self.handling_nan_evaluation(X, function, X_noise=X_noise, Y_noise=Y_noise)
        y = np.array(y)
        return y
    
    def compute_admissible_domain(self,sym_function):
        """Deprecated, as continuos_domain is superslow"""
        x = self.symbol
        res = continuous_domain(sym_function, self.symbol, S.Reals)
        return res

    def generate_batch(self,X, number_to_generate=0, X_noise=0, Y_noise=0, eq_storer=None , len_max=50, threshold = 2000):
        """Generate batch ready for being deployed in the network. A batch is guaranteed to have unique equations.
        Output is a np.array with X and y concatenated"""
        if not eq_storer:
            eq_storer = EquationsStorer(self)
        
        input_network = []
        output_network = []
        real_dict = []
        eq_in = set()
        i = 0
        while i<number_to_generate:
            fun_generator = self.generate_fun()
            tmp = tokenization.pipeline([{"Single": [fun_generator.elem_without_constant]}])
            seq = torch.Tensor(tmp)
            s = tokenization.get_string(tmp.squeeze())
            if utils.is_already_generated(s, eq_in,eq_storer.eqs_drop):
                continue

            if utils.is_too_long(seq, 3, len_max):
                eq_storer.add_eq_drop(s)
                continue

            inp = np.array(self.evaluate_function(X,fun_generator.elem_with_constant, X_noise=X_noise, Y_noise=Y_noise))
            if utils.are_there_nans(inp):
                eq_storer.add_eq_drop(s)
                continue

            if utils.is_max_not_within_threshold(inp,threshold):
                eq_storer.add_eq_drop(s)
                continue
                
            i += 1
            eq_storer.add_eq(eq_name=s, token=seq, eq_real=fun_generator.elem_with_constant, points=inp)
            eq_in.add(s)
            # input_network.append(inp)
            # output_network.append(seq)
            # real_dict.append(dictionary)
        return eq_storer
        # if return_additional_info:
        #     return input_network, output_network, real_dict, eq_in, eq_drop
        # else:
        #     return input_network, output_network
    def generate_fun_from_skeleton(self):
        raise NotImplementedError 

    def order_fun(self,group:list()) -> list():
        """Order elements of a group based on the basis_to_token method"""
        for i in group:
            if isinstance(i,list):
                return self.order_fun(i)
        return sorted(group, key=lambda x: str(x))

    @deprecated(version='0.2.0', reason="Deprecated in favor of new class structure")
    def generate_set(self, support, num_equations, len_max=50, threshold = 2000):
        """pad the sequence and return results"""
        eq_storer =  self.generate_batch(support,num_equations, len_max= 50, threshold=threshold)
        return eq_storer
    