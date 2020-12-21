from . import eqgenerator
from .. import utils
from .equationtracker import EquationTracker
import sympy.abc
from sympy.utilities.lambdify import lambdify
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import warnings
import logging
from .equationstructure import EquationStructures
from ..processing import tokenization, tensordataset
import typing
import torch
import copy
from sympy.calculus.util import continuous_domain
from .eqgenerator import basis_and_symbol_joiner, constant_adder_binomial, polynomial_single, Binomial_single, N_single, Composition_single
from sympy import Symbol, S


class DatasetCreator():
    def __init__(self, raw_basis_functions: list(), max_linear_terms: int = 1, max_binomial_terms: int = 0,max_N_terms: int = 0,
                 max_compositions :int = 2, division_on: bool= False, random_terms=True, constants_enabled = True,
                 constant_intervals_ext=[(-10,1),(1,10)], constant_intervals_int = [(1,3)])-> tuple():
        """Create a dataset"""
        self.raw_basis_functions_to_deprecate = []
        for basis_function in raw_basis_functions:
            if type(basis_function) == sympy.core.symbol.Symbol:
                self.symbol = basis_function
            else:
                if type(basis_function) != sympy.core.function.FunctionClass:
                    raise(TypeError, "Basis functions must be func or symbol")
                self.raw_basis_functions_to_deprecate.append(basis_function)
        self.raw_basis_functions =  raw_basis_functions

        #THESE TWO WILL BE DEPRECATE
        
        self.basis_functions_to_deprecate = [basis_function(self.symbol) for basis_function in self.raw_basis_functions_to_deprecate] #Does not generalize in two dimensions
        self.basis_functions_to_deprecate.insert(0,self.symbol)

        self.max_linear_terms = max_linear_terms #i.e. sin(x), x. Must be equal or smaller than total number of basis
        self.max_binomial_terms = max_binomial_terms  #i.e. sin(x)*x, cos(x)*x
        self.max_N_terms = max_N_terms #i.e. polynomial of maximum two basis function up 6 order
        self.max_compositions = max_compositions 
        self.division_on = division_on
        self.scaler = MinMaxScaler()
        self.discard_eq = set()
        self.random_terms = random_terms
        if constants_enabled:
            self.interval_ext = constant_intervals_ext
            self.interval_int = constant_intervals_int
        else:
            self.interval_ext = [(1,1)]
            self.interval_int = [(1,1)]
    
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
        singles = []
        raw = []
        n_linear_terms, n_binomial_terms, n_N_terms, n_compositions = self.number_of_terms()
        total_combinations = EquationStructures(self.raw_basis_functions)
        tracker = EquationTracker(total_combinations.polynomial)
        for i in range(n_linear_terms):#range(n_linear_terms):
            singles, raw = polynomial_single(tracker,singles,raw, 
                                                                    self.symbol, constant_interval=self.interval_int)
        
        singles = self.order_fun(singles)
        signles_clean = [EquationStructures.polynomial_joiner(num, self.symbol) for num in raw]

        binomial = []
        priotity_list = []
        for i in range(0): #range(n_binomial_terms):
            binomial, priotity_list = Binomial_single(total_combinations.binomial,binomial,priotity_list, 
                                                                    self.symbol, constant_interval=self.interval_int)
        binomial = self.order_fun(binomial)
        binomial_clean = [constant_adder_binomial(total_combinations.binomial[num], self.symbol) for num in priotity_list]

        N_terms = []
        priotity_list = []
        personalized_basis_fun = curr = set(random.choices(self.basis_functions_to_deprecate,k=2))
        for i in range(n_N_terms):
            N_terms, priotity_list = N_single(personalized_basis_fun,N_terms,priotity_list,6)
        N_terms = self.order_fun(N_terms)
        N_terms_clean = copy.deepcopy(N_terms)

        compositions = []
        raw = []
        tracker = EquationTracker(total_combinations.compositions)
        for i in range(n_compositions):
            compositions, raw = Composition_single(tracker,compositions, raw,
                                                self.symbol,constant_interval=self.interval_ext)
        compositions = self.order_fun(compositions)
        compositions_clean = [EquationStructures.composition_joiner(num, self.symbol) for num in raw]
        
        # division = []
        # priotity_list = []
        # if self.division_on:
        #     division, priotity_list = Division_single(self.basis_functions_to_deprecate,3)
        # division = self.order_fun(division)
        # division_clean = copy.deepcopy(division)

        dictionary = {"Single": singles, "binomial": binomial, "N_terms": N_terms, "compositions": compositions} #, "division": division}
        res, dictionary = self.assembly_fun(dictionary)
        dictionary_cleaned = {"Single": signles_clean, "binomial": binomial_clean, "N_terms": N_terms_clean, "compositions": compositions_clean}# , "division": division_clean}
        return res, dictionary, dictionary_cleaned #singles, binomial_terms, N_terms, compositions, division)

    def assembly_fun(self,dictionary): # singles,binomial_terms,N_terms,compositions,divisions):
        res = 0
        for key, item in dictionary.items():
            for idx, elem in enumerate(dictionary[key]):
                c = utils.random_from_intervals(self.interval_ext) 
                dictionary[key][idx] = dictionary[key][idx] * c
                res = res + dictionary[key][idx]
        return res, dictionary
        # for single in enumerate(dictionary["Single"]):
        #     res = res + utils.random_from_intervals(self.interval)*single
        #     dictionary["Single"][]
        # for binomial_term in enumerate(dictionary["binomial"]):
        #     res = res + utils.random_from_intervals(self.interval)*binomial_term
        # for N_term in enumerate(dictionary["N_terms"]):
        #     res = res + utils.random_from_intervals(self.interval)*N_term 
        # for composition in enumerate(dictionary["compositions"]):
        #     res = res + utils.random_from_intervals(self.interval)*composition 
        # for division in enumerate(dictionary["division"]):
        #     res = res + utils.random_from_intervals(self.interval)*division 

    
    @staticmethod
    def handling_nan_evaluation(X, lambda_fun, X_noise=0, Y_noise=0):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            y = []
            for i in X:
                try:
                    y.append(lambda_fun(i+np.random.normal(scale=X_noise**(1/2))) + np.random.normal(scale=Y_noise**(1/2)))
                except RuntimeWarning:
                    y.append(np.nan)
        return y

    # @staticmethod
    # def create_basis_function(raw,symbol,noise_interval):
    #     res = [basis_function(self.symbol) for basis_function in self.raw_basis_functions] 

    def evaluate_function(self,X,sym_function, X_noise=0, Y_noise=0):
        x = self.symbol
        function = lambdify(x, sym_function)

        # All the warnings are related to function not correctly evaluated. So we catch them and set a nan.
        y = self.handling_nan_evaluation(X, function, X_noise=X_noise, Y_noise=Y_noise)
        y = np.array(y)
        return y
    
    def compute_admissible_domain(self,sym_function):
        """Not use much, as continuos_domain is superslow"""
        x = self.symbol
        res = continuous_domain(sym_function, self.symbol, S.Reals)
        return res

    def generate_batch(self,X, number_to_generate, X_noise=0, Y_noise=0, return_real_dict=False):
        """Generate batch ready for being deployed in the network
        Output is a np.array with X and y concatenated"""
        input_network = []
        output_network = []
        real_dict = []
        i = 0
        while i<number_to_generate:
            fun, dictionary, dictionary_cleaned = self.generate_fun()
            i += 1
            input_network.append(np.array([X, self.evaluate_function(X,fun, X_noise=X_noise, Y_noise=Y_noise)]))
            output_network.append(dictionary_cleaned)
            real_dict.append(dictionary)
        if return_real_dict:
            return input_network, output_network, real_dict
        else:
            return input_network, output_network

    def order_fun(self,group:list()) -> list():
        """Order elements of a group based on the basis_to_token method"""
        for i in group:
            if isinstance(i,list):
                return self.order_fun(i)
        return sorted(group, key=lambda x: str(x))

    def generate_set(self, support, num_equations, isTraining = True, threshold = 2000):
        x_train = []
        y_train = []
        cnt = 0
        skipped = 0
        cond = True
        while cond == True:
            numerical, dictionary =  self.generate_batch(support,1)
            if np.max(numerical[0][1])<threshold and np.min(numerical[0][1])>-threshold: 
                x_train.append(numerical[0][1])
                y_train.append(torch.Tensor((tokenization.pipeline(dictionary)[0])))
                #print(tokenization.get_string(tokenization.pipeline(dictionary)[0]))
                cnt+=1
            else:
                skipped += 1
            if cnt == num_equations:
                cond = False
        if isTraining:
            self.scaler.fit(np.array(x_train).T) 
            x_train = self.scaler.transform(np.array(x_train).T).T
        x_train = torch.Tensor(x_train)
        l = [len(y) for y in y_train]
        q = np.max(l)
        y_train_p = torch.zeros(len(y_train),q)
        for i,y in enumerate(y_train):
            y_train_p[i,:] = torch.cat([y,torch.zeros(q-y.shape[0])])
        dataset = tensordataset.TensorDataset(x_train,y_train_p.long())
        info = self.get_info()
        info["isTraining"] = isTraining
        info["num_equations"] = num_equations
        info["threshold"] = threshold
        info["Support"] = support
        return dataset, info
    
    def get_info(self):
        info = dict()
        info["raw_basis_functions"] = self.raw_basis_functions
        info["max_linear_terms"] = self.max_linear_terms 
        info["max_binomial_terms"] = self.max_binomial_terms 
        info["max_N_terms"] = self.max_N_terms
        info["max_compositions"] = self.max_compositions
        info["division_on"] = self.division_on
        info["random_terms"] = self.random_terms
        info["interval_ext"] = self.interval_ext
        info["interval_int"] = self.interval_int
        return info