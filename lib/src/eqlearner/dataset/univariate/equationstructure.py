import itertools
import sympy
from .. import utils
import numpy as np
from collections import defaultdict
from sympy import Symbol

class CompositeTerms():
    @staticmethod
    def binomial():
        return 0


class EquationStructures():
    def __init__(self, basis_functions, poly_order = 6):
        self.basis_functions = basis_functions
        self.polynomial = self._polynomial_enumerator(basis_functions, order=poly_order)
        self.binomial = self.order_assigner(self._binomial(basis_functions))
        self.compositions = self._composition_enumerator(basis_functions)

    @staticmethod
    def _binomial(basis_functions):
        pr_basis = []
        factors = []
        for i in itertools.combinations_with_replacement(basis_functions, 2):
            pr_basis.append([i[0],i[1]])
        return pr_basis
    
    @staticmethod
    def _constant_enumerator():
        return [0,1]
        

    @classmethod
    def _polynomial_enumerator(cls, basis_funtion, order=6, drop_list=[], prefix = ()):
        combinations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for basis_fun in basis_funtion:
            for x in set(itertools.product([basis_fun,0],repeat=order)):
                for c in cls._constant_enumerator():
                    if all([i==0 for i in x]) or x in drop_list:
                        continue
                    combinations[basis_fun][utils.count_element(x)][c].append([c] + list(x))
                # if prefix:
                #     x = prefix + x 
                
        return combinations

    @staticmethod
    def polynomial_joiner(candidate,symbol,const_interval_ext=[(1,1)], constant_interval_int=[(1,1)]): 
        res = candidate[0]*utils.random_from_intervals(const_interval_ext)
        candidate_pol = candidate[1:]
        for idx, elem in enumerate(candidate_pol,1):
            if elem == 0:
                continue
            elif type(elem) == sympy.core.symbol.Symbol:
                    external = utils.random_from_intervals(const_interval_ext)
                    res = external*elem**(idx) + res
            elif type(elem) == sympy.core.function.FunctionClass:
                    interal = utils.random_from_intervals(constant_interval_int)
                    external = utils.random_from_intervals(const_interval_ext)
                    tmp = external*elem(interal*symbol)**(idx)
                    res = tmp + res
        return res

    @classmethod
    def composition_joiner(cls, candidate,symbol,const_interval_ext=[(1,1)], constant_interval_int=[(1,1)]): 
        res = 0
        res = cls.polynomial_joiner(candidate[1:],symbol,const_interval_ext, constant_interval_int)
        # for idx, elem in enumerate(candidate[1:],1):
        #     if elem == 0:
        #         continue
        #     elif type(elem) == sympy.core.symbol.Symbol:
        #             tmp = elem
        #             external = utils.random_from_intervals(const_interval_ext)
        #             res = tmp + external*elem**(idx)
        #     elif type(elem) == sympy.core.function.FunctionClass:
        #             interal = utils.random_from_intervals(constant_interval_int)
        #             external = utils.random_from_intervals(const_interval_ext)
        #             tmp = external*elem(interal*symbol)**(idx)
        #             res = tmp + res
        res = candidate[0](res)
        return res

    @classmethod
    def _composition_enumerator(cls,basis_funtions):
        combinations = defaultdict(lambda: defaultdict(list))
        
        for basis_fun in basis_funtions:
            if type(basis_fun) == sympy.core.symbol.Symbol:
                wacth_out = basis_fun
                continue
            #for x in set(itertools.product([basis_fun,0],repeat=order)):
            res = cls._polynomial_enumerator([x for x in basis_funtions if not x == basis_fun], order=3, drop_list=[(wacth_out,0,0)])
            cls.update_all(res,prefix = basis_fun)
            combinations[basis_fun] = res
            
        return combinations

    # def order_assigner_polynomial(self,list_of_terms):
    #     res = []
    #     for basis_fun in list_of_terms.keys():
    #         for degree in list_of_terms[basis_fun].keys():
    #             res.append(self.order_retriver(list_of_terms[basis_fun][degree]))

    @staticmethod
    def order_assigner(list_of_terms):
        return { k:i for k, i in  enumerate(list_of_terms)}

        
    @staticmethod
    def order_retriver(list_of_num):
        return { k:i for k, i in  enumerate(list_of_num)}

    @classmethod
    def visit_all(cls,d):
        counter = 0 
        for k, v in d.items():
            if isinstance(v, dict):
                counter = counter + cls.visit_all(v)
            else:
                counter = len(v) + counter
        return counter

    @classmethod
    def update_all(cls,d, prefix=None):
        for k, v in d.items():
            if isinstance(v, dict):
                cls.update_all(v, prefix)
            else:
                for l in v:
                    l.insert(0,prefix)
        return None