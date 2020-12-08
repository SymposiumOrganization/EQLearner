import itertools
import sympy
from .. import utils
import numpy as np
from collections import defaultdict
from sympy import Symbol
from sympy import sin, Symbol, log, exp, zoo, Id, sqrt
from .basiccomponents import PolynomialCompoenents
from itertools import repeat
import random
import pandas as pd

class CompositeTerms():
    @staticmethod
    def binomial():
        return 0


class EquationStructures():
    def __init__(self, basis_functions, symbols, num_elements=None):
        self.basis_functions = basis_functions
        self.symbols = symbols
        self.pol = self._polynomial_table(basis_functions,symbols,num_elements)
        self.generated = {}
        #self.polynomial = self._polynomial_table_processing(pol, num_elements.max_linear_terms)
        #self.polynomial = self._polynomial_enumerator(basis_functions, symbols, order=poly_order)
        #single_elements = self._single_element_table(basis_functions, symbols, permutations=[1,0,0])
        #self.multiplication = self._multiplication(single_elements)
        #self.compositions = self._composition_table(basis_functions, symbols)
        
    @staticmethod
    def _multiplication(single_elements, multiplication=2):
        single_elements['merge'] = 0
        res = pd.merge(single_elements,single_elements, how='left',on='merge',suffixes=('_0', '_1'))
        if multiplication>2:
            for i in multiplication-2:
                raise NotImplementedError
        bool_cond = (res['symbol_0'] == res['symbol_1']) & (res['basis_function_0'] == res['basis_function_1'])
        candidates = res[~bool_cond].copy()
        return candidates

    @staticmethod
    def _single_element_table(basis_functions, symbols, permutations=[1,0,0]):
        iterables = [symbols, basis_functions]
        index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])
        dfs = list()
        perm = 0
        for symbol in symbols:
            for basis_fun in basis_functions:
                col3 = list(itertools.permutations(permutations))
                m = {hash(x):x for x in col3}
                col1 = list(repeat(symbol,len(col3)))
                col2 = list(repeat(basis_fun,len(col3)))
                try:
                    index = list(map(cls.polynomial_joiner, col3,col1))
                except:
                    print("hello")
                d = {'symbol': col1 , 'basis_function': col2, 'raw': col3}
                #index = cls.polynomial_joiner(tmp,symbol,const_interval_ext=[(1,1)], constant_interval_int=[(1,1)])
                dfs.append(pd.DataFrame(d))
        
        df = pd.concat(dfs,keys=index)
        return df

        pr_basis = []
        factors = []
        for i in itertools.combinations_with_replacement(basis_functions, multiplication):
            #if i[0]*i[1] != 1:
            pr_basis.append([i[0],i[1]])
        #indexes_lin = np.where([np.sum([basis_functions[i] == pr_basis[j] for i in range(len(basis_functions))]) for j in range(len(pr_basis))])[0]
        #Ugly list comprehension to get rid of "fake binomial": Idea change it to 
        #pr_basis = np.array([i for j, i in enumerate(pr_basis) if j not in indexes_lin])
        return pr_basis

    @staticmethod
    def _binomial(basis_functions):
        pr_basis = []
        factors = []
        for i in itertools.combinations_with_replacement(basis_functions, 2):
            #if i[0]*i[1] != 1:
            pr_basis.append([i[0],i[1]])
        #indexes_lin = np.where([np.sum([basis_functions[i] == pr_basis[j] for i in range(len(basis_functions))]) for j in range(len(pr_basis))])[0]
        #Ugly list comprehension to get rid of "fake binomial": Idea change it to 
        #pr_basis = np.array([i for j, i in enumerate(pr_basis) if j not in indexes_lin])
        return pr_basis
    
    @staticmethod
    def _constant_enumerator():
        return [0,1]
        
    @classmethod
    def _polynomial_table(cls,basis_functions,symbols,number_of_elements):
        total_possibilities = []
        for x in (number_of_elements.number_of_symbols,number_of_elements.number_of_terms):
            parts = utils.constrained_partitions(x,number_of_elements.number_of_symbols,max_elem=x-1, min_elem=1)
            for single_part in parts:
                fin = []
                for idx, elem in enumerate(single_part):
                    res = list(itertools.combinations(basis_functions,elem))
                    fin.append(list(itertools.product([symbols[idx]],res)))
                add = list(itertools.product(*fin))
                total_possibilities.extend(add)

        # import pdb
        # pdb.set_trace()
        # for x in range(1,len(symbols)+1):
        #     part = utils.partitions(x)
        #     for single_part in part:
        #         fin = []
        #         single_part = sorted(single_part,reverse=True)
        #         for idx, elem in enumerate(single_part):
        #             res = list(itertools.combinations(basis_functions,elem))
        #             fin.append(list(itertools.product([symbols[idx]],res)))
        #         add = list(itertools.product(*fin))
        #         total_possibilities.extend(add)
        d = {}
        for x in range(number_of_elements.number_of_terms):
            d["var_{}".format(x)] = []
            d["bas_{}".format(x)] = []
        d["type"] = []
        for curr_family in total_possibilities:
            p = 0
            for var in curr_family:
                symbol = list(repeat(var[0],len(var[1])))
                for i in range(len(var[1])):
                    d["var_{}".format(p)].append(symbol[i])
                    d["bas_{}".format(p)].append(var[1][i])
                    p += 1
            d["type"].append(p)
        #Set length
        max_l = max([len(x) for x in d.values()])
        for key in d.keys():
            if len(d[key]) < max_l:
                old = d[key]
                d[key] = list(repeat(0,(max_l - len(d[key]))))
                d[key].extend(old)
        df = pd.DataFrame(d)
        #df["type"] = df.apply(cls.return_type,axis=1) 
        
        other = cls.dataframeHelper(symbols)
        const = pd.DataFrame({"merge":[0,0], "constant":[0,1]})
        fin = df.merge(other,how="left",on="type")
        bool_cond = False
        for p in range(number_of_elements.number_of_terms):
            bool_cond =  bool_cond | ((fin["bas_{}".format(p)] == "exp") &  (fin["number_{}".format(p)] > 1))

        fin = fin[~bool_cond].copy() 
        fin.reset_index(inplace=True,drop=True)
        return fin

    @classmethod
    def dataframeHelper(cls,symbols):
        d = {}
        d["type"] = []
        for i in range(1,len(symbols)+1):
            d["number_{}".format(i-1)] = []
        for i in range(1,len(symbols)+1):
            res = list(utils.constrained_partitions(num_elem=5,l=i,min_elem=1,max_elem=5))
            bf_zip = [list(np.pad(x,(0, len(symbols)-int(i)))) for x in res]    
            res = list(zip(*bf_zip))
            d["type"].extend([i]*len(bf_zip))
            for i in range(1,len(symbols)+1):
                d["number_{}".format(i-1)].extend((res[i-1]))
        res = cls.count_element(d,symbols)
        df = pd.DataFrame(d)
        df["number_of_elements"] = res
        return df

    @classmethod
    def dataframeHelper2(cls,symbols):
        d = {}
        d["type"] = []
        for i in range(1,len(symbols)+1):
            itertools.product([0,1],repeat=5)
        for i in range(1,len(symbols)+1):
            res = list(utils.constrained_partitions(num_elem=5,l=i,min_elem=1,max_elem=5))
            bf_zip = [list(np.pad(x,(0, len(symbols)-int(i)))) for x in res]    
            res = list(zip(*bf_zip))
            d["type"].extend([i]*len(bf_zip))
            for i in range(1,len(symbols)+1):
                d["number_{}".format(i-1)].extend((res[i-1]))
        res = cls.count_element(d,symbols)
        df = pd.DataFrame(d)
        df["number_of_elements"] = res
        assert 0
        return df
        
    @classmethod
    def count_element(cls,di,symbols):
        combinations = []
        for j in range(len(di["type"])):
            curr = 1
            for i in range(1,len(symbols)+1):
                rem = cls.permutation_repetition(di["number_{}".format(i-1)][j]-1)
                if rem:   
                    curr =  curr*rem
            combinations.append(curr*2) #*2 to account for the constant
        return combinations

    @staticmethod
    def permutation_repetition(value):
        if value>0:
            return 2**value
        return 0

    @staticmethod
    def return_type(row):
        if row["bas_1"] == 0:
            return 1
        if row["bas_2"] == 0:
            return 2
        if row["bas_3"] == 0:
            return 3
        if row["bas_4"] == 0:
            return 4
        else: 
            return 5


    @classmethod
    def _polynomial_table_processing(cls, df, num_elements):
        df.drop(labels=0, inplace=True)
        df['merge'] = 0
        df = df.astype('category')
        res = df.copy()
        assert 0
        if num_elements >= 2:
            res = pd.merge(res,df, how='left',on='merge',suffixes=('_0', '_1'))
            bool_cond = (res['symbol_0'] == res['symbol_1']) & (res['basis_function_0'] == res['basis_function_1'])
            res = res[~bool_cond]
        if num_elements >= 3:
            res = pd.merge(res,df, how='left',on='merge',suffixes=('_0', '_1'))
            bool_cond = ((res['symbol_0'] == res['symbol_1']) & (res['basis_function_0'] == res['basis_function_1'])) | \
                        ((res['symbol_0'] == res['symbol']) & (res['basis_function_0'] == res['basis_function'])) | \
                        ((res['symbol_1'] == res['symbol']) & (res['basis_function_1'] == res['basis_function'])) 
            res = res[~bool_cond]
        if num_elements > 3:
            # We will need to keep the dataset separate in this case
            raise NotImplementedError
        res.drop(labels="merge",axis=1,inplace=True)
        return res



    @classmethod
    def polynomial_joiner(cls,candidate,cost,num_elements): 
        res = 0
        clean_res = []

        cols = (list(candidate.columns))
        for i in range(num_elements.number_of_terms):
            curr_bas = candidate["bas_" + str(i)].iat[0]
            if curr_bas == 0:
                i = i-1
                break
            curr_num = candidate["number_" + str(i)].iat[0]
            curr_var = candidate["var_" + str(i)].iat[0]
            
            curr_bas = cls.symbol_mapper(curr_bas)
            clean_res_tmp, res_tmp = cls.create_term(curr_var,curr_bas,curr_num,cost)
            res = res + res_tmp
            clean_res.extend(clean_res_tmp) #+ "+" + str(clean_res_tmp)
        
        #c = random.randint(0,1)
        #if c:
        #res = res + c*random.randint(cost[0][0]*100,cost[0][1]*100)/100
        clean_res = sorted(clean_res)
        clean_res = "+".join(clean_res)
        return clean_res, res

    @staticmethod
    def symbol_mapper(symbol):
        if symbol in ["log","exp","Id","sqrt","sin"]:
            return eval(symbol)
        elif symbol == "inv":
            return lambda x: 1/x
        else:
            raise KeyError("Basis function not in the list")

    @staticmethod
    def create_term(var,basis,number,constant_external):
        clean_res = []
        if number == 1:
            r = random.randint(constant_external[0][0]*100,constant_external[0][1]*100)/100
            clean_res.append(str(basis(Symbol(var))))
            return  clean_res,r*basis(Symbol(var))
        else:
            candidates = set(itertools.product([0,1],repeat=number))
            candidates.difference_update({tuple([0]*(number))})
            val = random.choice(list(candidates))
            res = 0
            for idx, p in enumerate(val):
                if p == 1:
                    r = random.randint(constant_external[0][0]*100,constant_external[0][1]*100)/100
                    #if clean_res != 0:
                    clean_res.append(str(basis(Symbol(var))**(idx+1))) #+ "+" + str(clean_res)
                    # else:
                    #     clean_res = str(basis(Symbol(var))**(idx+1))
                    res = r*basis(Symbol(var))**(idx+1) + res
            return clean_res, res

    @classmethod
    def composition_joiner(cls, candidate,symbol,const_interval_ext=[(1,1)], constant_interval_int=[(1,1)]): 
        res = 0
        res = cls.polynomial_joiner(candidate[1:],symbol,const_interval_ext, constant_interval_int)
        res = candidate[0](res)
        return res

    @classmethod
    def _composition_table(cls,basis_funtions, symbols):
        df = pd.DataFrame(columns=["main_symbol","main_basis_function","inner"])
        o = cls._single_element_table(basis_funtions, [symbols[0]], permutations=[1])
        o.drop(columns="symbol",inplace=True)
        o['merge'] = 0
        i = cls._polynomial_table(basis_funtions, symbols, order=3, drop_list=[(Id,0,0),(0,0,0)])
        i['merge'] = 0
        res = pd.merge(o,i, how='left',on='merge',suffixes=('_0', '_1'))
        res = pd.merge(res,i,how='left',on='merge',suffixes=('_1','_2'))
        res.drop(columns=["merge","raw_0"],inplace=True)
        res = res[res.iloc[:,0]!=basis_funtions[0]].copy()
        res.rename({"basis_function_0":"basis_function","basis_function":"basis_function_2","raw":"raw_2"},axis=1,inplace=True)
        res = res.astype('category')
        bool_cond = (res['symbol_1'] == res['symbol_2']) & (res['basis_function_1'] == res['basis_function_2'])
        res = res[~bool_cond].copy()
        bool_cond = (res["raw_1"] == hash((0,0,0))) | (res["raw_2"] == hash((0,0,0)))
        res = res[~bool_cond].copy()
        return res

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

    
    # @classmethod
    # def _polynomial_enumerator(cls, basis_funtion, symbols, order=6, drop_list=[], prefix = ()):
    #     #combinations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    #     pol = PolynomialCompoenents()
    #     for symbol in symbols:
    #         for basis_fun in basis_funtion:
    #             for x in set(itertools.product([basis_fun,0],repeat=order)):
    #                 for c in cls._constant_enumerator():
    #                     if all([i==0 for i in x]) or x in drop_list:
    #                         continue
    #                     pol.combinations[symbol][basis_fun][utils.count_element(x)][c].append([c]+list(x))
    #                     pol.symbol[symbol][basis_fun][utils.count_element(x)][c] = symbol
    #                 # if prefix:
    #             #     x = prefix + x 
    #     return pol