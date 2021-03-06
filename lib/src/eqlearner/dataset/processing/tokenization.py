from io import BytesIO
import tokenize
import numpy as np
from typing import Any, List
from nptyping import NDArray

# def default_map():
#     default_map = {"x": 1, "sin": 2, "exp": 3, "log": 4, "(": 5, ")": 6, "**": 7, "*":8, "+":9, "/": 10, "E": 11, "S": 12, "F": 13} 
#     max_val =  max(list(default_map.values()))
#     numbers = {str(n): max_val+n for n in range(1,10)}
#     return {**default_map, **numbers}

def default_map():
    default_map = {"x": 1, "sin": 2, "exp": 3, "log": 4, "(": 5, ")": 6, "**": 7, "*":8, "+":9, 
                    "/": 10, "E": 11, "S": 12, "F": 13, "sqrt":14, "-": 15} 
    max_val =  max(list(default_map.values()))
    numbers = {str(n): max_val+n for n in range(1,10)}
    default_map = {**default_map, **numbers}
    max_val =  max(list(default_map.values()))
    return default_map, max_val
    

def reverse_map(default_map,symbols):
    inv_map = {v: k for k, v in default_map[0].items()}
    inv_map[12] = ""
    inv_map[13] = ""
    if symbols:
        inv_sym = {v: k for k, v in symbols.items()}
        inv_map = {**inv_map, **inv_sym}
    return inv_map

def extract_terms(d: dict()) -> 'tuple(dict())':
    """Just a binding to tokenize module """
    separated_dict = {}
    #tokenized_dict = {}
    for key, li in d.items():
        separated_list = []
        #tokenized_list = []
        for element in li:
            separated_term = [elem.string for elem in list(tokenize.tokenize(BytesIO(str(element).encode('utf-8')).readline))]
            #tokenized_term = [elem.exact_type for elem in list(tokenize.tokenize(BytesIO(str(element).encode('utf-8')).readline))]
            separated_list.append(separated_term)
            #tokenized_list.append(tokenized_term)
        separated_dict[key] = separated_list
        #tokenized_dict[key] = tokenized_list
    return separated_dict#, tokenized_dict

def numberize_terms(d, mapping=None,symbols=None):
    tokenized_dict = {}
    if not mapping:
        mapping, max_val = default_map()
    if symbols:
        sym_mapping = {s: max_val+idx+1 for idx, s  in enumerate(symbols)}
        mapping = {**mapping, **sym_mapping}
    #else:
    #    sym_mapping = dict()
    # import pdb
    # pdb.set_trace()
    for key, li in d.items():
        expression_list = []
        for expression in li:
            term_list = []
            for term in expression:
                if term in mapping:
                    term_list.append(mapping[term])
            #tokenized_term = [elem.exact_type for elem in list(tokenize.tokenize(BytesIO(str(element).encode('utf-8')).readline))]
            #tokenized_list.append(tokenized_term)
            if term_list:
                expression_list.append(term_list)
        tokenized_dict[key] = expression_list
        #tokenized_dict[key] = tokenized_list
    return tokenized_dict, mapping#, tokenized_dict



def flatten_seq(d, mapping=None):
    if not mapping:
        mapping, _ = default_map()
    int_sum = mapping["+"]
    res = [mapping["S"]]
    for key, li in d.items():
        for expression in li:
            res.extend(expression)
            #tokenized_term = [elem.exact_type for elem in list(tokenize.tokenize(BytesIO(str(element).encode('utf-8')).readline))]
            #tokenized_list.append(tokenized_term)
            res.append(int_sum)
        #tokenized_dict[key] = tokenized_list
    if len(res) == 1:
        return [mapping["S"],mapping["F"] ]
    else: 
        res[-1] = mapping["F"]
        return res

def _sequence(d):
    tmp1 = extract_terms(d)
    tokenized_dict, mapping  = numberize_terms(tmp1)
    return flatten_seq(tokenized_dict, mapping=mapping)

def pipeline(d):
    """Just the extract_terms, numberize_terms and flatten_seq methods set up after the other plus a padding to the longest sequence (pad symbol 0)"""
    tmp = list(map(_sequence, d))
    pad = len(max(tmp, key=len))
    return np.array([i + [0]*(pad-len(i)) for i in tmp])

def count_number_of_symbols(source: List[NDArray[Any]]) -> NDArray[Any]:
    res = []
    for s in source:
        res.append(int(s.max()))
    return res



def apply_inverse_mapping(string, mapping=None):
    if not mapping:
            tmp = default_map()
            mapping = reverse_map(tmp)
    curr = [mapping[digit] for digit in string]
    return curr

def get_string(string, mapping=None,sym_mapping=None):
    tokenized_dict = {}
    if not mapping:
        tmp = default_map()
        mapping = reverse_map(tmp,symbols=sym_mapping)
    mapping_string = mapping.copy()
    mapping_string[12] = ""
    mapping_string[13] = ""
    curr = "".join([mapping_string[digit] for digit in string])
    if len(string) < 2:
        return RuntimeError
    if len(string) == 2:
        return 0 
    return curr