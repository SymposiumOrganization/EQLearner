import random
from torch.utils.data import TensorDataset
import torch 
from ..processing import tokenization
import pdb 
import numpy as np

def dictionary_creator(training_list):
    res = dict()
    points = training_list[0]
    seq = training_list[1]
    for i in range(len(seq)):
        s = tuple(seq[i])
        res[s] = points[i]
    return res

def generate_training_set_from_dataset_creation(dictionary_creator, DatasetCreator, support, number=1):
    nums = []
    tokens = []
    dict_tok = set()
    l = len(list(dictionary_creator.keys())[0])
    while len(tokens)<number:
        res = DatasetCreator.generate_batch(support, 1)
        token = tokenization.pipeline(res[1])
        num = res[0][0][1]
        if is_to_drop(num, token, len_max = l):
            continue
        token = pad_sequence(token[0],l)
        token = tuple(token)
        if token in dict_tok:
            continue
        if token in dictionary_creator:
            nums.append(num)
            tokens.append(token)
            dict_tok.add(token)
    return tensor_dataset(nums,tokens)  
    
def generate_val_set_from_dataset_creation(dictionary_creator, DatasetCreator, support, number=1):
    nums = []
    tokens = []
    dict_tok = set()
    l = len(list(dictionary_creator.keys())[0])
    while len(tokens)<number:
        res = DatasetCreator.generate_batch(support, 1)
        token = tokenization.pipeline(res[1])
        num = res[0][0][1]
        if is_to_drop(num, token, len_max = l):
            continue
        token = pad_sequence(token[0],l)
        token = tuple(token)
        if token in dict_tok:
            continue
        if not token in dictionary_creator:
            nums.append(num)
            tokens.append(token)
            dict_tok.add(token)
    return tensor_dataset(nums,tokens)   

def pad_sequence(token,l):
    tmp = np.zeros((l,), dtype=np.int32)
    tmp[:len(token)] = token
    return tmp

def is_to_drop(num, token, len_max):
    if len(token[0]) > len_max:
        return True
    elif np.isnan(num).any():
        return True
    elif np.max(num) > 2000 or np.min(num) < -2000:
        return True
    return False

def unique_sets_of_tokens(training_seqs,k=10):
    res = random.sample(training_seqs.tolist(),k=k)
    return res

def training_set_creation(dictionary_creator,k=10):
    inp = []
    out = []
    candidates = random.sample(dictionary_creator.keys(),k=k)
    for i in candidates:
        inp.append(dictionary_creator[i])
        out.append(i)

    return tensor_dataset(inp,out)

def tensor_dataset(inp, out):
    inp = torch.tensor(inp)
    out = torch.tensor(out)
    tensor_dataset = TensorDataset(inp,out)
    return tensor_dataset
