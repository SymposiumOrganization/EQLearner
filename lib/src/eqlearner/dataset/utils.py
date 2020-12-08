import numpy as np
import random
import torch

def forecast_len(dictionary):
    counter = 0
    for key, items in dictionary.terms:
        if key == "Single":
            counter = len(items)*1
        if key == "binomial":
            counter = len(items)*3
        if key == "N_terms": 
            counter = len(items)*3
        if key == "compositions":
            counter = len(items)*3
    return forecast_len

def count_element(x):
    res = x.count(0)
    return len(x) - res

def random_from_intervals(intervals): # intervals is a sequence of start,end tuples
    total_size = sum(end-start for start,end in intervals)
    n = random.uniform(0, total_size)
    if total_size > 0:
        for start, end in intervals:
            if n < end-start:
                return round(start + n,3)
            n -= end-start
    else:
        return 1

def are_there_nans(val):
    if np.isnan(val.sum()):
        return True
    else:
        return False

def is_max_not_within_threshold(val, threshold): # intervals is a sequence of start,end tuples
    if np.max(val)<threshold and np.min(val)>-threshold:
        return False
    else:
        return True

def is_already_generated(s, eqs_drop, eq_in):
    if s in eqs_drop or s in eq_in:
        return True
    else:
        return False

def is_too_long(seq, len_min, max_len): # intervals is a sequence of start,end tuples
    if 3 <= len(seq.squeeze()) <= max_len:
        return False
    else:
        return True

def save_dataset(train_dataset=None, info_training= None, 
                test_dataset=None, info_testing=None,
                path="data/dataset"):
    assert info_training["isTraining"] == True
    assert info_testing["isTraining"] == False
    store_format = np.array((train_dataset,info_training, train_dataset, info_testing),dtype="object")
    np.save(path,store_format)

def load_dataset(path="data/dataset.npy"):
    store_format = np.load(path, allow_pickle=True)
    train_dataset,info_training, train_dataset, info_testing = store_format
    assert info_training["isTraining"] == True
    assert info_testing["isTraining"] == False
    return train_dataset,info_training, train_dataset, info_testing
    
def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def constrained_partitions(num_elem=None, l=None, min_elem=None, max_elem=None):
    allowed = range(max_elem, min_elem-1, -1)
    def helper(n, k, t):
        if k == 0:
            if n == 0:
                yield np.array(t)
        elif k == 1:
            if n in allowed:
                yield np.array(t + (n,))
        elif min_elem * k <= n <= max_elem * k:
            for v in allowed:
                yield from helper(n - v, k - 1, t + (v,))

    return helper(num_elem, l, ())

def is_already_generated(s, eqs_drop, eq_in):
    if s in eqs_drop or s in eq_in:
        return True
    else:
        return False