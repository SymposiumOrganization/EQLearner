import numpy as np
import random

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


def save_dataset(train_dataset=None, info_training= None, 
                test_dataset=None, info_testing=None,
                path="data/dataset"):
    assert info_training["isTraining"] == True
    assert info_testing["isTraining"] == False
    store_format = np.array((train_dataset,info_training, train_dataset, info_testing))
    torch.save(path,store_format)

def load_dataset(path="data/dataset.npy"):
    store_format = np.load(path, allow_pickle=True)
    train_dataset,info_training, train_dataset, info_testing = store_format
    assert info_training["isTraining"] == True
    assert info_testing["isTraining"] == False
    return train_dataset,info_training, train_dataset, info_testing
    
