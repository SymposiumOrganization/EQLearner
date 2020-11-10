from ..processing import tokenization
from sklearn.preprocessing import MinMaxScaler
from sympy import Symbol, lambdify
from eq_learner.processing.tokenization import get_string
import numpy as np
import torch

def list_all_equations_in_dataset(dataset):
    strings = list(map(tokenization.get_string,dataset_without_padding(dataset)))
    return strings

def dataset_without_padding(dataset):
    dataset_without_padding = list(map(remove_padding, dataset))
    return dataset_without_padding

def training_list(train_data):
    return train_data.tensors[0].squeeze().numpy(), train_data.tensors[1].squeeze().numpy()

def remove_padding(l):
    return l[l!=0]

def normalize(num_values, scaler=None):
    if scaler:
        xxx_n = scaler.transform(num_values.reshape(-1,1))
        return xxx_n.squeeze(), scaler
    else:
        scaler = MinMaxScaler()
        xxx_n = scaler.fit_transform(num_values.reshape(-1,1))
    return xxx_n.squeeze(), scaler


def compute_Y(tokens,interpolation, extrapolation, normalization=True):
    x = Symbol('x')
    string = get_string(tokens)
    function_gt = lambdify(x, string)
    y = np.array([function_gt(i) for i in interpolation])
    y_extra = np.array([function_gt(i) for i in extrapolation])
    if normalize:
        y, scaler = normalize(y)
        y_extra, _ = normalize(y_extra,scaler)
    return y, y_extra

def convert_dataset_to_neural_network(dataset, interpolation, extrapolation, normalize=True):
    tokens_numpy = [x.numpy() for x in dataset.tensors[1]] 
    tokens = dataset_without_padding(tokens_numpy)
    res = [ compute_Y(x, interpolation, extrapolation) for x in tokens]
    train, test = list(zip(*res))
    return torch.tensor(train), torch.tensor(test)
