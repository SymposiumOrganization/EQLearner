import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset

def loader():
    return torch.load('lib/test/test_data/train_data_int_comp_50k.pt')
    