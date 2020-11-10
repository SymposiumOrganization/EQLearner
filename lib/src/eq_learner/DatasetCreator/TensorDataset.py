from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch 
import numpy as np

class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index].cuda() for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
    
def dataset_loader(train_dataset,test_dataset, batch_size = 1024, valid_size = 0.20):
    num_train = len(train_dataset)
    num_test_h = len(test_dataset)
    indices = list(range(num_train))
    test_idx_h = list(range(num_test_h))
    np.random.shuffle(test_idx_h)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=0)
    test_loader_h = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader_h, valid_idx, train_idx
    



