import pytest
import torch
from eq_learner.evaluation import eqs_dataset_finder
import numpy as np

@pytest.fixture(scope="session")
def supports():
    return np.arange(0.1,3.1,0.1), np.arange(3,6.1,0.1)

@pytest.fixture(scope="session")
def training_dataset():   
    import torch
    from torch.utils.data import TensorDataset, DataLoader,Dataset 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_dataset = torch.load('lib/test/test_data/train_data_int_comp_50k.pt')
    return training_dataset

@pytest.fixture(scope="session")
def training_dataset_complete():   
    import torch
    from torch.utils.data import TensorDataset, DataLoader,Dataset 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_dataset = torch.load('data/1000_train.pt')
    return training_dataset

@pytest.fixture(scope="session")
def DatasetCreator():
    from sympy import sin, Symbol, log, exp
    from eq_learner.DatasetCreator import DatasetCreator
    x = Symbol('x')
    basis_functions = [x,sin,log,exp]
    DatasetCreator = DatasetCreator(basis_functions, constants_enabled=True)
    return DatasetCreator

@pytest.fixture(scope="session")
def device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

@pytest.fixture(scope="session")
def model(device):
    from eq_learner.architectures.old import decoder, encoder, seq2seq
    INPUT_DIM, OUTPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, DEC_LAYERS = 1, 20, 256, 512, 5, 5
    ENC_KERNEL_SIZE, DEC_KERNEL_SIZE, ENC_DROPOUT, DEC_DROPOUT = 3, 3, 0.25, 0.25
    TRG_PAD_IDX = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = encoder.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    dec = decoder.Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)
    model = seq2seq.Seq2Seq(enc, dec).to(device)
    model.load_state_dict(torch.load('data/benchmark_50k_small.pt'))
    return model
