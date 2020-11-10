import pytest
from sympy import sin, Symbol, log, exp, zoo
from eq_learner.DatasetCreator import DatasetCreator
from eq_learner.processing import tokenization
import numpy as np
import torch
import torch.nn as nn

# def test_encoder_with_standard_input():
#     pos = 
#     convconv_PoseNet.Encoder(4,8)


@pytest.fixture
def train_loader():
    from eq_learner.DatasetCreator import DatasetCreator, basis_functions
    import numpy as np
    from eq_learner.architectures.utils import dataset_loader
    generator = DatasetCreator(basis_functions,max_linear_terms=1, max_compositions=2, constants_enabled=True, random_terms=True)
    support = support = np.arange(0.1,3.1,0.1)
    train_dataset, info_training = generator.generate_set(support,3,isTraining=True)
    test_dataset, info_testing = generator.generate_set(support,1,isTraining=False)
    train_loader, valid_loader, test_loader, valid_idx, train_idx = dataset_loader(train_dataset,test_dataset)
    return train_loader

def test_cnn_architecture(train_loader):
    from eq_learner.architectures.cnn import Decoder, Encoder, Seq2Seq
    from eq_learner.architectures.embedding import NaiveEmbedding
    INPUT_DIM, OUTPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, DEC_LAYERS = 1, 30, 256, 512, 2, 2
    ENC_KERNEL_SIZE, DEC_KERNEL_SIZE, ENC_DROPOUT, DEC_DROPOUT = 3, 3, 0.25, 0.25
    TRG_PAD_IDX = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb = NaiveEmbedding(INPUT_DIM,EMB_DIM,ENC_DROPOUT,device)
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)
    model = Seq2Seq(emb, enc, dec).to(device)
    for i, batch in enumerate(train_loader):
        src = batch[0]
        trg = batch[1]
        output = model(src, trg)
    
    

    

@pytest.mark.skip(reason="Not implemented yet")
def test_playing_with_model():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=1, max_binomial_terms=1,max_compositions=1,max_N_terms=1,division_on=False)
    support = np.arange(-20,20,0.1)
    input_network, dictionaries =  fun_generator.generate_batch(support,20)
    output_network = torch.tensor(tokenization.pipeline(dictionaries)).long()
    input_network = torch.tensor(np.nan_to_num(input_network)).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIM = len(tokenization.default_map())
    EMB_DIM = 256
    HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = 10 # number of conv. blocks in encoder
    DEC_LAYERS = 10 # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = 3 # must be odd!
    DEC_KERNEL_SIZE = 3 # can be even or odd
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    TRG_PAD_IDX = 0
    fake_input_network = torch.cat((output_network,output_network),dim=1)
    # pose_net_encoder = convconv_PoseNet.PoseNetEncoder(input_network.shape[2],2)
    # tmp = pose_net_encoder(input_network)
    real_encoder = convconv_PoseNet.Encoder(input_dim=OUTPUT_DIM, emb_dim=int(OUTPUT_DIM**0.25), hid_dim=int((OUTPUT_DIM**0.25)*2), n_layers=ENC_LAYERS, kernel_size=ENC_KERNEL_SIZE, dropout=ENC_DROPOUT, device=device)
    conved, combined = real_encoder(fake_input_network)
    real_decoder = convconv_PoseNet.Decoder(output_dim=OUTPUT_DIM, emb_dim=int(OUTPUT_DIM**0.25), hid_dim=int((OUTPUT_DIM**0.25)*2), n_layers=ENC_LAYERS, kernel_size=ENC_KERNEL_SIZE, dropout=ENC_DROPOUT, trg_pad_idx=TRG_PAD_IDX, device=device)
    encoded_info = real_decoder(output_network,conved,combined)     


@pytest.mark.skip(reason="Not implemented yet")
def test_attention_with_point_net():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    fun_generator = DatasetCreator(basis_functions,max_linear_terms=1, max_binomial_terms=1,max_compositions=1,max_N_terms=1,division_on=False)
    support = np.arange(-20,20,0.1)
    input_network, dictionaries =  fun_generator.generate_batch(support,20)
    output_network = torch.tensor(tokenization.pipeline(dictionaries)).long()
    input_network = torch.tensor(np.nan_to_num(input_network)).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIM = len(tokenization.default_map())
    EMB_DIM = 256
    HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = 10 # number of conv. blocks in encoder
    DEC_LAYERS = 10 # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = 3 # must be odd!
    DEC_KERNEL_SIZE = 3 # can be even or odd
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    TRG_PAD_IDX = 0

    pose_net_encoder = convconv_PoseNet.PoseNetEncoder(input_network.shape[2],2)
    tmp = pose_net_encoder(input_network)
    real_decoder = convconv_PoseNet.Decoder(output_dim=OUTPUT_DIM, emb_dim=int(OUTPUT_DIM**0.25), hid_dim=int((OUTPUT_DIM**0.25)*2), n_layers=ENC_LAYERS, kernel_size=ENC_KERNEL_SIZE, dropout=ENC_DROPOUT, trg_pad_idx=TRG_PAD_IDX, device=device)
    encoded_info = real_decoder(output_network,tmp,tmp)