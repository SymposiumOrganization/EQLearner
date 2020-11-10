import torch 
import torch.nn as nn

class C_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        self.n_layers = n_layers
        
#        self.linear = nn.Linear(input_dim, emb_dim)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = emb_dim,  kernel_size = 3,padding = 1)
    
        self.relu = nn.ReLU()
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #Batch 2, Hidden sequence 4, Length state 6
        
        #src = [batch size,src len]
        
#        src = torch.unsqueeze(src.T,dim = 2)

        src = torch.unsqueeze(src,dim = 1)
    
        #src = [batch size,1,src len]
        #src = [src len, batch size,1]
        
        embedded = self.dropout(self.relu(self.conv(src)))

        embedded = embedded.permute(2,0,1)
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
