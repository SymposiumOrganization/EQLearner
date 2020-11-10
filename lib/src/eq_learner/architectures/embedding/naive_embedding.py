import torch 
import torch.nn as nn


class NaiveEmbedding(nn.Module):
    def __init__(self, 
                 input_dim, 
                 emb_dim, 
                 dropout, 
                 device,
                 max_length = 30):
        super().__init__()
        
        
        self.device = device
                        
        self.tok_linear = nn.Linear(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.dropout = nn.Dropout(dropout)
                        
    def forward(self, src):
        
        #src = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        #create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [0, 1, 2, 3, ..., src len - 1]
        
        #pos = [batch size, src len]
        
        #embed tokens and positions

        src = torch.unsqueeze(src,dim = 2)
        
        tok_embedded = ((self.tok_linear(src)))
        pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = pos_embedded = [batch size, src len, emb dim]
        
        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        
        return embedded