# Original implementation can be found in: https://github.com/reml-lab/mTAN/blob/main/src/models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, N))
       
    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))
    

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e4)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)
    

class enc_mtan_classif(nn.Module):
 
    def __init__(self, input_dim, query, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., num_classes=2, return_hidden=False, encoder_type='GRU', device='cuda'):
        super(enc_mtan_classif, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.num_classes = num_classes
        self.return_hidden = return_hidden
        self.encoder_type = encoder_type
        self.max_sequence_length = 365 
  
        self.classifier = nn.Linear(nhidden, self.num_classes)        

        if self.encoder_type == "GRU":
            self.enc = nn.GRU(nhidden, nhidden)
        if self.encoder_type == "RNN":
            self.enc = nn.RNN(nhidden, nhidden)
        if self.encoder_type == "LSTM":
            self.enc = nn.LSTM(nhidden, nhidden)
        
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1, device=self.device)
            self.linear = nn.Linear(1, 1, device=self.device)
    
    
    def learn_time_embedding(self, tt):
        # tt = tt.to(self.device)
        tt = tt.to(device=self.device, dtype=torch.float32)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device=self.device)
    
        # absolute (365 will bin measurements into months (33 days))
        position = 365.*pos.unsqueeze(2).to(device=self.device)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model)).to(device=self.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
         
    def forward(self, x, time_steps):
        x = x.to(device=self.device, dtype=torch.float16)
        time_steps = time_steps.to(device=self.device, dtype=torch.float32) 
        
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
            
        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)

        if self.encoder_type == "LSTM":
            _, (out, _) = self.enc(out)
        else: 
            _, out = self.enc(out)
        out = out.squeeze(0)
        
        if self.return_hidden:
            return self.classifier(out), out
        else:
            return self.classifier(out)


class enc_mtan_classif_transformer(nn.Module):
 
    def __init__(self, input_dim, query, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., num_classes=2, return_hidden=False, device='cuda'):
        super(enc_mtan_classif_transformer, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.num_classes = num_classes
        self.return_hidden = return_hidden
        
        self.classifier = nn.Linear(nhidden, self.num_classes)

        self.enc_layer = nn.TransformerEncoderLayer(d_model=self.nhidden, 
                                                    nhead=num_heads,
                                                    dim_feedforward=self.nhidden,
                                                    dropout=0.1)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=1)
        self.pool_1D = nn.AdaptiveAvgPool1d(1)
        
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1, device=self.device)
            self.linear = nn.Linear(1, 1, device=self.device)
        
        self.causal_mask = torch.triu(torch.full((self.query.shape[0], self.query.shape[0]), float('-inf'), device=device), diagonal=1)

    def learn_time_embedding(self, tt):
        tt = tt.to(device=self.device, dtype=torch.float32)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device=self.device)
        position = 365.*pos.unsqueeze(2).to(device=self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model)).to(device=self.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def time_embedding_tape(self, pos, d_model):       
        # Create positional encoding tensor
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device=self.device)
    
        # absolute (365 will bin measurements into months (33 days))
        # Create position tensor and unsqueeze to add dimension
        position = 365.*pos.unsqueeze(2).to(device=self.device)
        
        # Calculate div_term using log of frequency
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(np.log(self.freq) / d_model)).to(device=self.device)
        
        # Apply scaling factor (d_model/sequence_length) as in reference
        scale = d_model / (self.max_sequence_length * 4)
        
        # Calculate sine and cosine components with scaling
        pe[:, :, 0::2] = torch.sin((position * div_term) * scale)
        pe[:, :, 1::2] = torch.cos((position * div_term) * scale)
        
        return pe
       
    def forward(self, x, time_steps):
        x = x.to(device=self.device, dtype=torch.float16)
        time_steps = time_steps.to(device=self.device, dtype=torch.float32) 
        
        # time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
            
        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        
        # out = self.enc(out, mask=self.causal_mask)
        out = self.enc(out)
        out = out.permute(1, 2, 0) # [len, b, dim] > [b, dim, [len, b, dim]
        out = self.pool_1D(out)
        out = out.squeeze(-1)
        
        if self.return_hidden:
            return self.classifier(out), out
        else:
            return self.classifier(out)
        
