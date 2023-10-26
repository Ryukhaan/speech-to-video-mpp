import functools
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from futils import flow_util
from models.base_blocks import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model=256, d_ff=256, dropout=0.1) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        self.relu = nn.ReLU()

    def forward(self, x):
        # (B x seq_len x d_model) --> (B x seq_len x d_ff) --> (B x seq_len x d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

class PhonemeEmbeddings(nn.Module):
    def __init__(self, d_model=256, phoneme_size=68):
        super(PhonemeEmbeddings, self).__init__()
        self.d_model = d_model
        self.phoneme_size = phoneme_size
        self.embeddings = nn.Embedding(phoneme_size, d_model)
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEconding(nn.Module):
    def __init__(self, d_model=256, seq_len=20, dropout=0.1):
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Seq_len x 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 1 x seq_len x d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # B x seq_len x d_model
        return self.dropout(x)

class ResidualConnection(nn.Module):

    def __init__(self, features=256, dropout=0.1) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model=256, h=8, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class PNet(nn.Module):

    def __init__(self, coef_nc=19, descriptor_nc=256, nlayer=3):
        super(PNet, self).__init__()

        self.nlayer = nlayer
        nonlineartiry = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coef_nc, descriptor_nc, kernel_size=7, padding=0, bias=True)
        )

        for i in range(nlayer):
            net = nn.Sequential(nonlineartiry,
                                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3)
                                )
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdativeAvgPool1d(1)
        self.output_nc = descriptor_nc
    def forward(self, phoneme_emb, positional_emb):
        out = self.first(phoneme)
        for i in range(self.nlayer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out