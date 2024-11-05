import numpy as np
import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:, :x.shape[1]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x

def get_positional_encoding(d_model: int, max_len: int = 500):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # $10000^{\frac{2i}{d_{model}}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)
    # Add batch dimension
    encodings = encodings.unsqueeze(0).requires_grad_(False)
    return encodings


class TimeEncoder(nn.Module):
    def __init__(self, embed_dim, device):
        super(TimeEncoder, self).__init__()
        self.selection_layer = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = nn.Linear(64, embed_dim)
        self.device = device

    def forward(self, seq_time_step, mask):
        seq_time_step = torch.tensor(seq_time_step, device=self.device).unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = selection_feature.masked_fill(mask==True, -np.inf)
        return torch.softmax(selection_feature, 1)