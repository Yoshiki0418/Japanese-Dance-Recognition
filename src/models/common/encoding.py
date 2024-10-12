import torch
from torch import nn
import math

#-------------------------------------
#       PositionalEncoding
#-------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        pe = math.log(10000) / (half_dim - 1) # pe => positional_encoding
        pe = torch.exp(torch.arange(half_dim, device=device) * -pe)
        pe = time[:, None] * pe[None, :]
        pe = torch.cat((pe.sin(), pe.cos()), dim=-1)
        if pe.size(1) < self.dim:
            zero_padding = torch.zeros(pe.size(0), 1, device=device)
            pe = torch.cat([pe, zero_padding], dim=1)
        return pe