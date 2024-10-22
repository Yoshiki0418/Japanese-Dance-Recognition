import torch
import torch.nn as nn
from einops import rearrange
from src.models.common import *

class Transformer_Encoder(nn.Module):
    def __init__(self, dim:int, heads:int, dim_head:int, dropout:float =0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHead_Self_Attention(dim, heads=heads, dim_head=dim_head)
        self.ffn = FFN(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X_norm1 = self.norm(X)
        X_norm1 = self.attn(X_norm1)
        X2 = X + self.dropout(X_norm1)
        X_norm2 = self.norm(X2)
        X_norm2 = rearrange(X_norm2, "b c l -> b l c")
        X_norm2 = self.ffn(X_norm2)
        X_norm2 = self.dropout(X_norm2)
        X_norm2 = rearrange(X_norm2, "b l c -> b c l")
        output = X2 + X_norm2
        return output