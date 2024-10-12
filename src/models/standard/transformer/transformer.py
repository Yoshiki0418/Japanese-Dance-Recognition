import torch
import torch.nn as nn
from einops import rearrange
from src.models.common import *

class Transformer(nn.Module):
    def __init__(self, dim:int, length:int, dropout:float =0.1) -> None:
        super().__init__()
        self.pe = PositionalEncoding(dim=dim)
        self.norm = nn.LayerNorm((dim, length))
        self.attn = MultiHead_Attention(dim)
        self.ffn = FFN(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        time = torch.arange(X.shape[-1], device=X.device)
        time_emb = self.pe(time=time)
        time_emb = time_emb.transpose(0, 1).to(X.device)
        X1 = X + time_emb.unsqueeze(0)
        X_norm1 = self.norm(X1)
        X_norm1 = self.attn(X_norm1)
        X2 = X1 + self.dropout(X_norm1)
        X_norm2 = self.norm(X2)
        X_norm2 = rearrange(X_norm2, "b c l -> b l c")
        X_norm2 = self.ffn(X_norm2)
        X_norm2 = self.dropout(X_norm2)
        X_norm2 = rearrange(X_norm2, "b l c -> b c l")
        output = X2 + X_norm2
        return output