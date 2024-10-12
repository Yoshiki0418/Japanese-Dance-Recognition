import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import math
from src.models.common import *

class iTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_heads: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_norm: bool = False,
        d_ff: float = None,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.attn = MultiVariate_Attention(d_model=d_model, n_heads=n_heads)
        
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_enc, x_mark_enc=None, attn_mask=None) -> torch.Tensor:
        """
        x_enc: [Batch, Variate, Time]
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        _, N, _ = x_enc.shape # B N L

        """
        B: batch_size;    E: d_model; 
        L: seq_len;       S: pred_len;
        N: number of variate (tokens), can also includes covariates
        """

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        x = self.embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        new_x, attns = self.attn(x, x, x, attn_mask = attn_mask)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm2(x + y)
        
        return out

#--------------------------------
#      pre-LN構造
#--------------------------------
class iTransformer_preLN(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_heads: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_norm: bool = False,
        d_ff: float = None,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.attn = MultiVariate_Attention(d_model=d_model, n_heads=n_heads)
        
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_enc, x_mark_enc=None, attn_mask=None) -> torch.Tensor:
        """
        x_enc: [Batch, Variate, Time]
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        _, N, _ = x_enc.shape # B N L

        """
        B: batch_size;    E: d_model; 
        L: seq_len;       S: pred_len;
        N: number of variate (tokens), can also includes covariates
        """

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        x = self.embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

        x = self.norm0(x)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        new_x, attns = self.attn(x, x, x, attn_mask = attn_mask)

        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm2(x + y)
        
        return out
