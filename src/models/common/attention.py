import torch
from torch import nn, einsum
from einops import rearrange
from math import sqrt
import numpy as np
from src.models.utils.masking import TriangularCausalMask


#--------------------------------
#   Multi-Head Self-Attention
#--------------------------------
class MultiHead_Self_Attention(nn.Module):
    """
    dim : int
        入力データの次元数．埋め込み次元数と一致する．
    heads : int
        ヘッドの数．
    dim_head : int
        各ヘッドのデータの次元数．
     dropout : float
        Dropoutの確率(default=0.)．
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # 線形変換
        q, k, v = [rearrange(t, "b (h d) l -> b h d l", h=self.heads) for t in qkv]
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b (h d) l", l=l)
        out = self.to_out(out)
        return out
    

#--------------------------------
#   Multi-variate Attention
#--------------------------------
class MultiVariate_Attention(nn.Module):
    """
    d_model: モデルの次元数
    n_heads: ヘッドの数
    d_keys: 各ヘッドのキーの次元数
    d_values: 各ヘッドのバリューの次元数
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys=None,
        d_values=None
    ) -> None:
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = False,
        scale: float = None,
        attention_dropout: float = 0.3,
        output_attention: bool = False,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
        
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Multivariate Correlations Map
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)