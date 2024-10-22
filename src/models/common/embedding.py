import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

#----------------------------
#  Variate-based Embedding
#----------------------------
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in:int = 250, d_model:int = 516, dropout:float = 0.3):
        """
        各特徴（チャンネル）ごとにトークン化
        Paper link: https://arxiv.org/abs/2310.06625
        
        c_in: シーケン長
        d_model: 埋め込み次元
        dropout: ドロップアウト
        """
        super().__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        x: [Batch, Variate, Time]
        """
        if x_mark is None:
            # チャネルごとのトークン化
            x = self.value_embedding(x)
        else:
            # 共変量（タイムスタンプなど）と結合してトークン化
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        
        return self.dropout(x)
    
#----------------------------------------
#  Linear Projection of Flattened Patches
#----------------------------------------
class Patch_Embedding(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        dim: int,
        channels: int,
    ) -> None:
        super().__init__()

        image_height, image_width = self._tupler(image_size)
        patch_height, patch_width = self._tupler(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_embedding = nn.Sequential(
            # 平坦化(Flatten)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(img)
        return x

    def _tupler(t: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(tuple, t):
            return t
        else:
            return (t, t)

#--------------------------------
#     Positional Embedding
#--------------------------------
class Positional_Embedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int
    ) -> None:
        """
        Args:
            seq_len (int): Length of the input sequence (number of tokens).
            d_model (int): Dimension of each embedding vector for the tokens.
        """
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add position embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Input tensor with position embeddings added, shape [batch_size, seq_len, d_model].
        """
        return x + self.position_embedding
    
#-----------------------------------------
#   Extra learnable [class] embedding
#-----------------------------------------
class Extra_learnable_Embedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.emb = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    