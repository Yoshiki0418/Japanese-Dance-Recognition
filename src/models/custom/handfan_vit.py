import torch
from torch import nn
from src.models.standard import Transformer_Encoder
from src.models.common import Patch_Embedding, Positional_Embedding, Extra_learnable_Embedding

class Handfan_ViT(nn.Module):
    def __init__(
        self, 
        dim: int,
        heads: int,
        channels: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        dim_head: int,
        dropout: float,
        emb_drop: float,
        num_classes: int,
    ) -> None:
        super().__init__()

        image_height, image_width = self._tupler(image_size)
        patch_height, patch_width = self._tupler(patch_size)  

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.patch_emb = Patch_Embedding(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            channels=channels,
        )
        self.cls_tokens = Extra_learnable_Embedding(dim=dim)
        self.pos_emb = Positional_Embedding(seq_len=num_patches+1, d_model=dim)

        self.transformer_encoder = Transformer_Encoder(
             dim = dim,
             heads = heads,
             dim_head = dim_head,
             dropout=dropout,
        )
        
        self.dropout = nn.Dropout(emb_drop)

        self.mlp_head = nn.Linear(dim, num_classes)


    def _tupler(self, t: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(t, tuple):
            return t
        else:
            return (t, t)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_emb(img)
        print(f"Patch Embedding output shape: {x.shape}")
        x = self.cls_tokens(x)
        print(f"cls Embedding output shape: {x.shape}")
        x = self.pos_emb(x)
        print(f"Positional Embedding output shape: {x.shape}")
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        # CLSトークンの抽出
        x = x[:, 0]

        out = self.mlp_head(x)
        return out

        
