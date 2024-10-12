import torch
from torch import nn

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