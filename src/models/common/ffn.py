import torch
from torch import nn

#------------------------------
#     Feed Forward(MLP)
#------------------------------
class FFN(nn.Module):
    def __init__(self, dim, dff=240, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim*2)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(dim*2, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x