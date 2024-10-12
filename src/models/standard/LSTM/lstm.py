import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(
            self, 
            in_dim,
            out_dim,
            p_drop: float = 0.3,
            num_layers: int = 3,
            bidirectional: bool = True, #双方向にするか否か
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = in_dim,
            hidden_size = out_dim,
            num_layers = num_layers,
            dropout = p_drop,
            batch_first = bidirectional
        )

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        期待する入力Xの形状：(batch_size, features, seq_len)
        seq_len : シーケンス長（時系列データの長さ）
        features : 各時点に関連する特徴の数
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        出力されるXの形状： (batch_size, out_dim, seq_len)
        out_dim: LSTMクラスを初期化した際に設定された次元数
        """
        X = X.transpose(1, 2)
        X, (hn, cn) = self.lstm(X)
        X = self.dropout(X)
        X = X.transpose(1, 2)
        return X