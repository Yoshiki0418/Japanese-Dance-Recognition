import torch
import torch.nn as nn

class AdaNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        """
        AdaNormクラスの初期化
        :param normalized_shape: 正規化を行う次元の形状
        :param epsilon: 数値安定性のための微小値
        """
        super(AdaNorm, self).__init__()
        self.epsilon = epsilon
        
        # スケールパラメータとバイアスパラメータ
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # γ: スケーリング
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # β: シフト
        
        # AdaNormに特有のスケール係数を学習するためのパラメータ
        self.alpha = nn.Parameter(torch.ones(normalized_shape))  # α: 適応的スケーリング

    def forward(self, x):
        """
        順伝播
        :param x: 入力テンソル [Batch, Variate, d_model] などの形状
        :return: 正規化されたテンソル
        """
        # 各サンプル内で平均と分散を計算
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # 標準化（mean=0, std=1にする）
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        
        # 適応的なスケーリングを適用（α倍）
        x_norm = self.alpha * x_norm

        # γとβを使ってスケールとシフトを適用
        return self.gamma * x_norm + self.beta
