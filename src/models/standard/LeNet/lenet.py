import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_channel:int = 3, dropout:float = 0.1) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
    
    def forward(self, X):
        output = self.block(X)
        return output

