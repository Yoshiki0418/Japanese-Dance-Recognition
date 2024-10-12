import torch
from torch import nn
from src.models.standard import VGG16

class Handfan_VGG16(nn.Module):
    """
    input_image_size --> (224,224)
    """
    def __init__(
        self,
        in_channel: int = 3,
        num_classes: int = 2,
        p_drop: float = 0.2,
    ) -> None:
        
        super().__init__()

        self.conv = VGG16(input_channel=in_channel)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(4096, num_classes)
        )     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        out = self.head(x)
        return out
