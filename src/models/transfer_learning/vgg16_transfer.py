import torch.nn as nn
from torchvision import models

class VGG16Transfer(nn.Module):
    def __init__(self, num_classes:int = 2):
        super(VGG16Transfer, self).__init__()
        # 事前訓練済みのVGG16モデルをロード
        self.vgg16 = models.vgg16(pretrained=True)
        
        # 事前訓練済みの畳み込み層のパラメータを凍結
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        # 最後の全結合層を新しいタスク用に置き換え
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)

# class check
# model = VGG16Transfer()
# print(model)