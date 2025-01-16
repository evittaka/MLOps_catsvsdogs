import timm
from torch import nn


class MobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(MobileNetV3, self).__init__()
        self.model = timm.create_model("mobilenetv3_large_100", pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        return self.model(x)
