import hydra
from omegaconf import DictConfig
from torch import nn
import timm


class MobileNetV3(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(MobileNetV3, self).__init__()
        self.model = timm.create_model("mobilenetv3_large_100", pretrained=cfg.model.pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        return self.model(x)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    model = MobileNetV3(cfg)
    print(model)


if __name__ == "__main__":
    main()
