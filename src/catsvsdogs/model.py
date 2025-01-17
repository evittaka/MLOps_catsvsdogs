import hydra
import timm
from omegaconf import DictConfig
from torch import nn
from loguru import logger

logger.add("logs/model.log", rotation="10 MB", level="INFO")

class MobileNetV3(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(MobileNetV3, self).__init__()
        logger.info("Initializing MobileNetV3 model...")
        self.model = timm.create_model("mobilenetv3_large_100", pretrained=cfg.model.pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
        logger.info("MobileNetV3 model initialized successfully with configuration")
        
    def forward(self, x):
        logger.debug("Forward pass invoked.")
        return self.model(x)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting main function...")
    model = MobileNetV3(cfg)
    logger.info(f"Model architecture:\n{model}")
    logger.info("Model created successfully!")


if __name__ == "__main__":
    main()
