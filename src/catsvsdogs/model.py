import hydra
import pytorch_lightning as pl
import timm
from loguru import logger
from omegaconf import DictConfig
from torch import nn, optim

logger.add("logs/model.log", rotation="10 MB", level="INFO")


class MobileNetV3(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(MobileNetV3, self).__init__()
        logger.info("Initializing MobileNetV3 model...")
        self.learning_rate = cfg.train.lr
        self.criterium = nn.CrossEntropyLoss()  # Define the loss function
        
        self.model = timm.create_model("mobilenetv3_large_100", pretrained=cfg.model.pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
        logger.info("MobileNetV3 model initialized successfully with configuration")

        # Initialize lists to track loss and accuracy during training
        self.train_loss_history = []
        self.train_accuracy_history = []

    def forward(self, x):
        logger.debug("Forward pass invoked.")
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)

        # Track loss and accuracy
        accuracy = (preds.argmax(dim=1) == target).float().mean()
        self.train_loss_history.append(loss.item())
        self.train_accuracy_history.append(accuracy.item())
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def loss_fn(self, preds, target):
        return nn.CrossEntropyLoss()(preds, target)



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting main function...")
    model = MobileNetV3(cfg)
    logger.info(f"Model architecture:\n{model}")
    logger.info("Model created successfully!")


if __name__ == "__main__":
    main()