import timm
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


class MobileNetV3(pl.LightningModule):
    def __init__(self, pretrained: bool = True, num_classes: int = 2, learning_rate: float = 1e-2):
        super(MobileNetV3, self).__init__()
        self.learning_rate = learning_rate
        self.criterium = nn.CrossEntropyLoss()  # Define the loss function
        
        self.model = timm.create_model("mobilenetv3_large_100", pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        # Initialize lists to track loss and accuracy during training
        self.train_loss_history = []
        self.train_accuracy_history = []

    def forward(self, x):
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
        return optim.Adam(self.parameters(), lr = self.learning_rate)
    
    def loss_fn(self, preds, target):
        return nn.CrossEntropyLoss()(preds, target)
