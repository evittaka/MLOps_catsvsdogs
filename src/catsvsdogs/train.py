import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import typer

from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on the cats vs dogs dataset."""
    print("Training model")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MobileNetV3(pretrained=True, num_classes=2, learning_rate=lr)
    
    train_set, _ = catsvsdogs()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle = True)

    trainer = Trainer(max_epochs=epochs, devices = 1, accelerator = 'cpu')
    trainer.fit(model, train_dataloader)

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")

    # Save training statistics as a figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(model.train_loss_history)
    axs[0].set_title("Train loss")
    axs[1].plot(model.train_accuracy_history)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    typer.run(train)
