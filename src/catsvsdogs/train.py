import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from loguru import logger

from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

logger.add("logs/training.log", rotation="10 MB", level="INFO")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def loss_function():
    return torch.nn.CrossEntropyLoss()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train a model on the cats vs dogs dataset."""
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs

    logger.info("Starting model training")
    logger.info(f"Training configuration: lr={lr}, batch_size={batch_size}, epochs={epochs}")

    model = MobileNetV3(cfg).to(DEVICE)
    train_set, _ = catsvsdogs()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for _, (img, target) in progress_bar:
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            progress_bar.set_postfix({"loss": loss.item(), "accuracy": accuracy})

    logger.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    logger.info("Model saved to models/model.pth")

    # Save training statistics as a figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    logger.info("Training statistics saved to reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
