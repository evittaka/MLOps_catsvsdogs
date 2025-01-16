import matplotlib.pyplot as plt
import torch
import typer
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def train(
    lr: float = None,
    batch_size: int = None,
    epochs: int = None,
) -> None:
    """Train a model on the cats vs dogs dataset."""
    if not GlobalHydra().is_initialized():
        initialize(config_path="../../configs", job_name="train", version_base=None)
    hydra_cfg = compose(config_name="config")

    # Use CLI values if provided, otherwise fallback to config values
    lr = lr or hydra_cfg.train.lr
    batch_size = batch_size or hydra_cfg.train.batch_size
    epochs = epochs or hydra_cfg.train.epochs

    print("Training model")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MobileNetV3(hydra_cfg).to(DEVICE)
    train_set, _ = catsvsdogs()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
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

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")

    # Save training statistics as a figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    app()
