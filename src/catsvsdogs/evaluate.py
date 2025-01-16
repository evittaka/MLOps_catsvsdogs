import torch
import typer
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def evaluate(
    model_checkpoint: str = "models/model.pth",
    batch_size: int = 32,
) -> None:
    """Evaluate a trained model."""
    print("Evaluating model on test set")
    print(model_checkpoint)

    if not GlobalHydra().is_initialized():
        initialize(config_path="../../configs", job_name="evaluate", version_base=None)
    hydra_cfg = compose(config_name="config")

    # Initialize model using the configuration
    model = MobileNetV3(hydra_cfg).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))

    _, test_set = catsvsdogs()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)

    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()
