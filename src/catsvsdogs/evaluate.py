import torch
import typer
from catsvsdogs.model import MobileNetV3

from catsvsdogs.data import catsvsdogs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model."""
    print("Evaluating model ont test set")
    print(model_checkpoint)

    model = MobileNetV3().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = catsvsdogs()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = (
            img.to(DEVICE),
            target.to(DEVICE),
        )
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
