import hydra
import torch
from omegaconf import DictConfig

from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    print("Evaluating model on test set")
    print(cfg.evaluate.model_checkpoint)

    # Initialize model using the configuration
    model = MobileNetV3(cfg).to(DEVICE)
    # model.load_state_dict(torch.load(cfg.evaluate.model_checkpoint))
    model.load_state_dict(torch.load(cfg.evaluate.model_checkpoint, weights_only=True))

    _, test_set = catsvsdogs()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.evaluate.batch_size)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)

    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    evaluate()
