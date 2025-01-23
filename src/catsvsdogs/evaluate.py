import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix

from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

logger.add("logs/evaluation.log", rotation="10 MB", level="INFO")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model and log results to wandb."""
    wandb.init(project="catsvsdogs", name="evaluation")

    logger.info("Starting model evaluation...")
    logger.info(f"Using model checkpoint: {cfg.evaluate.model_checkpoint}")

    # Initialize model
    model = MobileNetV3(cfg).to(DEVICE)
    model.load_state_dict(torch.load(cfg.evaluate.model_checkpoint, weights_only=True))
    logger.info("Model loaded successfully")

    _, test_set = catsvsdogs()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.evaluate.batch_size)

    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []

    logger.info("Evaluating the model on the test set...")
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            predicted_classes = y_pred.argmax(dim=1)

            correct += (predicted_classes == target).float().sum().item()
            total += target.size(0)

            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = correct / total
    logger.info(f"Test accuracy: {accuracy:.4f}")
    wandb.log({"test_accuracy": accuracy})

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})

    logger.info("Evaluation complete, results logged to wandb.")


if __name__ == "__main__":
    evaluate()
