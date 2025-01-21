import subprocess
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import wandb
from catsvsdogs.data import catsvsdogs
from catsvsdogs.model import MobileNetV3

logger.add("logs/training.log", rotation="10 MB", level="INFO")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def loss_function():
    return torch.nn.CrossEntropyLoss()


def upload_model_to_gcs():
    """
    Uploads the trained model to Google Cloud Storage with both a timestamped and a latest version filename
    using the gsutil CLI command.
    """
    bucket_name = "mlops_catsvsdogs"  # Your GCS bucket name
    source_file_path = "models/model.pth"  # Local trained model path

    # Define GCS destination filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    destination_blob_name_timestamped = f"gs://{bucket_name}/models/model_{timestamp}.pth"
    destination_blob_name_latest = f"gs://{bucket_name}/models/model_latest.pth"

    try:
        # Upload the timestamped model using gsutil
        logger.info(f"Uploading {source_file_path} to {destination_blob_name_timestamped}")
        subprocess.run(["gsutil", "cp", source_file_path, destination_blob_name_timestamped], check=True)
        logger.info(f"Model uploaded to {destination_blob_name_timestamped}")

        # Upload the latest model
        logger.info(f"Uploading {source_file_path} to {destination_blob_name_latest}")
        subprocess.run(["gsutil", "cp", source_file_path, destination_blob_name_latest], check=True)
        logger.info(f"Model uploaded to {destination_blob_name_latest}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload model to GCS: {e}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train a model on the cats vs dogs dataset."""
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs

    wandb_run = wandb.init(
        project="catsvsdogs",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    logger.info("Starting model training")
    logger.info(f"Training configuration: lr={lr}, batch_size={batch_size}, epochs={epochs}")

    model = MobileNetV3(cfg).to(DEVICE)

    train_set, _ = catsvsdogs()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    trainer = Trainer(max_epochs=epochs, devices=1, accelerator="auto")
    trainer.fit(model, train_dataloader)

    wandb.log({"train_loss": model.train_loss_history, "train_accuracy": model.train_accuracy_history})

    logger.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")

    # Save model as an artifact for wandb
    artifact = wandb.Artifact(
        name="catsvsdogs_model",
        type="model",
        description="Model trained on the cats vs dogs dataset",
    )
    artifact.add_file("models/model.pth")
    wandb_run.log_artifact(artifact)
    logger.info("Model saved to models/model.pth")
    upload_model_to_gcs()
    logger.info("Model uploaded to GCS")

    # Save training statistics as a figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(model.train_loss_history)
    axs[0].set_title("Train loss")
    axs[1].plot(model.train_accuracy_history)
    axs[1].set_title("Train accuracy")
    # fig.savefig("reports/figures/training_statistics.png")

    wandb.log({"training_statistics": wandb.Image(fig)})
    logger.info("Training statistics saved to reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
