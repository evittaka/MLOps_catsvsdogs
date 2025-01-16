import shutil
from pathlib import Path

import hydra
import kagglehub
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm


class MyDataset(Dataset):
    """Custom dataset for preprocessing and loading data."""

    def __init__(self, raw_data_path: Path):
        self.data_path = raw_data_path

    def preprocess(self, cfg: DictConfig):
        """Preprocess the raw data and save it to the output folder."""
        # Check if data exists, download if necessary
        if not self.check_if_data_exists():
            print(f"Data is missing from {self.data_path}.")
            print(f"Downloading data to {self.data_path}...")
            self.download_data()
        else:
            print(f"Data already exists in {self.data_path}.")

        images_dir = self.data_path / "PetImages"
        transform = transforms.Compose(
            [
                transforms.Resize(cfg.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        images, targets = [], []
        class_map = {"Cat": 0, "Dog": 1}

        for class_name, label in class_map.items():
            class_dir = images_dir / class_name
            img_count = 0
            total_images = len(list(class_dir.iterdir()))

            for img_name in tqdm(
                class_dir.iterdir(),
                desc=f"Processing {class_name}",
                total=min(cfg.data.max_samples_per_class, total_images),
            ):
                if img_count >= cfg.data.max_samples_per_class:
                    break
                try:
                    img_path = img_name
                    images.append(transform(Image.open(img_path).convert("RGB")))
                    targets.append(label)
                    img_count += 1
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
            torch.stack(images), torch.tensor(targets), test_size=cfg.data.test_size, random_state=42
        )

        output_folder = Path(cfg.data.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        for split, imgs, lbls in [("train", train_imgs, train_lbls), ("test", test_imgs, test_lbls)]:
            torch.save(imgs, output_folder / f"{split}_images.pt")
            torch.save(lbls, output_folder / f"{split}_target.pt")
        print(f"Saved processed data to {output_folder}")

    def check_if_data_exists(self) -> bool:
        """Check if the dataset already exists."""
        pet_images_path = self.data_path / "PetImages"
        return pet_images_path.exists()

    def download_data(self):
        """Download the dataset using kagglehub."""
        try:
            dataset_path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
            downloaded_path = Path(dataset_path)
            if downloaded_path.exists() and downloaded_path != self.data_path:
                print(f"Moving downloaded dataset from {downloaded_path} to {self.data_path}...")
                self.move_contents_to_folder(downloaded_path, self.data_path)
            print(f"Dataset successfully downloaded to {self.data_path}.")
        except Exception as e:
            print(f"An error occurred while downloading the dataset: {e}")

    @staticmethod
    def move_contents_to_folder(src_folder: Path, dest_folder: Path):
        """Move all contents from `src_folder` to `dest_folder`."""
        dest_folder.mkdir(parents=True, exist_ok=True)
        for item in src_folder.iterdir():
            shutil.move(str(item), str(dest_folder))
        src_folder.rmdir()


def catsvsdogs() -> (
    tuple[
        torch.utils.data.Dataset,
        torch.utils.data.Dataset,
    ]
):
    """Return train and test datasets for cats vs dogs classification."""
    train_images = torch.load("data/processed/train_images.pt", weights_only=True)
    train_target = torch.load("data/processed/train_target.pt", weights_only=True)
    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_target = torch.load("data/processed/test_target.pt", weights_only=True)

    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)
    return train_set, test_set


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dataset = MyDataset(Path(cfg.data.raw_data_path))
    dataset.preprocess(cfg)


if __name__ == "__main__":
    main()
