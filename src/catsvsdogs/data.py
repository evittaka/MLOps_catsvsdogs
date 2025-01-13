from pathlib import Path
import typer
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import os
import shutil
import kagglehub

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        raise NotImplementedError("Dataset length logic needs implementation.")

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        raise NotImplementedError("Sample fetching logic needs implementation.")

    def preprocess(self, output_folder: Path, image_size=(128, 128), test_size=0.2, max_samples_per_class=500) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Check if data exists in folder
        if not check_if_data_exists(self.data_path):
            print(f"Data is missing from {self.data_path}")
            print(f"Getting data from Kaggle to {self.data_path}")
            download_data(self.data_path)
        else:
            print(f"Data already exists in {self.data_path}")

        images_dir = self.data_path / "PetImages"
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        images, targets = [], []
        class_map = {"Cat": 0, "Dog": 1}

        for class_name, label in class_map.items():
            class_dir = images_dir / class_name
            img_count = 0
            total_images = len(list(class_dir.iterdir()))

            for img_name in tqdm(class_dir.iterdir(), desc=f"Processing {class_name}", total=min(max_samples_per_class, total_images)):
                if img_count >= max_samples_per_class:
                    break
                try:
                    img_path = img_name
                    images.append(transform(Image.open(img_path).convert("RGB")))
                    targets.append(label)
                    img_count += 1
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
            torch.stack(images), torch.tensor(targets), test_size=test_size, random_state=42
        )

        output_folder.mkdir(parents=True, exist_ok=True)
        for split, imgs, lbls in [("train", train_imgs, train_lbls), ("test", test_imgs, test_lbls)]:
            torch.save(imgs, output_folder / f"{split}_images.pt")
            torch.save(lbls, output_folder / f"{split}_target.pt")
        print(f"Saved processed data to {output_folder}")

def check_if_data_exists(raw_data_path: Path) -> bool:
    """Check if the dataset already exists in `data/raw`."""
    pet_images_path = raw_data_path / "PetImages"

    if pet_images_path.exists():
        print(f"Dataset found at {pet_images_path}.")
        return True

    for subfolder in raw_data_path.rglob("*"):
        if (subfolder / "PetImages").exists():
            print(f"Dataset found in {subfolder}, moving to {raw_data_path}...")
            move_contents_to_folder(subfolder, raw_data_path)
            return True

    return False

def move_contents_to_folder(src_folder: Path, dest_folder: Path) -> None:
    """Move all contents from `src_folder` to `dest_folder`."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    for item in src_folder.iterdir():
        shutil.move(str(item), str(dest_folder))
    src_folder.rmdir()

def download_data(raw_data_path: Path) -> None:
    """Download data using kagglehub."""
    try:
        print("Downloading dataset...")
        dataset_path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

        downloaded_path = Path(dataset_path)
        if downloaded_path.exists() and downloaded_path != raw_data_path:
            print(f"Moving downloaded dataset from {downloaded_path} to {raw_data_path}...")
            move_contents_to_folder(downloaded_path, raw_data_path)

        print(f"Dataset successfully downloaded and moved to {raw_data_path}")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """Main preprocess entry point."""
    print("Starting preprocessing...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

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

if __name__ == "__main__":
    typer.run(preprocess)
