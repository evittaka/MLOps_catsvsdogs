from pathlib import Path
import typer
from torch.utils.data import Dataset
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

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Check if data exists in folder
        if not check_if_data_exists(self.data_path):
            print(f"Data is missing from {self.data_path}")
            print(f"Getting data from Kaggle to {self.data_path}")
            download_data(self.data_path)
        else:
            print(f"Data already exists in {self.data_path}")

        print(f"Preprocessing complete. Processed data will be in {output_folder}")


def check_if_data_exists(raw_data_path: Path) -> bool:
    """Check if the dataset already exists in `data/raw`."""
    pet_images_path = raw_data_path / "PetImages"
    
    # Check if the PetImages directory exists in the raw_data_path
    if pet_images_path.exists():
        print(f"Dataset found at {pet_images_path}.")
        return True
    
    # Additional safeguard: If it exists in a nested folder, move it
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
    # Remove the empty source folder
    src_folder.rmdir()


def download_data(raw_data_path: Path) -> None:
    """Download data using kagglehub."""

    try:
        print("Downloading dataset...")
        dataset_path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
        
        # Ensure the downloaded dataset is moved directly to `raw_data_path`
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


if __name__ == "__main__":
    typer.run(preprocess)
