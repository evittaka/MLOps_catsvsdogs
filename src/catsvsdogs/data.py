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
            download_data_check = input("Download data from Kaggle? [y/n]: ")
            if download_data_check.lower() == "y":
                print(f"Getting data from Kaggle to {self.data_path}")
                download_data(self.data_path)
            else:
                print("Data download skipped. Exiting preprocessing.")
                return
        else:
            print(f"Data already exists in {self.data_path}")

        print(f"Preprocessing complete. Processed data will be in {output_folder}")


def check_if_data_exists(raw_data_path: Path) -> bool:
    """Check if the dataset already exists in `data/raw` and flatten if necessary."""
    for subfolder in raw_data_path.rglob("*"):
        if (subfolder / "PetImages").exists():
            # If found in a nested subfolder, move it directly under `data/raw`
            if subfolder != raw_data_path:
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
        
        # Move the data to raw_data_path
        shutil.move(dataset_path, raw_data_path)
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
