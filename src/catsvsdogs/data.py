from pathlib import Path

import typer
from torch.utils.data import Dataset

import os

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Check if data exists in folder
        if check_if_data_exists(self.data_path) == False:
            print("Data is missing from {}".format(self.data_path))
            download_data_check = input("Download data from kaggle? [y/n]: ")
            if download_data_check == "y":
                print("Getting data from kaggle to {}".format(self.data_path))
                download_data(self.data_path)
        else:
            print("Data exists in {}".format(self.data_path))        

def check_if_data_exists(raw_data_path: Path) -> None:
    # Check if the dataset already exists
    return os.path.exists(raw_data_path / "PetImages")

def download_data(raw_data_path: Path) -> None:
    """ Get data from kaggle using kaggle api
    """

    # Get the dataset from kaggle
    print("Using kaggle API key located in User/.../.kaggle")
    import kaggle

    # Authenticate kaggle API from Users/.../.kaggle
    kaggle.api.authenticate()

    # Get the dataset and put under data/raw
    kaggle.api.dataset_download_files("shaunthesheep/microsoft-catsvsdogs-dataset", path=raw_data_path, unzip=True)

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
