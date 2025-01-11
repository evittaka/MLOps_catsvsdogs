import typer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import torch
import kaggle

def check_if_raw_data_exists(raw_data_path: str) -> bool:
    return os.path.exists(raw_data_path)

def download_data(raw_dir: str) -> None:
    """
    Download data from kaggle and save it to raw directory.
    """

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("shaunthesheep/microsoft-catsvsdogs-dataset", path=raw_dir, unzip=True)
    
def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """
    Process raw data and save it to processed directory.

    Args:
        raw_dir (str): Directory containing raw data.
        processed_dir (str): Directory to save processed data.
    """
    images_dir = os.path.join(raw_dir, "PetImages")

    if check_if_raw_data_exists(images_dir) == False:
        print("Data is missing from {}".format(raw_dir))
        download_data_check = input("Download data from kaggle? [y/n]: ")
        if download_data_check == "y":
            print("Getting data from kaggle to {}".format(raw_dir))
            download_data(raw_dir)
    else:
        print("Raw data exists in {}".format(raw_dir))

    # TODO: Read from config file or whatsoever
    image_size = (128, 128)
    test_size = 0.2

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images, targets = [], []

    class_map = {"Cat": 0, "Dog": 1}

    # TODO: Read from config file or whatsoever
    max_samples_per_class = 500  # Limit the number of samples per class

    for class_name, label in class_map.items():
        class_dir = os.path.join(images_dir, class_name)
        img_count = 0
        total_images = len(os.listdir(class_dir))  # Get total number of images in the class directory

        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}", total=min(max_samples_per_class, total_images)):
            if img_count >= max_samples_per_class:
                break
            try:
                img_path = os.path.join(class_dir, img_name)
                images.append(transform(Image.open(img_path).convert("RGB")))
                targets.append(label)
                img_count += 1
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        torch.stack(images), torch.tensor(targets), test_size=test_size, random_state=42
    )

    os.makedirs(processed_dir, exist_ok=True)
    for split, imgs, lbls in [("train", train_imgs, train_lbls), ("test", test_imgs, test_lbls)]:
        torch.save(imgs, os.path.join(processed_dir, f"{split}_images.pt"))
        torch.save(lbls, os.path.join(processed_dir, f"{split}_target.pt"))
    print(f"Saved processed data to {processed_dir}")

def catsvsdogs() -> (
    tuple[
        torch.utils.data.Dataset,
        torch.utils.data.Dataset,
    ]
):
    """Return train and test datasets for cats vs dogs classification."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess_data)
