import os
import requests
import zipfile

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

parentDir = Path(os.getcwd()).parent

data_path = Path('data/')
image_path = parentDir / data_path / 'pizza_steak_sushi'
train_path = parentDir / data_path / image_path / 'test'
test_path = parentDir / data_path / image_path / 'train'

def get_data():
    if image_path.is_dir():
        print(f"Directory Exists {image_path}")
    else:
        print(f"Directory Does Not Exist {image_path}")
        image_path.mkdir(parents=True, exist_ok=True)

    if train_path.is_dir() and test_path.is_dir():
        print(f"Directories Exist - {train_path} - {test_path}")
        print(f"Skipping Downloading - Skipping Extraction")
    else:
        print(f"Directories Does Not Exist - {train_path} - {test_path}")
        with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
            print(f"Downloading Data")
            request = requests.get(
                "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
            f.write(request.content)

        with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zipdata:
            print(f"Unzipping Data")
            zipdata.extractall(image_path)

        os.remove(data_path/'pizza_steak_sushi.zip')


def create_dataloader(
        train_dir: str,
        test_dir: str,
        transforms: transforms.Compose,
        batch_size: int,
        num_workers: int = 1
):
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms)

    class_list = train_dataset.classes

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, class_list
