import os
import requests
import zipfile
import torch
""
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from typing import List

def get_data(
    source: str,
    target: str,
):

    data_path = Path('data/')
    image_path = data_path / target
    print(image_path)
    
    train_path = data_path / image_path / 'test'
    test_path = data_path / image_path / 'train'

    

    if image_path.is_dir():
        print(f"Directory Exists {image_path}")
    else:
        print(f"Directory Does Not Exist {image_path}")
        image_path.mkdir(parents=True, exist_ok=True)

    src_file = Path(source).name 

    if train_path.is_dir() and test_path.is_dir():
        print(f"Directories Exist - {train_path} - {test_path}")
        print(f"Skipping Downloading - Skipping Extraction")
    else:
        print(f"Directories Does Not Exist - {train_path} - {test_path}")
        with open(data_path / src_file, 'wb') as f:
            print(f"Downloading Data")
            request = requests.get(source)
            f.write(request.content)

        with zipfile.ZipFile(data_path / src_file, 'r') as zipdata:
            print(f"Unzipping Data")
            zipdata.extractall(image_path)

        os.remove(data_path/src_file)

    return image_path

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transforms: transforms.Compose,
        augmented: transforms.Compose,
        batch_size: int,
        num_workers: int = 1
):
    if augmented:
        train_dataset = datasets.ImageFolder(train_dir, transform=augmented)
    else:
        train_dataset = datasets.ImageFolder(train_dir, transform=transforms)
        
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms)

    class_list = train_dataset.classes

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, class_list

def info_dataloader(
    train_dataloader:torch.utils.data.DataLoader,
    test_dataloader:torch.utils.data.DataLoader,
    class_list:List[str]
):
    print(f"Train Dataloader: Batch Size = {train_dataloader.batch_size} | Number of Batches = {len(train_dataloader)}")
    print(f"Test Dataloader: Batch Size = {test_dataloader.batch_size} | Number of Batches = {len(test_dataloader)}")
    print(f"Class List Length: {len(class_list)}")


