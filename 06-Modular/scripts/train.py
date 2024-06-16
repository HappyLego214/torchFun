import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm.auto import tqdm

import data_setup
import engine
import model_builder
import utils

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
HIDDEN_UNITS = 10
INPUT_SHAPE = 3
NUM_WORKERS = 0

def main():
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    data_path = Path('data')
    image_path = data_path / 'pizza_steak_sushi'

    script_dir = os.getcwd()
    parent_dir = os.path.dirname(script_dir)
    train_dir = parent_dir / image_path / 'train'
    test_dir = parent_dir / image_path / 'test'

    model_path = parent_dir / Path('models')

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_list = data_setup.create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        transform=data_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model_0 = model_builder.TinyVGG(input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=len(class_list))
    model_0.to(device)

    print(f"Using {device} For Training")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_0.parameters(), lr=LR)

    results = engine.train(
        model=model_0,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS
    )

    utils.save_model(
        model=model_0,
        target_dir=model_path,
        model_name='TinyVGG.pth'
    )


if __name__ == '__main__':
    main()
