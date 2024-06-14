import torch
import os 
import data_setup
import engine
import model_builder
import utils

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch import nn

from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
HIDDEN_UNITS = 10
INPUT_SHAPE = 3

device = 'cuda' if torch.cuda.is_available else 'cpu'

data_path = Path('data')
model_path = Path('models')
image_path = data_path/'pizza_steak_sushi'

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_list = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count()
)

model_0 = model_builder.TinyVGG(input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=len(class_list))
model_0.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_0.parameters, lr=LR)

results = engine.train(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device = device,
    epochs = EPOCHS
)

utils.save_model(
    model=model_0,
    target_dir=model_path,
    model_name='TinyVGG'
)