import data_setup
import engine
import model_builder
import utils

import torch
import os
from pathlib import Path
from torchvision import transforms

INPUT_SHAPE = 3
HIDDEN_UNITS = 10
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 10
NUM_WORKERS = 0
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = Path('data')
    image_path = data_path/'pizza_steak_sushi'
    exercise_model = Path('exercise_model')

    exercise_dir = os.getcwd()
    parent_dir = os.path.dirname(exercise_dir)

    train_dir = parent_dir/image_path/'train'
    test_dir = parent_dir/image_path/'test'

    data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_list = data_setup.create_dataloader(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        transforms=data_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model_0 = model_builder.TinyVGG(input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=len(class_list))
    model_0.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_0.parameters(), lr=LR)

    results = engine.train(model=model_0,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           device=device,
                           epochs=EPOCHS)

    utils.save_model(model=model_0,
                     target_dir=str(exercise_model),
                     model_name='TinyVGG.pth')


if __name__ == '__main__':
    main()