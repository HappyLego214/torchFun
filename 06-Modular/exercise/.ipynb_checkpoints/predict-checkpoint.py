import torch
import torchvision
import argparse
import model_builder
from typing import List
from torchvision import transforms
from pathlib import Path


def main():
    predict_image(model=model, image_path=image_path, transform=TRANSFORM)


def predict_image(
        model: torch.nn.Module,
        image_path=None,
        transform=None,
        device: torch.device = None
):
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image /= 255

    if transform:
        target_image = default_transform(target_image)

    model.to(device)
    with torch.inference_mode():
        logits = model(target_image.unsqueeze(dim=0)).to(device)
        probabilities = torch.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)
        prediction = class_names[labels]
        prediction_prob = probabilities.tolist()[0][labels.item()]

    print(f"Prediction Class: {prediction} | Probability of Prediction: {prediction_prob:.2f} ")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--model_name', type=str, default='', help='assign trained models - include extension')
    parser.add_argument('--image_name', type=str, help='image to be used for model prediction')
    parser.add_argument('--transform', type=bool, default=1, help='apply default transform to image')
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    IMAGE_NAME = args.image_name
    TRANSFORM = args.transform

    class_names = ['pizza', 'steak', 'sushi']

    default_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    model_path = Path('exercise_model')/MODEL_NAME
    image_path = Path('exercise_predictions')/IMAGE_NAME

    model = model_builder.TinyVGG(3, 20, len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    main()

