import torch
import torchvision
import argparse
from typing import List

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_image(
        model: torch.nn.Module,
        image_path: str,
        class_names: List[str],
        transform=None,
        device: torch.device = None
):
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image /= 255

    if transform:
        target_image = transform(target_image)
        
    model.to(device)
    with torch.inference_mode():
        logits = model(target_image.to(device))
        probabilities = torch.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)
        prediction = class_names[labels]

    print(f"Prediction Class: {prediction} | Probability of Prediction: {probabilities} ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--model')

    main()
