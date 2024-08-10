import torch
import torchvision
import argparse
from typing import List
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def predict_image(
        model: torch.nn.Module,
        image_path=None,
        transform=None,
        class_list=None,
        device: torch.device = None
):

    image = Image.open(image_path)
    if transform:
        transformed_image = transform(image).to(device)
    else: 
        transformed_image = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225])
        ]).to(device)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(transformed_image.unsqueeze(dim=0)).to(device)
        
    probabilities = torch.softmax(logits, dim=1)
    labels = torch.argmax(probabilities, dim=1)
    
    prediction = class_list[labels]
    prediction_prob = probabilities.max()

    plt.figure()
    plt.imshow(image)
    plt.title(f"Pred: {prediction} | Prob: {prediction_prob:.2f}")
    plt.axis()