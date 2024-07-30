import matplotlib.pyplot as plt
import numpy as np
import torch
import os 
import requests
import zipfile

from pathlib import Path

def save_model(
        model = torch.nn.Module,
        target_dir = str, 
        model_name = str,
):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path/model_name

    print(f"INFO: Saving Model To: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def view_dataloader_imgs(dataloader, class_list):
    
    imgs, labels = next(iter(dataloader))
    for i in range(10):
        target_img=imgs[i]
        # Scaling for display purposes of the image - not necessary
        sample_min, sample_max = target_img.min(), target_img.max()
        sample_scaled = (target_img - sample_min)/(sample_max - sample_min)

        plt.subplot(1,10,i+1)
        plt.imshow(sample_scaled.permute(1,2,0)) # Transfer image to Numpy Array
        plt.title(class_list[labels[i]])
        plt.axis(False)

def set_seeds(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()