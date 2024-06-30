from pathlib import Path
import matplotlib.pyplot as plt
import torch

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