import torch
from typing import Dict, List, Tuple
from tqdm.auto import tqdm    

def train_step(
       model: torch.nn.Module,
       dataloader: torch.utils.data.DataLoader,
       loss_fn: torch.nn.Module,
       optimizer: torch.optim.Optimizer,
       device: torch.device
) -> Tuple[float, float]:

    model.train()
    train_loss, train_acc = 0,0

    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax((torch.softmax(logits, dim=1)), dim=1)
        train_acc += ((preds == y).sum().item() / len(preds))
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc 

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
) -> Tuple[float, float]:
    
    model.eval()
    test_loss, test_acc = 0,0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss

            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            test_acc += ((preds == y).sum().item() / len(preds))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = 10
):
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


