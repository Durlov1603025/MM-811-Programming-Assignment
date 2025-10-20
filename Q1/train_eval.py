import torch
import torch.nn as nn

# ---------------------------------------------------------
# Training and evaluation utilities
# These functions are imported and used by main.py
# ---------------------------------------------------------


def train_epoch(model, device, dataloader, optimizer, criterion):
    """
    Train the model for one epoch on the training dataset.

    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()    
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(targets).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100.0
    return avg_loss, acc


def evaluate(model, device, dataloader, criterion):
    """
    Evaluate model on validation or test dataset.
    
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100.0
    return avg_loss, acc
