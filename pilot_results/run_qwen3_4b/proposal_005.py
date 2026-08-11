import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random

def augment(xb: torch.Tensor) -> torch.Tensor:
    """Apply random horizontal flip with 50% probability."""
    if random.random() < 0.5:
        return torch.flip(xb, dims=(3,))
    return xb

def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16x16
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 8x8
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        # ... (training loop continues)
    return model

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            output = model(xb)
            _, predicted = torch.max(output, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    return correct / total

# ... (rest of the code remains unchanged)
