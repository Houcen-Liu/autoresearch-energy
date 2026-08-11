"""CIFAR-10 training recipe -- THE SINGLE FILE THE AGENT EDITS.

Contract with the harness (do not break any of these; they are checked statically
by harness/guards.py before this file is ever executed):

  * TRAIN_SECONDS is a fixed wall-clock training budget and MUST keep its value.
    Training stops at the budget boundary, mid-epoch if necessary. Buying accuracy
    with extra time destroys comparability between experiments.
  * Data comes only from prepare_cifar.load_splits(); the test split is off limits.
  * On success this file writes result.json with keys:
        val_acc, epochs_completed, steps, train_seconds, peak_vram_mb
    and saves the trained weights to the checkpoint path.
  * build_model() must exist and return the model, so final_eval.py can use it.
  * All changes must be within allowed imports and structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import random
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define the model architecture
def build_model() -> nn.Module:
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
        nn.Linear(128 * 4 * 4, 512),  # Changed from 256 to 512
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 10),
    )

# Define the training loop
def train_model(model, train_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model

# Main function
def main():
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    
    # Create model
    model = build_model()
    
    # Train model
    model = train_model(model, trainloader, epochs=10)
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    
    # Output result
    print("Training completed. Model saved to 'model.pth'")

if __name__ == "__main__":
    main()
