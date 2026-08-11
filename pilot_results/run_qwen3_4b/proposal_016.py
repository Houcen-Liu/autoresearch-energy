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
  * build_model() must exist and return the model, so final_eval.py can rebuild it.

Everything else -- architecture, optimizer, schedule, augmentation, batch size,
regularisation -- is fair game.

BASELINE NOTE: this recipe is deliberately unoptimised (no augmentation, constant
LR, no normalisation layers) so that the agent has real headroom. Calibrated in
week 1 to land in the 0.60-0.85 validation band (gate G2). Once calibrated it is a
FIXED VARIABLE: pin the commit and never touch it again.
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Load data
    train_dataset = ...  # Placeholder for actual data loading
    test_dataset = ...   # Placeholder for actual data loading

    # Create model
    model = build_model()

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer configuration
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_acc = correct / total

    # Save results
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()

def build_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 10)
    )
    return model

if __name__ == "__main__":
    main()
