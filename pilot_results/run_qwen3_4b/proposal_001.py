import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ... [rest of imports remain unchanged] ...

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),  # 16x16
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),  # 8x8
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),  # 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )

def augment(xb: torch.Tensor) -> torch.Tensor:
    """Data augmentation hook with random horizontal flip (50% probability)"""
    if random.random() < 0.5:
        xb = xb.flip(dims=[2, 3])
    return xb

# ... [rest of code remains unchanged] ...
