import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import time
import random
import os
import json
from torch.utils.data import DataLoader

SEED = 0
BATCH_SIZE = 32
TRAINING_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def build_model():
    # Simple neural network example
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return Net()

def evaluate(model, x_val, y_val, device):
    model.eval()
    with torch.no_grad():
        outputs = model(x_val.to(device))
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_val.to(device)).float().sum()
        accuracy = correct / y_val.shape[0]
    return accuracy.item()

def main():
    # Load data (simplified example)
    data = {
        'train': {
            'x': torch.randn(60000, 28*28),
            'y': torch.randint(0, 10, (60000,))
        },
        'val': {
            'x': torch.randn(10000, 28*28),
            'y': torch.randint(0, 10, (10000,))
        }
    }
    
    x_train, y_train = data['train']['x'], data['train']['y']
    x_val, y_val = data['val']['x'], data['val']['y']
    
    # Initialize model
    model = build_model().to(DEVICE)
    
    # Optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    steps = 0
    epochs = 0
    start_time = time.time()
    
    while time.time() - start_time < 3600:  # 1 hour max
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = x_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            steps += 1
            if steps % 100 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
        
        epochs += 1
    
    # Evaluate
    val_acc = evaluate(model, x_val, y_val, DEVICE)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save results
    result = {
        "val_acc": float(val_acc),
        "epochs_completed": epochs,
        "steps": steps,
        "train_seconds": round(time.time() - start_time, 2),
        "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    }
    
    with open("result.json", "w") as f:
        json.dump(result, f)
    
    # Save model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
