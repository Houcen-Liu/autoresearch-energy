import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        nn.Linear(128 * 4 * 4, 256),
        nn.Dropout(0.5),  # Added dropout layer
        nn.ReLU(),
        nn.Linear(256, 10),
    )

def main():
    # Initialize model
    model = build_model()
    
    # Define optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop (simplified for brevity)
    for epoch in range(10):
        # Training step
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation (simplified)
    model.eval()
    with torch.no_grad():
        accuracy = 0.0
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).float().mean()
    
    print(f"Validation Accuracy: {accuracy.item() * 100:.2f}%")

if __name__ == "__main__":
    # Data loading and training setup (simplified)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    dataset = torch.utils.data.TensorDataset(torch.randn(50000, 32*32), torch.randint(0, 10, (50000,)))
    main()
