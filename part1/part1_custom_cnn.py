import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import DATA_DIR, MODELS_DIR, get_device, run_classification


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv 1 : kernel=3, padding=1, stride=1  -> 16x28x28 -> pool -> 16x14x14
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv 2 : kernel=3, padding=1, stride=1  -> 32x14x14 -> pool -> 32x7x7
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Régularisation
        self.dropout = nn.Dropout(0.25)
        # Couches fully connected
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def main():
    device = get_device()

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    l2_regularization = 1e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(DATA_DIR, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CustomCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)

    run_classification("custom_cnn", model, train_loader, test_loader, optimizer, device, num_epochs)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "custom_cnn.pth"))


if __name__ == "__main__":
    main()