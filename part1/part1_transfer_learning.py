import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import DATA_DIR, get_device, run_classification


def build_model(choice):
    if choice == "vgg16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif choice == "alexnet":
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("model doit être 'vgg16' ou 'alexnet'")
    # On gèle la partie convolutive, on ne fine-tune que le classifieur
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 10)  # 1000 classes ImageNet -> 10 chiffres
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-subset", type=int, default=10000,
                        help="nb d'images d'entraînement (0 = tout MNIST, à réserver à Kaggle/Colab)")
    parser.add_argument("--test-subset", type=int, default=2000, help="0 = tout")
    args = parser.parse_args()

    device = get_device()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = torchvision.datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(DATA_DIR, train=False, transform=transform, download=True)
    if args.train_subset:
        train_ds = Subset(train_ds, range(args.train_subset))
    if args.test_subset:
        test_ds = Subset(test_ds, range(args.test_subset))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = build_model(args.model).to(device)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    run_classification(f"transfer_{args.model}", model, train_loader, test_loader,
                       optimizer, device, args.epochs)


if __name__ == "__main__":
    main()