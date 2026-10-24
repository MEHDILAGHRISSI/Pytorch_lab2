import argparse
import os
import sys
import time

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import DATA_DIR, MODELS_DIR, classification_metrics, get_device, save_results


class MNISTDetectionDataset(Dataset):
    """MNIST transformé en détection : 1 boîte englobante par chiffre."""

    def __init__(self, root, train=True):
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img_tensor = F.to_tensor(img).repeat(3, 1, 1)  # Faster R-CNN attend 3 canaux

        pos = torch.where(img_tensor[0] > 0)
        if pos[0].numel() == 0:
            xmin, ymin, xmax, ymax = 0, 0, 28, 28
        else:
            xmin, xmax = torch.min(pos[1]).item(), torch.max(pos[1]).item()
            ymin, ymax = torch.min(pos[0]).item(), torch.max(pos[0]).item()
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

        target = {
            "boxes": torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32),
            "labels": torch.as_tensor([label + 1], dtype=torch.int64),  # 0 = fond
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=11, img_size=128):
    # min_size/max_size : évite de upscaler les 28x28 jusqu'à 800 px (très lent)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=img_size, max_size=img_size
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    # 1) Prédictions -> accuracy / F1 (label du détecteur le plus confiant)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([im.to(device) for im in images])
            for out, tgt in zip(outputs, targets):
                if len(out["scores"]) > 0:
                    pred = out["labels"][torch.argmax(out["scores"])].item() - 1
                else:
                    pred = -1  # aucune détection
                preds.append(pred)
                labels.append(tgt["labels"][0].item() - 1)
    acc, f1 = classification_metrics(labels, preds)

    # 2) Loss de détection sur le jeu de test (mode train mais sans gradient)
    model.train()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            total += sum(model(images, targets).values()).item()
            n += 1
    return acc, f1, total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-subset", type=int, default=6000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    train_ds = Subset(MNISTDetectionDataset(DATA_DIR, train=True), range(args.train_subset))
    test_ds = Subset(MNISTDetectionDataset(DATA_DIR, train=False), range(args.test_subset))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = sum(model(images, targets).values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {losses.item():.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - start
    print(f"Temps d'entraînement Faster R-CNN : {train_time:.2f} s")

    acc, f1, loss = evaluate(model, test_loader, device)
    print(f"[faster_rcnn] Accuracy: {acc * 100:.2f}% | F1: {f1:.4f} | Loss: {loss:.4f}")
    save_results("faster_rcnn", acc, f1, loss, train_time)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "faster_rcnn.pth"))


if __name__ == "__main__":
    main()