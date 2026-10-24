"""Fonctions communes : chemins, métriques, boucle d'entraînement/évaluation."""
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
for _d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def classification_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", labels=list(range(10)), zero_division=0)
    return acc, f1


def save_results(name, acc, f1, loss, train_time):
    data = {
        "model": name,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "loss": None if loss is None else float(loss),
        "train_time_s": float(train_time),
    }
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Résultats sauvegardés dans {path}")


def run_classification(name, model, train_loader, test_loader, optimizer, device, num_epochs):
    """Entraîne puis évalue un classifieur 10 classes. Sauvegarde les métriques en JSON."""
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"[{name}] Epoch [{epoch + 1}/{num_epochs}], Loss: {running / len(train_loader):.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - start
    print(f"[{name}] Temps d'entraînement : {train_time:.2f} s")

    model.eval()
    all_preds, all_labels, test_loss = [], [], 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    test_loss /= len(test_loader)

    acc, f1 = classification_metrics(all_labels, all_preds)
    print(f"[{name}] Accuracy: {acc * 100:.2f}% | F1: {f1:.4f} | Loss: {test_loss:.4f}")
    save_results(name, acc, f1, test_loss, train_time)
    return acc, f1, test_loss, train_time