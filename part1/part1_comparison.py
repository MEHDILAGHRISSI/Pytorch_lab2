import torch
import time
from sklearn.metrics import f1_score, accuracy_score
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from part1_custom_cnn import CustomCNN
from part1_faster_rcnn import MNISTDetectionDataset, collate_fn

# 1. Setup Device & Shared Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

# CNN Test Loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
cnn_test_data = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)
cnn_test_loader = DataLoader(dataset=cnn_test_data, batch_size=batch_size, shuffle=False)

# R-CNN Test Loader
rcnn_test_data = MNISTDetectionDataset(root='../data', train=False)
rcnn_test_loader = DataLoader(rcnn_test_data, batch_size=8, shuffle=False, collate_fn=collate_fn)


# 2. Evaluation Functions
def evaluate_cnn(model, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    running_loss, all_preds, all_labels = 0.0, [], []

    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_time = time.time() - start_time
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro'), running_loss / len(
        test_loader), eval_time


def evaluate_rcnn(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    start_time = time.time()
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                if len(output['scores']) > 0:
                    best_idx = torch.argmax(output['scores'])
                    pred_label = output['labels'][best_idx].item() - 1  # Shift back from background
                else:
                    pred_label = -1
                all_preds.append(pred_label)
                all_labels.append(targets[i]['labels'][0].item() - 1)

    eval_time = time.time() - start_time
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds,
                                                           average='macro'), 0.0, eval_time  # Loss requires custom eval for R-CNN


if __name__ == '__main__':
    results = {}

    # Load and Evaluate Custom CNN
    cnn_model = CustomCNN().to(device)
    cnn_model.load_state_dict(torch.load('custom_cnn.pth', map_location=device))
    acc, f1, loss, _ = evaluate_cnn(cnn_model, cnn_test_loader)
    results['Custom CNN'] = {'Accuracy': acc, 'F1 Score': f1, 'Loss': loss}

    # Print Table
    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'Loss':<10}")
    print("-" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {metrics['Accuracy']:.4f}     | {metrics['F1 Score']:.4f}   | {metrics['Loss']:.4f}")