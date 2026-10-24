import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Custom Dataset for Faster R-CNN on MNIST
class MNISTDetectionDataset(Dataset):
    def __init__(self, root, train=True):
        # Load base MNIST data
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img_tensor = F.to_tensor(img)

        # Faster R-CNN expects 3-channel images
        img_tensor = img_tensor.repeat(3, 1, 1)

        # Extract bounding box from non-zero pixels
        pos = torch.where(img_tensor[0] > 0)
        if pos[0].numel() == 0:  # Handle completely blank images just in case
            xmin, ymin, xmax, ymax = 0, 0, 28, 28
        else:
            xmin = torch.min(pos[1]).item()
            xmax = torch.max(pos[1]).item()
            ymin = torch.min(pos[0]).item()
            ymax = torch.max(pos[0]).item()

            # Ensure valid box dimensions
            if xmin == xmax: xmax += 1
            if ymin == ymax: ymax += 1

        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Shift labels by 1 because 0 represents the background in Faster R-CNN
        labels = torch.as_tensor([label + 1], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img_tensor, target


# Collate function required for handling dictionaries in object detection dataloaders
def collate_fn(batch):
    return tuple(zip(*batch))


batch_size = 8  # Kept small to prevent OOM errors on limited GPUs
train_dataset = MNISTDetectionDataset(root='./data', train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 2. Faster R-CNN Model Setup
# Load a pre-trained model for fine-tuning
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Replace the classifier with a new one for MNIST (10 digits + 1 background = 11 classes)
num_classes = 11
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)

# 3. Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 4. Training Loop
num_epochs = 3
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {losses.item():.4f}")

train_time = time.time() - start_time
print(f"Total Faster R-CNN Training Time: {train_time:.2f} seconds")