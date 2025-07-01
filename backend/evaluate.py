import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random

# -------------------- Configuration --------------------
data_dir = '../data'  # your dataset folder
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pth')
img_height, img_width = 224, 224
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Transform --------------------
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- Dataset Classes --------------------
class TeaLeafDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------- Load and Split Dataset --------------------
full_dataset = TeaLeafDataset(data_dir)
image_paths = full_dataset.images
labels = full_dataset.labels

class_images = defaultdict(list)
for img_path, label in zip(image_paths, labels):
    class_images[label].append(img_path)

train_paths, val_paths = [], []
train_labels, val_labels = [], []
random.seed(42)

for label, paths in class_images.items():
    random.shuffle(paths)
    val_count = max(1, int(0.2 * len(paths)))
    val_paths.extend(paths[:val_count])
    train_paths.extend(paths[val_count:])
    val_labels.extend([label] * val_count)
    train_labels.extend([label] * (len(paths) - val_count))

print(f"Total samples: {len(image_paths)}")
print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

# -------------------- Load Model --------------------
model = models.resnet50(weights=None)
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# -------------------- Evaluate --------------------
val_dataset = SubsetDataset(val_paths, val_labels, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# -------------------- Metrics --------------------
exclude_folders = {'testing diseased dataset', 'testing healthy leaves'}
class_names = [name for name in full_dataset.classes if name not in exclude_folders]
class_indices = [full_dataset.class_to_idx[name] for name in class_names]

# Only keep pairs where both y_true and y_pred are in class_indices
filtered_pairs = [
    (yt, yp) for yt, yp in zip(y_true, y_pred)
    if yt in class_indices and yp in class_indices
]
if filtered_pairs:
    filtered_y_true, filtered_y_pred = zip(*filtered_pairs)
else:
    filtered_y_true, filtered_y_pred = [], []

# Remap indices to 0...N-1 for metrics
index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
remapped_y_true = [index_map[y] for y in filtered_y_true]
remapped_y_pred = [index_map[y] for y in filtered_y_pred]

print("\nClass Distribution in Validation Set:")
for i, name in enumerate(class_names):
    count = remapped_y_true.count(i)
    print(f"{name}: {count} samples")

print("\nClassification Report:\n")
print(classification_report(remapped_y_true, remapped_y_pred, target_names=class_names, zero_division=0))

conf_matrix = confusion_matrix(remapped_y_true, remapped_y_pred, labels=list(range(len(class_names))))
print("\nConfusion Matrix:\n", conf_matrix)

accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print(f"\nAccuracy: {accuracy:.2f}")

print("\nPer-Class Accuracy:")
with np.errstate(divide='ignore', invalid='ignore'):
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, name in enumerate(class_names):
    acc = per_class_acc[i] if not np.isnan(per_class_acc[i]) else 0
    print(f"{name}: {acc * 100:.2f}%")

# -------------------- Confusion Matrix Plot --------------------
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))
plt.close()

report_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.txt')
report_lines = []

def log_and_print(msg):
    print(msg)
    report_lines.append(str(msg))

# -------------------- Model & Dataset Info --------------------
log_and_print(f"Model path: {model_path}")
log_and_print(f"Device used: {device}")
log_and_print(f"Image size: {img_height}x{img_width}")
log_and_print(f"Batch size: {batch_size}")
log_and_print(f"Number of classes: {len(full_dataset.classes)}")
log_and_print(f"Class names: {full_dataset.classes}")
log_and_print(f"\nTotal samples: {len(image_paths)}")
log_and_print(f"Training samples: {len(train_paths)}")
log_and_print(f"Validation samples: {len(val_paths)}")

log_and_print("\nClass Distribution in Validation Set:")
for i, name in enumerate(class_names):
    count = remapped_y_true.count(i)
    log_and_print(f"{name}: {count} samples")

log_and_print("\nClassification Report:\n")
try:
    report = classification_report(remapped_y_true, remapped_y_pred, target_names=class_names, zero_division=0)
    log_and_print(report)
except Exception as e:
    log_and_print(f"Error generating classification report: {e}")

conf_matrix = confusion_matrix(remapped_y_true, remapped_y_pred, labels=list(range(len(class_names))))
log_and_print("\nConfusion Matrix:\n" + np.array2string(conf_matrix))

accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix) if np.sum(conf_matrix) > 0 else 0
log_and_print(f"\nAccuracy: {accuracy:.2f}")

log_and_print("\nPer-Class Accuracy:")
with np.errstate(divide='ignore', invalid='ignore'):
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, name in enumerate(class_names):
    acc = per_class_acc[i] if not np.isnan(per_class_acc[i]) else 0
    log_and_print(f"{name}: {acc * 100:.2f}%")

if not remapped_y_true or not remapped_y_pred:
    log_and_print("\nWARNING: No valid samples in validation set after filtering. Check your dataset and class filtering.")

# Write all report lines to the file
with open(report_path, 'w', encoding='utf-8') as f:
    for line in report_lines:
        f.write(line + '\n')


