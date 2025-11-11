# src/models/train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json
import random

# ---------------- CONFIG ----------------
BASE_DIR = os.getcwd()
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

X_PATH = os.path.join(PREPROCESSED_DIR, "X.npy")
Y_PATH = os.path.join(PREPROCESSED_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(PREPROCESSED_DIR, "label_mapping.json")

AUGMENTATION = True          # Apply small noise to landmarks
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.001
EARLY_STOPPING_PATIENCE = 10
NOISE_STD = 0.01            # Standard deviation for Gaussian noise

# ---------------- REPRODUCIBILITY ----------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD DATA ----------------
X = np.load(X_PATH)
y = np.load(Y_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)

print(f"✅ Loaded preprocessed data | X shape: {X.shape}, y shape: {y.shape}")
print(f"Detected classes: {label_mapping}")

# ---------------- DATA SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ---------------- DATA AUGMENTATION ----------------
def augment_landmarks(batch):
    if AUGMENTATION:
        noise = np.random.normal(0, NOISE_STD, batch.shape)
        return batch + noise
    return batch

# ---------------- PYTORCH TENSORS ----------------
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# ---------------- MODEL ----------------
class MudraNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = MudraNet(input_dim, num_classes).to(device)
print(f"✅ Model created | Input dim: {input_dim}, Output classes: {num_classes}")

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING ----------------
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "mudra_model_best.pth")
best_acc = 0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    permutation = torch.randperm(X_train.size()[0]).to(device)
    epoch_loss = 0

    for i in range(0, X_train.size()[0], BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]

        # Apply small Gaussian noise for augmentation
        batch_x = torch.tensor(augment_landmarks(batch_x.cpu().numpy()), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, preds = torch.max(test_outputs, 1)
        acc = (preds == y_test).float().mean().item()

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {epoch_loss:.4f} | Test Accuracy: {acc:.4f}")

    # Early stopping
    if acc > best_acc:
        best_acc = acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("⏹️ Early stopping triggered!")
            break

print(f"💾 Training complete. Best test accuracy: {best_acc:.4f}")
print(f"Model saved to: {MODEL_PATH}")
