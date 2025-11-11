# src/models/train_feedback_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load preprocessed mudra data
X = np.load("data/preprocessed/X.npy")
y = np.load("data/preprocessed/y.npy")

# Simulate deviation-based feedback labels (example: 0=good, 1=thumb adjust, 2=finger straighten)
# In real implementation, you would label some samples manually
feedback_labels = np.random.randint(0, 3, size=len(y))

X_train, X_test, y_train, y_test = train_test_split(X, feedback_labels, test_size=0.2)

class FeedbackNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

model = FeedbackNet(X.shape[1], 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

for epoch in range(30):
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/30 | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/feedback_model.pth")
print("✅ Feedback model saved.")
