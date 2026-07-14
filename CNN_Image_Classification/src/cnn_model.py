import torch
import torch.nn as nn


class CNN(nn.Module) :
    def __init__(self) :
        super(CNN, self).__init__()

        # Convolution Block 1

        self.conv1 = nn.Conv2d(
        3,
        16,
        kernel_size=3,
        padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        # Convolution Block 2

        self.conv2 = nn.Conv2d(
        16,
        32,
        kernel_size=3,
        padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        # Convolution Block 3
        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)


        # POOLING

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # FULLY CONNECTED LAYERS

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

#------------------------------------------
#          MODEL INITIALIZATION
#------------------------------------------


import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import train_loader, test_loader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


# CREATING CNN MODEL

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# TRAINING LOOP

for epoch in range(num_epochs):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss/len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )

    

# =====================================================
# Save Model
# =====================================================
import os

os.makedirs("../models", exist_ok=True)

torch.save(model.state_dict(), "../models/saved_cnn_model.pth")

print("\nModel saved successfully!")


# =====================================================
# Final Evaluation
# =====================================================

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# =====================================================
# Classification Report
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# =====================================================
# Training & Validation Accuracy
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    test_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# =====================================================
# Training & Validation Loss
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    test_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()