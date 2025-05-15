import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- CONFIG ---------------------- #
DATA_DIR = "data/chest_xray"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- DATASET LOADING ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# -------------------- MODEL SETUP ------------------- #
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- TRAINING LOOP ----------------- #
def train_model():
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {correct/total:.4f}")

    torch.save(model.state_dict(), "models/cnn_pneumonia_resnet18.pth")

# -------------------- EVALUATION -------------------- #
def evaluate_model(loader, set_name="Test"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n{set_name} Set Evaluation:")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=train_set.classes))

# -------------------- MAIN ENTRY -------------------- #
if __name__ == "__main__":
    print(f"Training on device: {DEVICE}")
    train_model()
    evaluate_model(test_loader)
