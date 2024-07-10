import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_dir = "/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/s_cars_ood/"

train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_tfms)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle=True, num_workers = 2)

dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir+"test", transform = test_tfms)
testloader = torch.utils.data.DataLoader(dataset2, batch_size = 16, shuffle=False, num_workers = 2)

# Load the pre-trained ResNet18 model
resnet18 = models.resnet18(pretrained=True)

# Modify the final fully connected layer
num_classes = 196
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_classes)
resnet18.to(device)
# Print the modified model architecture
#print(resnet18)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_model(train_loader, val_loader, model, criterion, num_epochs, lr, optimizer_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    max_train_acc = 0.0
    max_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_losses = []
        train_accs = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the correct device
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            acc = accuracy(outputs, labels)
            train_accs.append(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        max_train_acc = max(max_train_acc, train_acc)

        # Validation Phase
        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to the correct device
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                acc = accuracy(outputs, labels)
                val_accs.append(acc.item())

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        max_val_acc = max(max_val_acc, val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return max_train_acc, max_val_acc


criterion, num_epochs, lr, optimizer = nn.CrossEntropyLoss(),25, 0.001, optim.SGD

train_model(trainloader, testloader, resnet18.to(device), criterion, num_epochs, lr, optimizer)