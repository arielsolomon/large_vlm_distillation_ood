import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image


root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet16_cifar_classification/data/'
CIFAR10_C_DATA_PATH = root + 'cifar10_c/CIFAR-10-C/'
corrupted_classes = [cls[:-4] for cls in os.listdir(CIFAR10_C_DATA_PATH)]

class CorruptedCIFAR10C(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        """
        Initializes the CorruptedCIFAR10C dataset class.

        Args:
            data_path (str): Path to the corrupted data file (.npy).
            labels_path (str): Path to the corrupted labels file (.npy).
            transform (torchvision.transforms, optional): Transformations to apply to the data. Defaults to None.
        """
        self.data = np.load(data_path)
        self.targets = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset (number of data samples).
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
def main(c_class):
    print('\nFor corruption data: ', c_class.split('/')[-1][:-4], '\n\n')
    root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_cifar_classification/data/'
    # Paths to the datasets
    CIFAR10_PATH = root + 'cifar10/'

    CIFAR10_C_LABELS_PATH = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_cifar_classification/data/cifar10_c/labels.npy'
    CIFAR10_C_DATA_PATH = c_class
    print(CIFAR10_C_DATA_PATH)



    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    EPOCHS = 25

    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Loading CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=False, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # testc_dataset = CorruptedCIFAR10C(data_path=CIFAR10_C_DATA_PATH, labels_path=CIFAR10_C_LABELS_PATH,
    #                                  transform=transform_test)
    # testc_loader = DataLoader(testc_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    # Define test loader for CIFAR-10 test data
    test_dataset = datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # test_dataset = datasets.CIFAR10(root=CIFAR10_C_DATA_PATH, train=False, download=False, transform=transform_test)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Custom dataset class for CIFAR-10-C
    class CIFAR10C(Dataset):
        def __init__(self, data_path, labels_path, transform=None):
            self.data = np.load(data_path)
            self.targets = np.load(labels_path)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    # Load CIFAR-10-C dataset
    cifar10c_dataset = CIFAR10C(data_path=CIFAR10_C_DATA_PATH, labels_path=CIFAR10_C_LABELS_PATH,
                                transform=transform_test)
    cifar10c_loader = DataLoader(cifar10c_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize ResNet18 model
    model = resnet18(pretrained=False, num_classes=10)
    model = model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training function
    def train(model, loader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_epoch_loss = running_loss / len(loader)
        train_epoch_acc = 100. * correct / total

        print(f'Train Epoch: {epoch} | Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc:.2f}%')

        return train_epoch_loss, train_epoch_acc

    # Testing function
    def testc(model, loader, criterion, epoch, dataset_name='Test'):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total

        print(f' Test corrupted Epoch: {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    def test_cifar10(model, loader, criterion, epoch):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total

        print(f'Test CIFAR-10 Epoch: {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    # Training and testing the model
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    cifar10c_losses, cifar10c_accuracies = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test_cifar10(model, test_loader, criterion, epoch)
        cifar10c_loss, cifar10c_acc = testc(model, cifar10c_loader, criterion, epoch, dataset_name='CIFAR10-C')

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        cifar10c_losses.append(cifar10c_loss)
        cifar10c_accuracies.append(cifar10c_acc)

        scheduler.step()
    corrupt_cls = c_class.split('/')[-1][:-4]
    print(f'\nBest test accuracy in corruption data: , {corrupt_cls}, is {max(test_accuracies)}, \n\n')

    with open(root[:-5]+'corruption_accuracies_23_06_add_test.txt','a') as f:
        f.write(f'corruption accuracy, {corrupt_cls}, test_c accuracies, {max(cifar10c_accuracies)}, train accuracies, {max(train_accuracies)}'
                f', test_accuracies, {max(test_accuracies)} \n')

    # Plotting the results
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.plot(epochs, cifar10c_losses, label='CIFAR10-C Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, cifar10c_accuracies, label='CIFAR10-C Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.savefig(str(corrupt_cls)+'23_06_add_test.png')


for c_class in corrupted_classes:
    c_class = CIFAR10_C_DATA_PATH+c_class+'.npy'
    main(c_class)
