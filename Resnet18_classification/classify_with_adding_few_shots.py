from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

dataset_dir = "/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/s_cars_ood_adding_ood/"

train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_ind_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_train = torchvision.datasets.ImageFolder(root=dataset_dir+"test_2add", transform = train_tfms)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size = 5, shuffle=True, num_workers = 4)

dataset_test_ood = torchvision.datasets.ImageFolder(root=dataset_dir+"test", transform = test_tfms)
testloader_ood = torch.utils.data.DataLoader(dataset_test_ood, batch_size = 32, shuffle=False, num_workers = 4)
#
dataset_test_ind = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_ind_tfms)
testloader_ind = torch.utils.data.DataLoader(dataset_test_ind, batch_size = 32, shuffle=False, num_workers = 4)

# def add_samples_to_train_dataset(dataset, dataset3, samples_to_add):
#     new_indices = random.sample(range(len(dataset3)), samples_to_add)
#     new_samples = [dataset3.samples[i] for i in new_indices]  # Note the use of `samples` attribute
#     dataset.samples.extend(new_samples)
#     dataset.targets.extend([dataset3.targets[i] for i in new_indices])
#     return dataset  # Return the updated dataset


def train_model(trainloader, testloader_ind, testloader_ood, model, criterion, optimizer, scheduler,
                n_epochs=5):
    losses = []
    train_accuracies = []
    test_ood_accuracies = []
    test_ind_accuracies = []
    model = model.to(device)
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        model.train()
        for i, data in enumerate(tqdm(trainloader, desc="Training few shots on ood", leave=False)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * running_correct / len(trainloader)
        print(
            f"Epoch {epoch + 1}, duration: {epoch_duration:.2f} s, loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.2f}")

        losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        model.eval()
        test_ood = eval_model(model, testloader_ood)
        test_ood_accuracies.append(test_ood)
        test_ind = eval_model(model, testloader_ind)
        test_ind_accuracies.append(test_ind)

        scheduler.step(test_ood)
        since = time.time()

        # # Add samples to train_dataset after each epoch
        # dataset = add_samples_to_train_dataset(dataset, dataset3, 5)  # Update the dataset
        # trainloader = DataLoader(dataset, batch_size=32,
        #                          shuffle=True)  # Reinitialize the DataLoader with the updated dataset

    print('Finished Training')
    return model, losses, train_accuracies, test_ood_accuracies, test_ind_accuracies


def eval_model(model, testloader_ood):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader_ood, desc="test", leave=False)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100.0 * correct / total
    print(f'Test acc: {test_acc:.2f}')
    return test_acc



model_ft = torch.load('resten18_trained_on_ind_S_cars.pt')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 196)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.01,momentum=0.9)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

model, losses, train_accuracies, test_ood_accuracies, test_ind_accuracies = train_model(trainloader, testloader_ind, testloader_ood, model_ft, criterion, optimizer, lrscheduler,n_epochs=5)