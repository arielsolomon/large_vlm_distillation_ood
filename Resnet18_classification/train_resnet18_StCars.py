import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from stanfordCars_dataloader import Scars



root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/Stanfordcars/'


def main(root):

    # Paths to the datasets
    train_datapath, test_datapath = root+'train_set/', root+'test_set/'

    train_labels, test_labels = root+'anno_train.csv', root+'anno_test.csv'




    BATCH_SIZE = 8
    LEARNING_RATE = 0.1
    EPOCHS = 100

    # Transformations
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # Transformations
    train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print()
    # Loading StanCar dataset
    train_dataset = Scars(data_path=train_datapath, labels_path=root+'anno_train.csv', transform=train_tfms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    test_dataset = Scars(data_path=test_datapath, labels_path=root+'anno_test.csv', transform=test_tfms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)




    # Initialize ResNet18 model
    # model = resnet18(pretrained=False, num_classes=196)
    # model = model#.cuda()

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 196)
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
            inputs, targets = inputs.cuda(),  targets.cuda()

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

    def test(model, loader, criterion, epoch):
        model = model.cuda()
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

        print(f'Test Epoch: {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    # Training and testing the model
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []


    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion, epoch)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        scheduler.step()
    print(f'\nBest test accuracy : is {max(test_accuracies)}, \n\n')



    # Plotting the results



if __name__ == '__main__':
    main(root)

