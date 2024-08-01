import pandas as pd
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
import wandb

wandb.init(project="classify_r50_dist_against_non_finetuned_CLIP_sweep")  # Replace with your project name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

dataset_dir = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/s_cars_ood_ind_test_test_val/"
output_dir = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/Exp/"
mkdir(output_dir)
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
# training is done with 5 samples of each class of the ood data
dataset_train = torchvision.datasets.ImageFolder(root=dataset_dir+"test_to_add/", transform = train_tfms)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size = 1, shuffle=True, num_workers = 4)
# inference is done with same classes, different instances
dataset_test_ood = torchvision.datasets.ImageFolder(root=dataset_dir+"test/", transform = test_tfms)
testloader_ood = torch.utils.data.DataLoader(dataset_test_ood, batch_size = 50, shuffle=True, num_workers = 4)
#test on train is done on ind data to make sure that the model as saved reconginzes ind data

dataset_test_ind = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = test_tfms)
testloader_ind_on_train = torch.utils.data.DataLoader(dataset_test_ind, batch_size = 186, shuffle=True, num_workers = 4)

sweep_config = {
    'method': 'random'
    }
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'lr': {
        'values': [0.01, 0.001, 0.001]
        },
    'momentum': {
          'values': [0.7, 0.8, 0.9]
        },

    }





def train_model(trainloader, testloader_ood,testloader_ind_on_train,  model, criterion, optimizer, lrscheduler,
                n_epochs=5):
    losses = []
    train_ood_accuracies = []
    test_ood_accuracies = []
    test_ind_on_train_accuracies = []
    model = model.to(device)
    for epoch in range(n_epochs):
        model = model.to(device)
        model.eval()
        name = 'test_on_ood'
        test_ood = eval_model(model, testloader_ood, name)
        test_ood_accuracies.append(test_ood)
        wandb.log({"test_ood_acc": test_ood})
        name2 = 'test_ind_on_train'
        test_ind_on_train = eval_model(model, testloader_ind_on_train, name2)
        test_ind_on_train_accuracies.append(test_ind_on_train)
        wandb.log({"test_ind_on_train": test_ind_on_train})

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
        epoch_acc = epoch_acc = 100/5*running_correct/len(trainloader)# 100 * running_correct / len(trainloader)
        print(
            f"\nEpoch {epoch + 1}, duration: {epoch_duration:.2f} s, OOD_train_loss: {epoch_loss:.4f}, ood_train acc: {epoch_acc:.2f}")
        wandb.log({"epoch": epoch + 1, "train_ood_loss": epoch_loss, "train_ood_acc": epoch_acc})
        losses.append(epoch_loss)
        train_ood_accuracies.append(epoch_acc)
        mkdir(output_dir+'fewshots_non_finetuned_CLIP_10samples_step')
        torch.save(model, "{}model_{}.pt".format(output_dir+'fewshots_non_finetuned_CLIP',epoch))
        # model.eval()
        # name = 'test_on_ood'
        # test_ood = eval_model(model, testloader_ood, name)
        # test_ood_accuracies.append(test_ood)
        # wandb.log({"epoch": epoch + 1, "test_ood_acc":  test_ood})
        # name2 = 'test_ind_on_train'
        # test_ind_on_train = eval_model(model, testloader_ind_on_train,name2)
        # test_ind_on_train_accuracies.append(test_ind_on_train)
        # wandb.log({"test_ind_on_train": test_ind_on_train})


        lrscheduler.step(test_ood)
        since = time.time()

    print('Finished Training')
    return model, losses, train_ood_accuracies, test_ood_accuracies, test_ind_on_train_accuracies


def eval_model(model, testloader_ood, name):
    correct = 0.0
    total = 0.0
    testloader_ood = testloader_ood
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
    print(f'\n{name}: {test_acc:.2f}')
    return test_acc


model_path = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/31_07_resnet50_distiled_non_fine_tuned_clip_3_losses.pth'
model_dist = torch.load(model_path)
model_dist_dist = model_dist['state_dict']
model = resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 768)
model.load_state_dict(model_dist_dist)
load_status = model.load_state_dict(model_dist_dist)
print("Missing keys:", load_status.missing_keys)
print("Unexpected keys:", load_status.unexpected_keys)

print(f'\nmodel ft, resnet50_native_train_epoch8_best_model was loaded\n')
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 196)
model_ft = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.01,momentum=0.9)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
n_epochs = 40
model, losses, train_ood_accuracies, test_ood_accuracies, test_ind_accuracies = train_model(trainloader, testloader_ood,testloader_ind_on_train,  model_ft, criterion, optimizer, lrscheduler,
                n_epochs)
df_train_ood_acc = pd.DataFrame(train_ood_accuracies)
df_test_ood_accuracies = pd.DataFrame(test_ood_accuracies)
df_test_ind_accuracies = pd.DataFrame(test_ind_accuracies)
df_train_ood_acc.to_csv('s_cars_few_shot_train/train_ood_acc.csv', header=None)
df_test_ood_accuracies.to_csv('s_cars_few_shot_train/test_ood_acc.csv', header=None)
df_test_ind_accuracies.to_csv('s_cars_few_shot_train/test_ind_acc.csv', header=None)