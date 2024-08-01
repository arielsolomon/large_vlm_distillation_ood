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
model_paths = ['/home/user1/ariel/fed_learn/large_vlm_distillation_ood/31_07_resnet50_distiled_non_fine_tuned_clip_3_losses.pth',
              '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet50_distilation_3_losses_against_fine_tuned_clip_31_07.pth',
              '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/naive_resnet50_on_s_cars.pt']
# Sweep configuration
sweep_config = {
    'method': 'random',
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'lr': {
            'values': [0.01, 0.001, 0.0001]
        },
        'momentum': {
            'values': [0.7, 0.8, 0.9]
        },
        'model_path': {
            'values': model_paths
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="classify_r50_dist_against_non_finetuned_CLIP_sweep")

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

dataset_train = torchvision.datasets.ImageFolder(root=dataset_dir+"test_to_add/", transform=train_tfms)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
dataset_test_ood = torchvision.datasets.ImageFolder(root=dataset_dir+"test/", transform=test_tfms)
testloader_ood = torch.utils.data.DataLoader(dataset_test_ood, batch_size=50, shuffle=True, num_workers=4)
dataset_test_ind = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform=test_tfms)
testloader_ind_on_train = torch.utils.data.DataLoader(dataset_test_ind, batch_size=186, shuffle=True, num_workers=4)

def train_model(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config


        model_path = config.model_path
        if config.model_path == '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/naive_resnet50_on_s_cars.pt':
            model= torch.load(config.model_path)
        else:
            model_data = torch.load(config.model_path)
            model_state_dict = model_data['state_dict']
            model = resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 768)
            model.load_state_dict(model_state_dict)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

        lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.9)
        n_epochs = 40

        losses, train_ood_accuracies, test_ood_accuracies, test_ind_on_train_accuracies = [], [], [], []

        for epoch in range(n_epochs):
            model.eval()
            test_ood_acc = eval_model(model, testloader_ood, 'test_on_ood')
            test_ood_accuracies.append(test_ood_acc)
            wandb.log({"test_ood_acc": test_ood_acc})

            test_ind_on_train_acc = eval_model(model, testloader_ind_on_train, 'test_ind_on_train')
            test_ind_on_train_accuracies.append(test_ind_on_train_acc)
            wandb.log({"test_ind_on_train_acc": test_ind_on_train_acc})

            since = time.time()
            running_loss, running_correct = 0.0, 0.0
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
            epoch_acc = 100 / 5 * running_correct / len(trainloader)
            print(f"\nEpoch {epoch + 1}, duration: {epoch_duration:.2f} s, OOD_train_loss: {epoch_loss:.4f}, ood_train acc: {epoch_acc:.2f}")
            wandb.log({"epoch": epoch + 1, "train_ood_loss": epoch_loss, "train_ood_acc": epoch_acc})
            losses.append(epoch_loss)
            train_ood_accuracies.append(epoch_acc)
            mkdir(output_dir + 'fewshots_non_finetuned_CLIP_10samples_step')
            torch.save(model, "{}model_{}.pt".format(output_dir + 'fewshots_non_finetuned_CLIP', epoch))

            lrscheduler.step(test_ood_acc)

        print('Finished Training')

        df_train_ood_acc = pd.DataFrame(train_ood_accuracies)
        df_test_ood_accuracies = pd.DataFrame(test_ood_accuracies)
        df_test_ind_accuracies = pd.DataFrame(test_ind_on_train_accuracies)
        df_train_ood_acc.to_csv('s_cars_few_shot_train/train_ood_acc.csv', header=None)
        df_test_ood_accuracies.to_csv('s_cars_few_shot_train/test_ood_acc.csv', header=None)
        df_test_ind_accuracies.to_csv('s_cars_few_shot_train/test_ind_acc.csv', header=None)
        return model, losses, train_ood_accuracies, test_ood_accuracies, test_ind_on_train_accuracies

def eval_model(model, dataloader, name):
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for data in tqdm(dataloader, desc="test", leave=False):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100.0 * correct / total
    print(f'\n{name}: {test_acc:.2f}')
    return test_acc

# Run the sweep
wandb.agent(sweep_id, function=train_model)
