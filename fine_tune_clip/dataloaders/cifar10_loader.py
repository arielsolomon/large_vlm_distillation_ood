import os.path
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import datasets
import numpy as np
import torchvision
class CIFAR10C(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, name: str,
                 transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + 'impulse_noise.npy')
        target_path = os.path.join(root, name +'impulse_noise_labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)
def load_data(root: str, batch_size=32, ood=False):
    """
    Load CIFAR-10 dataset.

    Parameters:
    root (str): Root directory where the dataset will be stored.
    batch_size (int, optional): Batch size for data loaders. Defaults to 32.
    ood (bool, optional): Flag indicating if the data is out-of-distribution (OOD). Defaults to False.

    Returns:
    Tuple[DataLoader, DataLoader]: DataLoader for the training set and DataLoader for the test set.
    """
    assert os.path.exists(root), f'{root} does not exist'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) if not ood else transforms.Compose(
        [transforms.ToTensor(), transforms.ColorJitter(contrast=0.5, brightness=1.0),
         transforms.Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))]
    )

    print(f'load_data to {root}, ood? {ood}')
    train_set = CIFAR10(root, train=True, download=False, transform=transform)
    test_set = CIFAR10(root, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, test_loader


def load_raw_data(root):
    train_set = CIFAR10(root, train=True, download=False)
    train_set, validation_set = train_val_split(train_set)
    test_set = CIFAR10(root, train=False, download=False)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return train_set, validation_set, test_set, classes

def load_raw_dataC(root):
    train_set = CIFAR10C(root, 'train_', transform=None, target_transform=None)
    validation_set = CIFAR10C(root, 'val_', transform=None, target_transform=None)
    test_set = CIFAR10C(root, 'test_', transform=None, target_transform=None)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return train_set, validation_set, test_set, classes


def train_val_split(train_set, split_ratio= 0.8, shuffle = True):
    from torch.utils.data import DataLoader, Subset
    dataset_size = len(train_set)
    split_index = int(dataset_size * split_ratio)
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)
    # Create Subset objects for the training and validation sets
    train_subset = Subset(train_set, indices[:split_index])
    validation_subset = Subset(train_set, indices[split_index:])
    return train_subset, validation_subset
