
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import pandas as pd
import os, glob
from PIL import Image
class Scars(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        """
        Initializes the CorruptedCIFAR10C dataset class.

        Args:
            data_path (str): Path to the corrupted data file (.npy).
            labels_path (str): Path to the corrupted labels file (.npy).
            transform (torchvision.transforms, optional): Transformations to apply to the data. Defaults to None.
        """

        self.data = sorted(glob.glob(os.path.join(data_path,'*.jpg')))
        self.targets =  pd.read_csv(labels_path, header=None).iloc[:,-1].values -1
        self.transform = transform or ToTensor()

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
        img = Image.open(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target



