import pickle
import os
import glob
import numpy as np

root = '/Data/federated_learning/data/cifar10/cifar-10-batches-py/'
for fold in os.listdir(root):
    with open(os.path.join(root, fold), 'rb') as f:
      batch_data = pickle.load(f, encoding='latin1')  # Handle potential encoding issues
      # Access data and labels using keys like 'data' and 'labels'
      images = batch_data['data']
      labels = batch_data['labels']
