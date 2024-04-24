import os
import shutil
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = '/Data/federated_learning/large_vlm_distillation_ood/data/cifar-10-batches-py/'
dest = '/Data/federated_learning/large_vlm_distillation_ood/data/cifar10/images/'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
im_w = 32
im_h = 32
num_ch = 3  # Red, Green, Blue
cifar_refs = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
# Reshape the flattened array into a 3D tensor

file_index = 0
for ind, file in enumerate(os.listdir(root)):
    dict = unpickle(root+file)
    data = dict[b'data']
    labels = dict[b'labels']
    #reshaped_image = cifar_data.reshape((image_height, image_width, num_channels))

    for file, label in zip(data, labels):
        file = file.reshape((im_h,im_w,num_ch), order='F')
        file = file.astype(np.uint8)
        img = Image.fromarray(file)
        fold_name = str(label)+'_'+str(cifar_refs[label]+'/')
        if not os.path.exists(dest+fold_name):
            os.mkdir(dest+fold_name)
        img.save(dest+fold_name+str(cifar_refs[label])+'_'+str(file_index)+'.png')

        file_index+=1

