import os
import shutil
from pathlib import Path
import argparse
import glob

''' step 1: scan image dataset
    step 2: for n images copy to val, val_train, open labels for val, val_train, train and put in the form:
    train/class_name/image_name'''
root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/'

n = 500
def get_file_list(path, file_type):
    return(glob.glob(os.path.join(path, '*'+file_type)))
dir_list = os.listdir(root)
filenames = ["train.txt",  "val.txt", 'val_on_train.txt']
file_list_train, file_list_val,file_list_val_on_train = [],[], []
for dir in dir_list:
    if dir=='train':

        class_path = root+dir+'/'
        for clas in os.listdir(class_path):
            file_list_train.append(get_file_list(class_path+clas, 'jpg'))
        file_list_train = sum(file_list_train, [])
    elif dir=='val':
        class_path = root+dir+'/'
        for clas in os.listdir(class_path):
            file_list_val.append(get_file_list(class_path+clas, 'jpg'))
        file_list_val = sum(file_list_val, [])
    else:
        class_path = root+dir+'/'
        for clas in os.listdir(class_path):
            file_list_val_on_train.append(get_file_list(class_path+clas, 'jpg'))
        file_list_val_on_train = sum(file_list_val_on_train, [])

for file in file_list_train:
    with open(root+"train.txt", 'a') as f:
        f.write('/'.join(file.split('/')[-3:])+'\n')

for file1 in file_list_val:
    with open(root+"val.txt", 'a') as f1:
        f1.write('/'.join(file1.split('/')[-3:])+'\n')

for file2 in file_list_val_on_train:
    with open(root+"val_on_train.txt", 'a') as f2:
        f2.write('/'.join(file2.split('/')[-3:])+'\n')
