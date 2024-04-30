import os
import shutil
from pathlib import Path
import argparse
import glob

''' step 1: scan image dataset
    step 2: for n images copy to val, val_train, open labels for val, val_train, train and put in the form:
    train/class_name/image_name'''
root = '/Data/federated_learning/large_vlm_distillation_ood/data/cifar10/images/'

n = 500
t_v_vt = ['train','val','val_on_train']
def get_file_list(path, file_type):
    return(glob.glob(os.path.join(path, '*'+file_type)))
dir_list = os.listdir(root)
filenames = ["train.txt", "val.txt", "val_train.txt"]
for cls in t_v_vt:
    if not os.path.exists(root[:-7] + cls):
        os.mkdir(root[:-7] + cls)

for dir in dir_list:
    class_path = root+dir+'/'
    for item in t_v_vt:
        if not os.path.exists(class_path.replace('images/',item+'/')):
            os.mkdir(class_path.replace('images/',item+'/'))
    file_list = get_file_list(class_path, 'jpg')
    val_files, val_on_train_files,train_files = file_list[:n], file_list[n:n+n], file_list[n+n::]
    for file1 in val_files:
        shutil.copyfile(file1, class_path.replace('images/',t_v_vt[1]+'/')+file1.split('/')[-1])
        with open(root[:-7]+t_v_vt[1]+'/'+filenames[1], 'a') as f1:
            f1.write(t_v_vt[1]+file1.split('images')[-1]+'\n')
    for file2 in val_on_train_files:
        shutil.copyfile(file2, class_path.replace('images/', t_v_vt[2] + '/') + file2.split('/')[-1])
        with open(root[:-7]+t_v_vt[2]+'/'+filenames[2], 'a') as f2:
            f2.write(t_v_vt[2]+file2.split('images')[-1]+'\n')
    for file in train_files:
        shutil.copyfile(file, class_path.replace('images/', t_v_vt[0] + '/') + file.split('/')[-1])
        with open(root[:-7]+t_v_vt[0]+'/'+filenames[0], 'a') as f:
            f.write(t_v_vt[0]+file.split('images')[-1]+'\n')
