import glob, os
import numpy as np
import pandas as pd

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/s_cars_ood_ind_fixed/'

folds = ['train/', 'val_on_train/', 'val/']

for fold in folds:
    dir = src+fold
    dir_list = os.listdir(dir)
    for folder in dir_list:
        new_folder = folder.replace(' ', '_')
        try:
            os.rename(dir+folder, dir+new_folder)
        except:
            pass


