import os, glob
import pandas as pd
from shutil import move, copy
import numpy as np
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def choices(dir):
    sample_num = len(os.listdir(dir))
    choices = np.random.choice(sample_num, size=round(sample_num*0.15))
    return list(choices)

root = '/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/s_cars_ood_orig/'
dest = '/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/s_cars_ood_ind/'
mkdir(dest)
mkdir(dest+'test_ind')
mkdir(dest+'train')

train_list_dir = os.listdir(root+'train/')

for fold in train_list_dir:
    mkdir(dest+'test_ind/'+fold)
    mkdir(dest+'train/'+fold)
df_anno_train = pd.read_csv(root+'anno_train.csv', header=None)
df_anno_test_ind = pd.DataFrame()

print(df_anno_train.head(3))

for fold in tqdm(train_list_dir):
    choice = choices(root+'train/'+fold)
    files = glob.glob(os.path.join(root+'train/'+fold,'*jpg'))
    selected_files = [files[i] for i in choice]
    new_train_files = [item for item in files if item not in selected_files]
    for file in selected_files:
        condition = file.split('/')[-1]
        try:
            copy(file, dest+'test_ind/'+fold)
            condition_met = df_anno_train[0] == condition
            rows_to_move = df_anno_train[condition_met]
            df_anno_test_ind = pd.concat([df_anno_test_ind, rows_to_move])
            df_anno_train = df_anno_train[~condition_met]
        except:
            pass
    for file_train in new_train_files:
        copy(file_train, dest+'train/'+fold)
df_anno_test_ind.reset_index(drop=True, inplace=True)
df_anno_train.reset_index(drop=True, inplace=True)
df_anno_test_ind.to_csv(dest+'anno_test_ind.csv')
df_anno_train.to_csv(dest+'anno_train.csv')
