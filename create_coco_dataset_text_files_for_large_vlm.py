import os, glob
from shutil import copy

src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017_orig_for_large_vlsm/'
root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/'
classes = os.listdir(src)

fold = ['val/']#['val_on_train/', 'val/', 'train/']
file_list = []
for split in fold:
    path = root+split
    for cl in os.listdir(path):
        f_list = os.listdir(path+cl)
        for file in f_list:
            with open(root+split+'.txt', 'a') as f:
                f.write(split+cl+'/'+file+'\n')

