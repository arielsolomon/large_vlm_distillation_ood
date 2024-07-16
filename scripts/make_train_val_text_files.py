import os, glob
import pandas as pd
import numpy as np



src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/s_cars_ood_ind_fixed/'

dst = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/data/StanfordCars/'
names_processed = []
names_df = pd.read_csv(src+'names.csv', header=None)
names = sorted(list(names_df.iloc[:,0]))
names_processed.append([name.replace(' ', '_') for name in names])
names_processed = names_processed[0]
indices = list(np.arange(1, names_df.shape[0]+1, 1))
dict_names = {k:v for k,v in zip(names_processed,indices)}
train_list = sorted(os.listdir(src+'train/'))
val_list = sorted(os.listdir(src+'val/'))
val_on_train_list = sorted(os.listdir(src+'val_on_train/'))

def write_txts(tts_list, dict_names, dst, src,tts):
    with open(dst+'val.txt', 'w') as f:
        for fold in tts_list:
            for file in os.listdir(src+tts+fold):
                ind = dict_names[fold]
                row = f'{tts}{fold}/{file}'
                f.write(row+'\n')
write_txts(val_list, dict_names, dst, src, 'val/')