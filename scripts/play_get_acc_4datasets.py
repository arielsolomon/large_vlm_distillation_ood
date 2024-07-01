import os, glob
import pandas as pd

root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/'
paths = ['checkpoint_Food101_img_mse','checkpoint_CUB_img_mse','checkpoint_Flower102_img_mse','checkpoint_SUN397_img_mse']
file_name = 'log.txt'
def get_max_acc(path,file_name):
    df = pd.read_csv(path+file_name, sep='	')
    val_acc = max(df['Valid Acc.'])
    return path, val_acc

df_max_ac = pd.DataFrame(columns=['dataset', 'accuracy'])
rows = []
for path in paths:

    path, acc = get_max_acc(root+path+'/',file_name)
    row = pd.DataFrame({'dataset': [path.split('/')[-2].split('_')[1]], 'accuracy': [acc]})
    rows.append(row)
df_max_ac = pd.concat(rows, ignore_index=True)
df_max_ac['accuracy'] = df_max_ac['accuracy'].round(2)
df_max_ac.to_csv('/home/user1/ariel/fed_learn/large_vlm_distillation_ood/paper_val_accuracies.csv')