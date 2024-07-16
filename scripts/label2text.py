import os, glob
import pandas as pd

root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/s_cars_ood_ind/'
dst  = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/s_cars_ood_ind_fixed/label2text.txt'

df_names = pd.read_csv(root+'names.csv', header=None)
df_names.index.name = 'Index'
df_names = df_names.reset_index()
df_names['Index'] = df_names['Index']+1
df_names = df_names[[0, 'Index']]
df_names.columns = ['Class','Class_num']
for index, row in df_names.iterrows():
    with open(dst, 'a') as f:
        # f.write(str(row['Class_num'])+'_'+row['Class']+' '+str(row['Class_num'])+' '+str(row['Class_num'])+'_'+row['Class']+'\n')
        f.write(row['Class'].replace(' ', '_') + ' ' + str(row['Class_num']) + ' ' + row['Class'].replace(' ', '_')+ '\n')