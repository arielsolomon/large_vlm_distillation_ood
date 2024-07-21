import pandas as pd
import json
src = '/Data/federated_learning/large_vlm_distillation_ood/faster_Rcnn/s_cars_ood_ind/'
dst = '/Data/federated_learning/large_vlm_distillation_ood/faster_Rcnn/s_cars_ood_ind/'

anno_list = ['anno_train.csv','anno_val.csv','anno_val_on_train.csv']
#
#
#
# # def fill_anno(anno_list):
# #     for i, file in enumerate(anno_list):
# #         anno_dir = {'file_name': [], 'boxes': [], 'labels': [], 'folder_name': []}
# #         df = pd.read_csv(src+anno_list[i], header=None)
# #
# #         for _, row in df.iterrows():
# #             box = row[1:5].tolist()
# #             label = row[5]
# #             name = row[0]
# #             anno_dir['boxes'].append(box)
# #             anno_dir['label'].append(label)
# #             anno_dir['f_name'].append(name)
# #         with open(dst + file.replace('.csv', '.json'), 'w') as f:
# #             json.dump(anno_dir, f, indent=4)
# # fill_anno(anno_list)anno
#
#
#
# src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/s_cars_ood_ind/'
# dst ='/home/user1/ariel/fed_learn/large_vlm_distillation_ood/s_cars_ood_ind/'

anno_list = ['anno_train.csv', 'anno_val.csv', 'anno_val_on_train.csv']
df_names = pd.read_csv(src+'names.csv',header=None)
def fill_anno(anno_list, df_names):
    for file in anno_list:
        anno_dir = {'file_name': [], 'boxes': [], 'label': [], 'folder_name': []}
        df = pd.read_csv(src + file, header=None)

        for _, row in df.iterrows():
            file_name = row[0]
            box = row[1:5].tolist()
            label = row[5]
            folder_name = df_names.iloc[int(row[5])-1]  # Assuming the file names follow a pattern like 'anno_train.csv'


            anno_dir['file_name'].append(file_name)
            anno_dir['boxes'].append(box)
            anno_dir['label'].append(label)
            anno_dir['folder_name'].append(folder_name[0])
        file = file.replace('anno_', 'instances_')
        with open(dst + file.replace('.csv', '.json'), 'w') as f:
            json.dump(anno_dir, f, indent=4)

fill_anno(anno_list, df_names)

