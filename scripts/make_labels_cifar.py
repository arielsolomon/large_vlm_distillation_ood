import os, glob

train_val = ['val/', 'train/','val_on_train/']
for item in train_val:

    root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/'+item
    dir_list = os.listdir(root)
    # for dir in dir_list:
    #     file_names = glob.glob(os.path.join(root+dir+'/', '*.jpg'))
    #     for file in file_names:
    #         old_name = file
    #         cls_name = file.split('/')[-2].split('_')[1]
    #         new_name = '/'.join(file.split('/')[:-1])+'/'+cls_name+'_'+file.split('/')[-1]
    #         os.rename(old_name, new_name)
    file_names = glob.glob(os.path.join(root+dir_list[2]+'/', '*.jpg'))
    for file in file_names:
        old_name = file
        cls_name = 'traffic_light'#file.split('/')[-2].split('_')[1]
        new_name = '/'.join(file.split('/')[:-1])+'/'+cls_name+'_'+file.split('/')[-1][8:]
        os.rename(old_name, new_name)
