import os, glob

splits = ['train/','val/','val_on_train/']
root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/'
for split in splits:
    path = root+split+'5_traffic_light/'
    file_list = glob.glob(os.path.join(path, '*.jpg'))
    if split=='train':

        for file in file_list:
            os.replace(file, '/'.join(file.split('/')[:-1]) + '/' + 'traffic_light_' + file.split('/')[-1].split('_')[1])

    for file in file_list:
        os.rename(file, '/'.join(file.split('/')[:-1])+'/'+'traffic_light_'+file.split('/')[-1].split('_')[1])