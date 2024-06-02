import os, glob
from shutil import copy

src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017_orig_for_large_vlsm/'
root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/'
classes = os.listdir(src)

fold = ['val_on_train/', 'train/', 'val/']

# want to do:
    # list train images
    # copy 200 from train to val_on_train
    # list the remaining train images to train.txt
    # list the val_on_train to val_on_train.txt
def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

file_train, file_val_on_train = [], []
for folder in fold:
    mk_dir(root+folder)
    path = root+folder
    for item in classes:


        cls = item.split('_')[-1]
        img_list = glob.glob(os.path.join(src+item+'/coco-2017/train/data/', '*.jpg'))
        for img in img_list[:200]:
            new_img = root+item+'/'+fold[0]+img.split('/')[-1]
            file_val_on_train.append(new_img)

            #copy(img, root+fold[0]+item+'/')
            img_list.remove(img)

        for image in img_list:
            #copy(image, root+'/train/'+item)
            file_train.append(image)

for val_img in  file_val_on_train:
    with open(root+'/val_on_train.txt', 'a') as f:
        f.write(fold[0]+item+'/'+val_img.split('/')[-1]+'\n')
for train_img in file_train:
    with open(root+'/train.txt', 'a') as f:
        f.write(fold[1]+item+'/'+train_img.split('/')[-1]+'\n')



