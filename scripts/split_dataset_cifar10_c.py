import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.image as mat_img

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def scan_dict(dict, dest):
    for item in list(dict.keys()):
        path_name = str(item)+'_'+dict[item]
        make_path(dest+path_name)


root = '/Data/federated_learning/data/cifar10_c/'
dest = '/Data/federated_learning/large_vlm_distillation_ood/cifar10/val/'

n = 500
val = 'val'

filename = "val.txt"
img_files = np.load(root+'gaussian_noise.npy')
label_files = np.load(root+'labels.npy')
cifar_refs = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
scan_dict(cifar_refs, dest)
dir_list = os.listdir(dest)
choices = np.random.choice(img_files.shape[0], size=500)

for dir in dir_list:
    for choice in choices:
        name = cifar_refs[label_files[choice]]+'_'+str(choice)
        if name.split('_')[0]==dir.split('_')[1]:
            mat_img.imsave(dest+dir+'/'+name+'.jpg', img_files[choice,:,:,:])
            with open(dest + '/' + filename, 'a') as f:
                f.write('val/'+dir+'/'+name+'.jpg'+'\n')
        else:
            pass


