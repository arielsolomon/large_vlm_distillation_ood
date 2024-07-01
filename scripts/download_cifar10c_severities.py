import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.image as mat_img

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def scan_dict(dict, dest):
    for item in list(dict.keys()):
        path_name = str(item)+'_'+dict[item]
        make_path(dest+path_name)

def delete_content(directory):
  for root, directories, files in os.walk(directory):
    for file in files:
      # Construct the full path to the file
      file_path = os.path.join(root, file)
      # Delete the file
      os.remove(file_path)



root = '/home/user1/ariel/fed_learn/datasets/cifar10_c/'
dest = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/cifar10/val/'
if not os.path.exists(dest):
    os.mkdir(dest)

val = 'val'
val_dest = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/data/cifar10/'
filename = "val.txt"
def make_dataset(root, cor_type):
    img_files = np.load(root+cor_type)
    label_files = np.load(root+'labels.npy')
    cifar_refs = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
    scan_dict(cifar_refs, dest)
    dir_list = os.listdir(dest)

    # i prefere to do each corruption with 5 severities
    severities = {'sev1':10000,'sev2':20000,'sev3':30000,'sev4':40000,'sev5':50000}
    chosen_sev = list(severities.keys())[0]
    #number of files at each corruption severity:
    for dir in dir_list:
        path_to_clean = dest+dir
        delete_content(path_to_clean)
        for i in range(0,severities['sev1'],1):
            name = cifar_refs[label_files[i]]+'_'+str(chosen_sev)
            if name.split('_')[0]==dir.split('_')[1]:
                mat_img.imsave(dest+dir+'/'+name+'_'+str(i)+'.jpg', img_files[i,:,:,:])
                with open(val_dest + filename, 'a') as f:
                    f.write('val/'+dir+'/'+name+'_'+str(i)+'.jpg'+'\n')
            else:
                pass
cor_type = 'impulse_noise.npy'
make_dataset(root, cor_type)
