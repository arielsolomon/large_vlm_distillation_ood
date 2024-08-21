import os
import glob

def del_files(img_path, lbl_path):
    counter = 0
    img_list = glob.glob(os.path.join(img_path, '*.jpg'))
    lbl_list = [os.path.basename(lbl)[:-4] for lbl in glob.glob(os.path.join(lbl_path, '*.txt'))]  # Extract filenames without extensions
    for file in img_list:
        f_name = os.path.basename(file)[:-4]  # Extract filename without extension
        if f_name not in lbl_list:
            os.remove(img_path+f_name+'.jpg')
            counter += 1
    print(f'\nNum of files not in labels: {counter}')

# Load data
src = '/work/large_vlm_distillation_ood/datasets/coco/coco/' # Adjust path if needed
images = os.path.join(src, 'images/val/')
labels = os.path.join(src, 'labels/val/')

del_files(images, labels)
