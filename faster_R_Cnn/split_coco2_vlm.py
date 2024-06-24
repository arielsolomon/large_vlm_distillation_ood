import json, os, glob
from shutil import copy

img_path = '/Data/federated_learning/large_vlm_distillation_ood/faster_Rcnn/data/coco/images/'
lbl_path = '/Data/federated_learning/large_vlm_distillation_ood/faster_Rcnn/data/coco/annotations/'
dst = '/Data/federated_learning/large_vlm_distillation_ood/faster_Rcnn/data/coco_4_vlm/'
dir_list = os.listdir(img_path)
def dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
dir_exists(dst)
dir_exists(dst+'data/')

# view json:
with open(lbl_path+'instances_train2017.json') as f:
    coco_data = json.load(f)

image_id_to_file_name = {image['id']: image['file_name'] for image in coco_data['images']}

# Now you can match annotations to images using the image_id
annotations = coco_data['annotations']
coco_classes = {
    0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush'}

# want to do:
    #1. scrol image2file and get image id and image file name
    #2. according to image id, locate annotation for image
    #3. locate from annotation the image category
    #4. make directory at number of category if it not exists
    #5. copy image (image file name) to the created directory
    #6. open text file and append each in new line: "directory name(train)"/category#_category_name/category_name_imagefile_name

#1
for annot in annotations:
    img_id = annot['image_id']
    img_file = image_id_to_file_name[img_id]
    dir_exists(dst+dir_list[0])
    cat_num = annot['category_id']
    if cat_num >80:
        cat_num=0

    dir_exists(dst+dir_list[0]+'/'+str(cat_num)+'_'+coco_classes[cat_num])
    copy(img_path+dir_list[0]+'/'+img_file, dst+dir_list[0]+'/'+str(cat_num)+'_'+coco_classes[cat_num]+'/'+coco_classes[cat_num]+'_'+img_file)
    with open(dst+'data/train.txt','a') as f:
        f.write('train/'+str(cat_num)+'_'+coco_classes[cat_num]+'/'+coco_classes[cat_num]+'_'+img_file+'\n')



