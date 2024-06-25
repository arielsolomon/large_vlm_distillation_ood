import json, os, glob
from shutil import copy

img_path = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/faster_R_Cnn/coco/images/'
lbl_path = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/faster_R_Cnn/coco/annotations/'
dst = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/faster_R_Cnn/coco_4_vlm/'
dir_list = os.listdir(img_path)
def dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
dir_exists(dst)
dir_exists(dst+'data/')

# view json:
for dir in dir_list[1:]:
    anot_path = lbl_path+'instances_'+dir[:-4]+'2017.json'
    with open(anot_path) as f:
        coco_data = json.load(f)

    image_id_to_file_name = {image['id']: image['file_name'] for image in coco_data['images']}

    # Now you can match annotations to images using the image_id
    annotations = coco_data['annotations']
    coco_classes_extracted = {}
    for i, item in enumerate(coco_data['categories']):
        coco_classes_extracted[item['id']]=item['name']
    old_coco_classes = {
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
    coco_classes = {k-1:v for k,v in old_coco_classes.items()}
    # want to do:
        #1. scrol image2file and get image id and image file name
        #2. according to image id, locate annotation for image
        #3. locate from annotation the image category
        #4. make directory at number of category if it not exists
        #5. copy image (image file name) to the created directory
        #6. open text file and append each in new line: "directory name(train)"/category#_category_name/category_name_imagefile_name

    #1
# num of categories in annotation file exceeds the known coco2017 class number(80), I will disregard annotations over known category number
    for annot in annotations:
        img_id = annot['image_id']
        img_file = image_id_to_file_name[img_id]
        dir_exists(dst+dir)
        cat_num = annot['category_id']
        dir_exists(dst+dir+'/'+str(cat_num)+'_'+coco_classes_extracted[cat_num])
        copy(img_path+dir+'/'+img_file, dst+dir+'/'+str(cat_num)+'_'+coco_classes_extracted[cat_num]+'/'+coco_classes_extracted[cat_num]+'_'+img_file)
        with open(dst+'data/'+dir[:-4]+'.txt','a') as f:
            f.write(dir[:-4]+'/'+str(cat_num)+'_'+coco_classes_extracted[cat_num]+'/'+coco_classes_extracted[cat_num]+'_'+img_file+'\n')



