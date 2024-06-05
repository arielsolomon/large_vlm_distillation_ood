import torch
BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 640 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco_dataset/images/train'
# validation images and XML files directory
VALID_DIR = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco_dataset/images/valid'
# classes: 0 index is reserved for background
CLASSES = [
    'person','bicycle','car','airplane','train','boat','traffic light','dog'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/Rcnn/faster-r-cnn-google-colab/outputs/'
train_batch_size = 8