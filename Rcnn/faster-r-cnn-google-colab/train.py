from torchvision import transforms
import numpy as np

from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,train_batch_size
)
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
import torch
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
plt.style.use('ggplot')

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))


    # Create a normalization transform using torchvision
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a fixed size (example)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        #images = [data_transforms(image).to(DEVICE) for image in images]

        resized_images = []
        for image in images:
            # Resize each image (replace with your resizing method)
            resized_image = resize_function(image, target_size)
            resized_images.append(resized_image)

        # Stack images along batch dimension (assuming resized_images is a list of tensors)
        images = torch.stack(resized_images, dim=0)

        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

class YoloDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images", split))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels", split))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.split, self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.split, self.labels[idx])

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_center *= img.width
                y_center *= img.height
                width *= img.width
                height *= img.height

                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))
if __name__ == '__main__':
    # train_dataset = create_train_dataset()
    # valid_dataset = create_valid_dataset()
    # train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    # valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    # print(f"Number of training samples: {len(train_dataset)}")
    # print(f"Number of validation samples: {len(valid_dataset)}\n")
    # Create dataset and dataloader

    dataset_root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco_dataset/'

    train_dataset = YoloDataset(root=dataset_root, split='train', transforms=None)
    test_dataset = YoloDataset(root=dataset_root, split='test', transforms=None)
    val_dataset = YoloDataset(root=dataset_root, split='val', transforms=None)
    dataset = YoloDataset(dataset_root, transforms=None)#get_transform(train=True))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    # initialize the model and move to the computation device
    #model = create_model(num_classes=NUM_CLASSES)
    model = torch.load('fasterrcnn_model.pth')
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel()
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # save the best model till now if we have the least loss in the...
        # ... current epoch
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(epoch, model, optimizer)
        # save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss)
        
        # sleep for 5 seconds after each epoch
        time.sleep(5)

        