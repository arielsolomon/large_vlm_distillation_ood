import os
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import numpy as np
import torch.utils.data
import torchvision.transforms as T
from engine import train_one_epoch, evaluate

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load labels (assuming YOLO format)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        boxes = []
        labels = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = (x_center - width / 2) * img.width
            y_min = (y_center - height / 2) * img.height
            x_max = (x_center + width / 2) * img.width
            y_max = (y_center + height / 2) * img.height
            boxes.append([x_min, y_min, x_max, y_max])
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
        return len(self.images)

def save_images_with_boxes(image, boxes, labels, save_path):
    import cv2
    import numpy as np

    # Convert image to numpy array
    image_np = np.array(image)

    # Draw bounding boxes and labels on image
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save image with bounding boxes
    cv2.imwrite(save_path, image_np)

# Custom function to calculate AP for each class
def calculate_ap(coco_eval, iou_type='bbox'):
    precision = coco_eval.coco_eval[iou_type].eval['precision']
    recalls = coco_eval.coco_eval[iou_type].eval['recall']
    aps = []
    for precision_per_class, recall_per_class in zip(precision, recalls):
        if len(precision_per_class) > 0 and len(recall_per_class) > 0:
            ap = np.mean(precision_per_class)
            aps.append(ap)
    return aps

# Load ResNet model
resnet_model = torchvision.models.resnet18(pretrained=False)  # Set pretrained=False to load your own weights
weights_path = "model_best.pt"
state_dict = torch.load(weights_path)
resnet_model.load_state_dict(state_dict)
backbone = resnet_fpn_backbone('resnet18', pretrained=False)

# Modify Mask R-CNN backbone
maskrcnn_model = MaskRCNN(backbone=backbone, num_classes=91)  # 91 for COCO dataset classes

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset
dataset = CustomDataset(image_dir='path/to/images', label_dir='path/to/labels', transforms=None)  # Add transforms if needed

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialize optimizer
params = [p for p in maskrcnn_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop with validation
num_epochs = 5
for epoch in range(num_epochs):
    # Training
    train_one_epoch(maskrcnn_model, optimizer, train_dataloader, device, epoch, print_freq=10)

    # Validation
    coco_evaluator = evaluate(maskrcnn_model, val_dataloader, device=device)
    aps = calculate_ap(coco_evaluator)
    mean_ap = np.mean(aps)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation mAP: {mean_ap}")

# Save images with bounding boxes
save_images_with_boxes(maskrcnn_model, train_dataloader, save_dir='output_images', device=device)
