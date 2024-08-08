# what should distilation script include:
import torch
from ultralytics import YOLO
import os
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Define your transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),])

# loading data

transform = transforms.Compose([
    transforms.Resize((224, 224))])#,  # Resize images if needed
    #transforms.ToTensor(),  # Convert PIL Image or numpy array to tensor
#])
src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/ultralytics/ultralytics/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    images, targets = zip(*batch)
    images = [transforms.ToTensor()(image) for image in images]
    return torch.stack(images, 0), targets

# Load data
dataset = CocoDetection(
    root=src+'coco128/images/train2017',
    annFile=src+'coco128/annotations/instances_train2017.json',
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn  # Use the custom collate function
)


student = YOLO('yolov8n.pt')
teacher = YOLO('yolov8x.pt')
epochs = 50
lr = 1e-4

# defining loss function (culber divergens loss)

def kl_divergence_loss_orig(student_outputs, teacher_outputs, temperature=0.7):
    """
    Calculate KL divergence loss between teacher and student outputs.

    Args:
        student_outputs (Tensor): The output logits from the student model.
        teacher_outputs (Tensor): The output logits from the teacher model.
        temperature (float): Temperature parameter for scaling logits.

    Returns:
        Tensor: The KL divergence loss.
    """
    # Apply temperature scaling
    student_probs = F.softmax(student_outputs / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=-1)

    # Compute KL divergence loss
    kl_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=-1),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return kl_loss

def kl_divergence_loss(student_outputs, teacher_outputs, temperature=0.7):
    student_probs = F.softmax(student_outputs / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=-1)
    kl_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=-1),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    return kl_loss

def logits(p):
    return torch.log(p/(1-p))

def trainer(student, teacher, dataloader, epochs,lr):
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    s_model = student.to(device)#.train()
    t_model = teacher.to(device)#.eval()
    
    for epoch in tqdm(range(epochs), desc='Train'):

        for image, label in dataloader:

            s_output, t_output = s_model(image), t_model(image)
            detection_loss = s_model(image, label)
            kl_loss = kl_divergence_loss(logits(s_output[0].boxes.conf), logits(t_output[0].boxes.conf), temperature=0.7)
        
        # Combine losses
            total_loss = detection_loss + kl_loss
            print(f'loss {total_loss}')
        # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return s_model

trainer(student, teacher, dataloader, epochs,lr)