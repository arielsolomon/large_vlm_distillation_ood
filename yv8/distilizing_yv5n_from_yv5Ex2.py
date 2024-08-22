import sys
import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from PIL import Image
import wandb
from datetime import datetime

wandb.login(key='de945abee07d10bd254a97ed0c746a9f80a818e5')
current_date = datetime.now()
date = current_date.strftime("%d_%m_%Y")
wandb.init(project='nano_Ex_ob_distilation'+date)  # Replace with your project name
# Define your transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/coco/'  # Adjust path if needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'



images = src+'images/train/'
labels = src+'labels/train/'

class yolo(Dataset):

    def __init__(self, images_path, labels_path, transform, size):
        self.img_path = images_path
        self.lbl_path = labels_path
        self.transform = transform
        self.size = size

        self.image_list = sorted(os.listdir(self.img_path))
        self.labels_list = sorted(os.listdir(self.lbl_path))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        self.img_file, self.lbl_file = self.image_list[idx], self.labels_list[idx]
        img1 = Image.open(os.path.join(self.img_path, self.img_file)).convert("RGB")
        img = transforms.Resize(self.size)(img1)
        lbl = self._load_label(os.path.join(self.lbl_path, self.lbl_file))
        return self.transform(img), lbl

    def _load_label(self,lbl_path):
        with open(lbl_path, 'r') as f:
            labels = f.read().strip().split('\n')
        return torch.tensor([list(map(float, line.split())) for line in labels], dtype=torch.float16)

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # Stack images into a single tensor
    # print(f'batchsize {batch}')
    return images, labels

size = (224,224)

trainset = yolo(images, labels, transform,size)
trainloader =  DataLoader(trainset, batch_size=32,
                        shuffle=True,
                        collate_fn=custom_collate_fn, num_workers=4)



cwd = os.getcwd()
# Load YOLOv5 models (modify names as needed)
student = torch.hub.load('ultralytics/yolov5', 'yolov5n').to(device)  # Student model
#since I have cuda memory problem, I will downgrade to yolov5l model
model_path = cwd+'/yolov5x.pt'
# teacher model
teacher =  torch.load(model_path)['model'].float()


for param in student.parameters():
    param.requires_grad = True

for param in teacher.parameters():
    param.requires_grad = True

teacher.to(device)
teacher.eval()
# Set training parameters
epochs = 500
lr = 1e-3
temperature = 0.07

def kl_divergence_loss(student_outputs, teacher_outputs, temperature):
    # Access student and teacher outputs based on YOLOv5 structure (modify if needed)
    student_probs = F.softmax(student_outputs / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return kl_loss

def trainer(student, teacher, trainloader, epochs, lr,temperature):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    loss_list = []
    s_detection_loss = []
    for epoch in range(epochs):
        print(f'\nEpoch# {epoch+1}\n')
        sys.stdout.flush()
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = list(labels)
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            # Get student and teacher model outputs
            s_output = student(images)
            with torch.no_grad():
                t_output = teacher(images)

            # Access student model's pre-softmax confidence scores (modify based on YOLOv5 structure)
            student_conf = s_output[..., 4]  # Assuming confidence scores at index 4

            # Calculate detection loss from student model
            detection_loss = torch.tensor(student(images, labels)[0], requires_grad=True)

            #s_detection_loss.append(detection_loss)
            wandb.log({"student_detection_loss_diluted_data": detection_loss})

            # Calculate KL divergence loss WITH WEIGHTING
            loss_weight = 1e2
            kl_loss = kl_divergence_loss(student_conf, t_output[0][..., 4],temperature)
            wandb.log({"KL loss": kl_loss})
            total_loss = (kl_loss+ detection_loss/loss_weight).sum()/detection_loss.shape[0]
            wandb.log({"Total loss": total_loss})
            #print(f'\rTotal Loss: {total_loss.item():.4f} epoch {epoch}', end='')
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.cpu().detach().numpy())
            if epoch % 25 == 0:
                torch.save(student, f'distilled_nano_{epoch}_{date}.pt')
            torch.cuda.empty_cache()
    return student, loss_list,s_detection_loss

# Train the student model using knowledge distillation from the teacher model
trained_student_model, losses,s_detection_losses = trainer(student, teacher, trainloader, epochs, lr,temperature)
# torch.save(trained_student_model, cwd+'/distiled_student_diluted_data'+date+'_.pt')
# df = pd.DataFrame(losses)
# df_s_losses = pd.DataFrame(s_detection_losses)
# df_s_losses.to_csv(cwd+'/S_losses_diluted_data'+date+'.csv', header=False)
# df.to_csv(cwd+'/Total_losses_diluted_data'+date+'.csv', header=False)
