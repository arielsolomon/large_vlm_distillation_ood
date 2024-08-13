import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from PIL import Image
from models.experimental import attempt_load
from models.yolo import Model 


# Define your transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load data
src = '/data/Projects/fed_learn_fasterRcnn/large_vlm_distillation_ood/yolov5/'  # Adjust path if needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'



images = src+'coco128/images/train2017/'
labels = src+'coco128/labels/train2017/'

class yolo(Dataset):

    def __init__(self, images_path, labels_path, transform, size):
        self.img_path = images_path
        self.lbl_path = labels_path
        self.transforrm = transform
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
        return self.transforrm(img), lbl

    def _load_label(self,lbl_path):
        with open(lbl_path, 'r') as f:
            labels = f.read().strip().split('\n')
        return torch.tensor([list(map(float, line.split())) for line in labels], dtype=torch.float16)

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # Stack images into a single tensor
    return images, labels

size = (224,224)

trainset = yolo(images, labels, transform,size)
trainloader =  DataLoader(trainset, batch_size=4,
                        shuffle=True,
                        collate_fn=custom_collate_fn, num_workers=2)




# Load YOLOv5 models (modify names as needed)
student = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # Student model
model_path = '/data/Projects/fed_learn_fasterRcnn/large_vlm_distillation_ood/yolov5/runs/train/exp/yolov5x.pt'
# teacher model
teacher =  torch.load(model_path)['model'].float()


for param in student.parameters():
    param.requires_grad = True

for param in teacher.parameters():
    param.requires_grad = True

teacher.to(device)
teacher.eval()
# Set training parameters
epochs = 50
lr = 1e-4

def kl_divergence_loss(student_outputs, teacher_outputs, temperature=0.7):
    # Access student and teacher outputs based on YOLOv5 structure (modify if needed)
    student_probs = F.softmax(student_outputs / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return kl_loss

def trainer(student, teacher, trainloader, epochs, lr):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='Train'):
        for image, label in trainloader:
            # print(f"\nLabel format: {label}")
            # Move images and labels to the correct device
            image = image.to(device)
            label = label[0].to(device) 

            # Get student and teacher model outputs
            s_output = student(image)
            with torch.no_grad():
                t_output = teacher(image)

            # Access student model's pre-softmax confidence scores (modify based on YOLOv5 structure)
            student_conf = s_output[..., 4]  # Assuming confidence scores at index 4

            # Calculate detection loss from student model
            detection_loss = student(image, label)[0]  # Assuming loss is the first element

            # Calculate KL divergence loss
            kl_loss = kl_divergence_loss(student_conf, t_output[0][..., 4])

            # Combine losses (adjust weights if needed)
            total_loss = (kl_loss + detection_loss).sum()  # Ensure total_loss is a scalar
            print(f'\nRequire grad? {s_output.requires_grad}')  # Should be True
            print(f'\nRequire grad? {t_output[0].requires_grad}')  # Should be True
            print(f'loss {total_loss.item()}')
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return student

# Train the student model using knowledge distillation from the teacher model
trained_student_model = trainer(student, teacher, trainloader, epochs, lr)
