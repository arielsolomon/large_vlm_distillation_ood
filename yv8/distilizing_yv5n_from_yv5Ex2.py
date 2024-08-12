import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define your transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load data
src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/ultralytics/ultralytics/'  # Adjust path if needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    images, targets = zip(*batch)
    images = [transforms.ToTensor()(img.resize((224, 224))) for img in images]
    return torch.stack(images, 0), targets

dataset = CocoDetection(
    root=src+'coco128/images/train2017',
    annFile=src+'coco128/annotations/instances_train2017.json',
    transform=None
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn  # Use the custom collate function
)

# Load YOLOv5 models (modify names as needed)
student = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # Student model
teacher = torch.hub.load('ultralytics/yolov5', 'yolov5x').to(device).eval()  # Teacher model

# Set training parameters
epochs = 50
lr = 1e-4

def kl_divergence_loss(student_outputs, teacher_outputs, temperature=0.7):
    # Access student and teacher outputs based on YOLOv5 structure (modify if needed)
    student_probs = F.softmax(student_outputs / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return kl_loss

def trainer(student, teacher, dataloader, epochs, lr):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='Train'):
        for image, label in dataloader:
            print(f"\nLabel format: {label}")
            break
            # Move images and labels to the correct device
            image = image.to(device)
            label = [{k: v.to(device) for k, v in t.items()} for t in label]

            # Get student and teacher model outputs
            s_output = student(image)
            with torch.no_grad():
                t_output = teacher(image)

            # Access student model's pre-softmax confidence scores (modify based on YOLOv5 structure)
            student_conf = s_output[..., 4]  # Assuming confidence scores at index 4

            # Calculate detection loss from student model
            detection_loss = student(image, label)[0]  # Assuming loss is the first element

            # Calculate KL divergence loss
            kl_loss = kl_divergence_loss(student_conf, t_output[..., 4])

            # Combine losses (adjust weights if needed)
            total_loss = (kl_loss + detection_loss).sum()  # Ensure total_loss is a scalar

            print(f'loss {total_loss.item()}')
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return student

# Train the student model using knowledge distillation from the teacher model
trained_student_model = trainer(student, teacher, dataloader, epochs, lr)
