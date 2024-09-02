from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import wandb
from datetime import datetime
from utils.general import non_max_suppression as nms
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# Initialize wandb
wandb.login(key='de945abee07d10bd254a97ed0c746a9f80a818e5')
current_date = datetime.now()
date = current_date.strftime("%d_%m_%Y")
pr_name = '02_09_losses_modifications'
wandb.init(project=pr_name + date)  # Replace with your project name

# Define your transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
src = '/Data/federated_learning/large_vlm_distillation_ood/datasets/coco/10p_coco/'  # Adjust path if needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
images = src + 'images/train/'
labels = src + 'labels/train/'


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

    def _load_label(self, lbl_path):
        with open(lbl_path, 'r') as f:
            labels = f.read().strip().split('\n')
        return torch.tensor([list(map(float, line.split())) for line in labels], dtype=torch.float16)


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # Stack images into a single tensor
    return images, labels


# def kl_divergence_loss(student_outputs, teacher_outputs, temperature):
#     # Determine the minimum length to compare
#
#     min_len = min(student_outputs.size(0), teacher_outputs.size(0))
#
#     # Truncate both tensors to the same length
#     student_outputs = student_outputs[:min_len]
#     teacher_outputs = teacher_outputs[:min_len]
#
#     # Apply temperature scaling to soften the distributions
#     student_probs = F.softmax(student_outputs / temperature, dim=0)
#     teacher_probs = F.softmax(teacher_outputs / temperature, dim=0)
#
#
#     # Compute KL divergence loss
#     kl_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * (temperature ** 2)
#
#     return kl_loss
def kl_divergence_loss(teacher_output, student_output, temperature, device, epsilon=1e-7):
    """Calculates KL divergence between teacher and student outputs with temperature,
       ensuring alignment and handling different tensor lengths.

    Args:
        teacher_output: Tensor of teacher outputs (probabilities).
        student_output: Tensor of student outputs (probabilities).
        temperature: Temperature factor for softening the distribution.

    Returns:
        KL divergence loss.
    """

    # Ensure valid probability distributions
    teacher_output = torch.clamp(teacher_output, min=1e-7, max=1.0)
    student_output = torch.clamp(student_output, min=1e-7, max=1.0)

    # Apply temperature scaling
    teacher_output = teacher_output / temperature
    student_output = student_output / temperature

    # Find the minimum length
    min_length = min(teacher_output.shape[0], student_output.shape[0])

    # Pad shorter tensor with zeros to match the longer tensor
    teacher_output.to(device), student_output.to(device)
    if teacher_output.shape[0] < student_output.shape[0]:
        teacher_output = torch.cat([teacher_output, torch.zeros(student_output.shape[0] - teacher_output.shape[0]).to(device)])
    elif student_output.shape[0] < teacher_output.shape[0]:
        student_output = torch.cat([student_output, torch.zeros(teacher_output.shape[0] - student_output.shape[0]).to(device)])

    # Calculate KL divergence
    kl_loss = torch.sum(teacher_output * torch.log(teacher_output / student_output+epsilon))
    # Average the loss across the entire length
    kl_loss = kl_loss / teacher_output.shape[0]  # Or student_output.shape[0]
    if kl_loss == torch.inf:
        kl_loss = -0.02
    return kl_loss
def detection_loss(predictions, targets, weight_cls_loss, weight_bbox_loss):
    pred_boxes, pred_cls = [], []
    for pred in predictions:
        pred_boxes.append(pred[..., :4])
        pred_cls.append(pred[..., 4]) # for Cross entropy, one needs the confidence score
    pred_boxes = torch.cat(pred_boxes, dim=0)
    pred_cls = torch.cat(pred_cls, dim=0)
    pred_cls = torch.log(pred_cls)

    target_boxes = targets[..., 1:]
    target_cls = targets[..., 0]

    # Compute the cost matrix for box matching
    cost_matrix = torch.cdist(pred_boxes.float(), target_boxes.float(), p=1)  # L1 distance between boxes
    matched_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())


    # Convert the indices back to tensors
    pred_idx = torch.tensor(matched_indices[0], dtype=torch.long)
    target_idx = torch.tensor(matched_indices[1], dtype=torch.long)

    # Select the matched predictions and targets
    matched_pred_boxes = pred_boxes[pred_idx]
    matched_target_boxes = target_boxes[target_idx]
    matched_pred_cls = pred_cls[pred_idx].reshape(1,-1)
    matched_target_cls = target_cls[target_idx]

    # Calculate losses
    box_loss = F.mse_loss(matched_pred_boxes, matched_target_boxes)
    cls_loss = F.cross_entropy(matched_pred_cls, matched_target_cls.reshape(1,-1))

    total_loss = box_loss/weight_bbox_loss + cls_loss/weight_cls_loss
    return total_loss

def get_conf(predictions):
    confs = []
    # Extract confidence scores
    for ind, item in enumerate(predictions):
        if item.numel() < 1:
            pass
        elif item.shape[0] > 1:
            for ind2, i in enumerate(item):
                conf = item[ind2][4]
                confs.append(conf)
        else:
            conf = item[0][4]
            confs.append(conf)
    return confs

size = (640, 640)
trainset = yolo(images, labels, transform, size)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)

cwd = os.getcwd()

student = torch.hub.load('ultralytics/yolov5', 'yolov5n').to(device)  # Student model
model_path = cwd + '/yolov5x.pt'
teacher = torch.load(model_path)['model'].float()

for param in student.parameters():
    param.requires_grad = True

for param in teacher.parameters():
    param.requires_grad = True

teacher.to(device)
teacher.eval()

# Set training parameters
epochs = 1500
lr = 1e-3
temperature = 1.5
nms_threshold = 0.5  # Adjust as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainer(student, teacher, trainloader, epochs, lr, temperature, device):

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    loss_list = []
    weight_cls_loss, weight_bbox_loss = 1e05, 1e02

    for epoch in range(epochs):
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            labels = list(labels)
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            img_size = images.shape[-1]
            # Get student and teacher model outputs
            s_output = student(images)
            with torch.no_grad():
                t_output = teacher(images)

            # Extract raw outputs
            s_predictions = s_output  # Adjust based on actual output format
            t_predictions = t_output[0]  # Modify based on actual output format

            # Calculate detection loss before NMS
            # detection_loss_val = detection_loss(s_predictions, labels[0], device)


            """
            I also need to get predictions/confidence s cores from the student in eval mode so I can apply it to 
            KL loss function"""
            # Apply NMS

            s_predictions = nms(s_predictions)

            t_predictions = nms(t_predictions)

            # not elegant but this is so far....
            for ten in s_predictions:
                if ten.numel()>0:
                    if ten.shape[0]>1:
                        for bo in ten:
                            box = bo[:4]/img_size
                            bo[:4] = box
                    else:
                        box1 = ten[0][:4]/img_size
                        ten[:,:4]=box1

            for ten in t_predictions:
                if ten.numel()>0:
                    if ten.shape[0]>1:
                        for bo in ten:
                            box = bo[:4]/img_size
                            bo[:4] = box
                    else:
                        box1 = ten[0][:4]/img_size
                        ten[:,:4]=box1

            s_predictions = [torch.tensor(pred.to(torch.float16), dtype=torch.float16, requires_grad=True) for pred in s_predictions]
            t_predictions = [torch.tensor(pred.to(torch.float16), dtype=torch.float16, requires_grad=True) for pred in t_predictions]
            # Calculate detection loss after NMS
            d_loss = detection_loss(s_predictions, labels[0],weight_cls_loss, weight_bbox_loss)

            # Log both losses

            wandb.log({"student_detection_loss_diluted_data_after_nms": d_loss})
            # student_conf = torch.tensor(get_conf(s_predictions), requires_grad=True)
            # teacher_conf = torch.tensor(get_conf(t_predictions), requires_grad=True)
            student_conf = get_conf(s_predictions)
            teacher_conf = get_conf(t_predictions)

            # Ensure the confidences are tensors of the right dtype and device
            student_conf = torch.tensor(student_conf, dtype=torch.float16, device=device, requires_grad=True)
            teacher_conf = torch.tensor(teacher_conf, dtype=torch.float16, device=device, requires_grad=True)

            # Calculate KL divergence loss
            kl_loss = kl_divergence_loss(student_conf, teacher_conf, temperature,device, epsilon=1e-7)
            wandb.log({"KL loss": kl_loss})
            # Total loss
            total_loss = kl_loss + d_loss
            wandb.log({"Total loss": total_loss})
            #print(f'\rTotal Loss: {total_loss.item():.4f} epoch {epoch}', end='')
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.cpu().detach().numpy())
            if epoch % 25 == 0:
                torch.save(student, f'{pr_name}_{epoch}_{date}.pt')
                torch.save(student.state_dict(), f'state_dict_{pr_name}_{epoch}_{date}.pt')
            torch.cuda.empty_cache()
    return student, loss_list

trained_student_model, losses = trainer(student, teacher, trainloader, epochs, lr, temperature, device)
