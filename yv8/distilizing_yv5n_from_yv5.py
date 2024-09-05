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


# Initialize #wandb
wandb.login(key='de945abee07d10bd254a97ed0c746a9f80a818e5')
current_date = datetime.now()
date = current_date.strftime("%d_%m_%Y")
pr_name = '05_09_kl_loss_equal_dist_size_temp07_s_model_from_Lmodel'
wandb.init(project=pr_name + date)  # Replace with your project name

# Define your transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
src = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/coco_10p/'  # Adjust path if needed
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


def add_samples(student_out, teacher_out):
    if teacher_out.size(0) == 0 or student_out.size(0) == 0:
        print("Skipping KL loss calculation due to zero-length dimension.")
        return torch.tensor(0.0).to(device)
    # if teacher size bigger than student size it gets 1, else it gets 0
    min_index = 0 if teacher_out.shape[0] < student_out.shape[0] else 1
    t_out, s_out = teacher_out.clone(), student_out.clone()
    if student_out.size(0)==0 or teacher_out.size(0)==0:
        return student_out, teacher_out 
    elif t_out.size(0)==s_out.size(0):
        return student_out, teacher_out
    else:

        if min_index==0:
            mean = t_out.mean()
            std = t_out.std()
            additional_samples = torch.normal(mean=float(mean), std=float(std), size=(1,(s_out.size(0) - t_out.size(0)))).to(device)
            if additional_samples.dim()>1 and additional_samples.shape[1]>1:
                additional_samples = additional_samples.squeeze()
            elif additional_samples.shape ==(1,1):
                additional_samples = additional_samples[:,0]
            if teacher_out.dim()>1:
                teacher_out = teacher_out.squeeze()
            teacher_out = torch.cat((teacher_out, additional_samples))
            # try:
            #     teacher_out = torch.cat((teacher_out, additional_samples))
            # except:
            #     print('debug')
                

        else:
            mean = s_out.mean()
            std = s_out.std()
            additional_samples = torch.normal(mean=float(mean), std=float(std), size=(1,(t_out.size(0) - s_out.size(0)))).to(device)
            if additional_samples.dim()>1 and additional_samples.shape[1]>1:
                additional_samples = additional_samples.squeeze()
            elif additional_samples.shape ==(1,1):
                additional_samples = additional_samples[:,0]
            if student_out.dim()>1:
                student_out = student_out.squees()
            student_out = torch.cat((student_out, additional_samples))
            # try:
            #     student_out = torch.cat((student_out, additional_samples))
            # except:
            #     print('break student')
        
        return student_out, teacher_out


def kl_divergence_loss(teacher_output, student_output, temperature, device, epsilon=1e-7):

    if teacher_output.size(0) == 0 or student_output.size(0) == 0:
        print("Skipping KL loss calculation due to zero-length dimension.")
        return torch.tensor(0.0).to(device) 
    # Ensure valid probability distributions
    teacher_output = torch.clamp(teacher_output, min=1e-7, max=1.0)
    student_output = torch.clamp(student_output, min=1e-7, max=1.0)

    # Apply temperature scaling
    teacher_output = teacher_output / temperature
    student_output = student_output / temperature
    
    # Find the minimum length
    student_output, teacher_output =  add_samples(student_output, teacher_output)

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
inx_tmp = 0 
size = (640, 640)
trainset = yolo(images, labels, transform, size)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)

cwd = os.getcwd()

student = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # Student model
model_path = cwd + '/yolov5l.pt'
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
temperature = 0.07
nms_threshold = 0.5  # Adjust as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# start training
def trainer(student, teacher, trainloader, epochs, lr, temperature, device, inx_tmp):
    inx_tmp = 0
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    weight_cls_loss, weight_bbox_loss = 1e03, 1e0

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
            #print('\n', inx_tmp, '\n')
            inx_tmp += 1
            # extract/focus predictions after NMS

            s_predictions = nms(s_predictions)

            t_predictions = nms(t_predictions)

            # not elegant but this is so far....
            for ten in s_predictions:
                # if tensor is not empty
                if ten.numel()>0:
                    # if single bounding box
                    if ten.shape[0]>1:
                        for bo in ten:
                            # normalize according to imgage dimension for values between 0 to 1
                            box = bo[:4]/img_size
                            bo[:4] = box
                    # for multiple boxes
                    else:
                        box1 = ten[0][:4]/img_size
                        ten[:,:4]=box1

            for ten in t_predictions:
                #same here
                if ten.numel()>0:
                    if ten.shape[0]>1:
                        for bo in ten:
                            box = bo[:4]/img_size
                            bo[:4] = box
                    else:
                        box1 = ten[0][:4]/img_size
                        ten[:,:4]=box1
            # Moving them to tensor type hald
            s_predictions = [torch.tensor(pred.to(torch.float16), dtype=torch.float16, requires_grad=True) for pred in s_predictions]
            t_predictions = [torch.tensor(pred.to(torch.float16), dtype=torch.float16, requires_grad=True) for pred in t_predictions]
            
            # Calculate detection loss after NMS to be weighted regarding KL_divergence loss
            
            d_loss = detection_loss(s_predictions, labels[0],weight_cls_loss, weight_bbox_loss)

            # Log both losses

            #wandb.log({"student_detection_loss_diluted_data_after_nms": d_loss})

            student_conf = get_conf(s_predictions)
            teacher_conf = get_conf(t_predictions)

            # Ensure the confidences are tensors of the right dtype and device
            student_conf = torch.tensor(student_conf, dtype=torch.float16, device=device, requires_grad=True)
            teacher_conf = torch.tensor(teacher_conf, dtype=torch.float16, device=device, requires_grad=True)

            # Calculate KL divergence loss
            kl_loss = kl_divergence_loss(student_conf, teacher_conf, temperature,device, epsilon=1e-7)
            #wandb.log({"KL loss": kl_loss})
            # Total loss
            total_loss = kl_loss + d_loss
            #wandb.log({"Total loss": total_loss})
            #print(f'\rTotal Loss: {total_loss.item():.4f} epoch {epoch}', end='')
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                torch.save(student, f'{pr_name}_{epoch}_{date}.pt')
                torch.save(student.state_dict(), f'state_dict_{pr_name}_{epoch}_{date}.pt')
            torch.cuda.empty_cache()
    return student, loss_list

trained_student_model, losses = trainer(student, teacher, trainloader, epochs, lr, temperature, device,inx_tmp)
