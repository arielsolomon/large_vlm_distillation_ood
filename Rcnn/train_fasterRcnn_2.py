import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import random
import time
import datetime
from collections import defaultdict, deque
from torchvision.transforms import functional as F


# Custom dataset for loading YOLOv5 format data
import os

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



# Define the transformations
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = image.width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


# Load custom ResNet-18 model and use it as the backbone for Faster R-CNN
def get_model(num_classes, resnet50_weights_path):
    backbone_state_dict = torch.load(resnet50_weights_path)

    # Remove the last fully connected layers 'fc.weight' and 'fc.bias'
    # del backbone_state_dict['state_dict']['fc.weight']
    # del backbone_state_dict['state_dict']['fc.bias']

    # Load the ResNet-50 backbone model
    backbone = torchvision.models.resnet50()

    # Load the modified state_dict to the backbone model
    backbone.load_state_dict(backbone_state_dict.state_dict())

    # Set the out_channels attribute
    backbone.out_channels = 2048
    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # ROI Pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


# Function to collate the batch
def collate_fn(batch):
    return tuple(zip(*batch))


# Define metric logging class
class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
        ])
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# Training and evaluation functions
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    to_tensor = torchvision.transforms.ToTensor()
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(to_tensor(image).to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()

if __name__ == "__main__":
    # Set the root directory of your dataset
    dataset_root = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco_dataset/"

    # Set the path to the pre-trained ResNet-18 weights
    # resnet50_weights_path = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/Rcnn/faster_rcnn_chat/coco_8cls_resnet_distilled.pth"
    resnet50_weights_path = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/Rcnn/test_distilled_model_resnet50.pth"

    # Define other necessary parameters
    num_classes = 8  # Update this with the number of classes in your dataset
    train_batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    print_freq = 50

    # Create dataset and dataloader
    train_dataset = YoloDataset(root=dataset_root, split='train', transforms=None)
    test_dataset = YoloDataset(root=dataset_root, split='test', transforms=None)
    val_dataset = YoloDataset(root=dataset_root, split='val', transforms=None)
    dataset = YoloDataset(dataset_root, transforms=None)#get_transform(train=True))
    data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)

    # Get the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = get_model(8, resnet50_weights_path)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        lr_scheduler.step()
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
