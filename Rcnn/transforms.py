import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F  # Importing the functional module

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
