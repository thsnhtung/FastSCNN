import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms as transforms


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            print("flip")
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomRotate(object):
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob
    def __call__(self, image, target):
        if random.random() > self.rotate_prob:
            print("RandomRotate")
            angle = random.randint(-10, 10)
            print("angle: ", angle)
            image = F.rotate(image, angle)
            target = F.rotate(target, angle)
        return image, target




class CropTop(object):
    def __call__(self, image, target):
        image = image.crop((0,84,320, 180))
        target = target.crop((0,84,320, 180))
        return image, target


class MyNormalize(object):
    def __init__(self):
        self.img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, image, target):
        image = self.img_transforms(image)
        target = torch.tensor(np.array(target))
        return image, target


class Label_Indice(object):
    def __init__(self, labelMap):
        self.labelMap = labelMap
    
    def __call__(self, image, target):
        target = self.label_indices(target, self.labelMap)
        return image, target

    def label_indices(self, target, colormap2label):
        target = np.array(target)
        target = target.astype(np.int32)
        idx = ((target[:, :, 0] * 256 + target[:, :, 1]) * 256 +
            target[:, :, 2])
        img = colormap2label[idx]
        return  img

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class MyTransform(object):
    def __init__(self, labelMap):
        self.labelMap = labelMap
        self.transforms = [CropTop(), RandomRotate(0.5), RandomHorizontalFlip(0.5), Label_Indice(self.labelMap), MyNormalize()]

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

