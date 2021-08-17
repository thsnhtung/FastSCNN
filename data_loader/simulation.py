"""Cityscapes Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2


from PIL import Image, ImageOps, ImageFilter

__all__ = ['CitySegmentation']

COLORMAP = [[0, 0, 0], [255, 255, 255]]
CLASSES = ['background', 'road']


def colormap2label():
        colormap2label = np.zeros(256**3)
        for i, colormap in enumerate(COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                        colormap[2]] = i
        return colormap2label

class Simulation(data.Dataset):
    NUM_CLASS = len(CLASSES)


    def __init__(self, root='', img_transforms=None, size = (160, 320), crop_top = False, **kwargs):
        super(Simulation, self).__init__()
        self.root = root

        img_transforms = transforms.Compose([
        #transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.size = size
        self.img_transform = img_transforms

        self.COLORMAP2LABEL = colormap2label()                      #create label map

        self.img_dir = str(os.path.join(self.root, 'image'))
        self.label_dir = str(os.path.join(self.root, 'label'))
        self.Check()               # check the integrity of dataset

        self.crop_top = crop_top

        self.image_files = os.listdir(self.img_dir)
        self.len = len(self.image_files)


    def Check(self):
        for i in os.listdir(self.img_dir):
            label_file = os.path.join(self.label_dir, i) 
            assert os.path.isfile(label_file) , "This image has no label file " + label_file

        for i in os.listdir(self.label_dir):
            img_file = os.path.join(self.img_dir, i) 
            assert os.path.isfile(img_file) , "This label has no image " + img_file

    def __getitem__(self, index):
        img_file = os.path.join(self.img_dir, self.image_files[index])  
        img = Image.open(img_file)
        img = img.crop((0,84,320, 180))
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        label_file = os.path.join(self.label_dir, self.image_files[index])
        label = np.asarray(Image.open(label_file).crop((0,84,320, 180))) 
        #label = cv2.resize(label, (self.size[1], self.size[0]) , cv2.INTER_NEAREST)
        label = self.label_indices(label, self.COLORMAP2LABEL)
        label = torch.tensor(label)


        if self.crop_top:
            print('Chua lam')

        return img, label

    def __len__(self):
        return self.len
    
    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    def label_indices(self, colormap, colormap2label):
        colormap = np.array(colormap)
        colormap = colormap.astype(np.int32)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
            colormap[:, :, 2])
        img = colormap2label[idx]
        return  img



if __name__ == '__main__':
    dir = r'C:\Users\Asus\Downloads\TestFastSCNN'
    dataset = Simulation(root = dir)
    img, label = dataset[0]
    print(img.size())
    print(label.size())