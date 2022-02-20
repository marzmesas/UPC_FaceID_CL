import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image


transformations = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation=0.5, hue=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5)
])
    

frame = Image.open('./GTV-Database-UPC/ID01/ID01_001.bmp')
frame = transformations(frame)
frame.show()

