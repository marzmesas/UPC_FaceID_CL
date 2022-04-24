import torch
from torchvision import datasets, transforms as T
import os
import cv2
transform = T.ToTensor()
# Dataset = datasets.ImageNet(".", split="train", transform=transform)

path_dir = './Datasets/Cropped-IMGS-2'

means = []
stds = []
for file in os.listdir(path_dir):
    img = cv2.imread(os.path.join(path_dir, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    means.append(torch.mean(img, dim=(1,2)))
    stds.append(torch.std(img, dim=(1,2)))

stacked = torch.stack(means)

mean = stacked.mean(0)
std = stacked.std(0)

print(mean)
print(std)