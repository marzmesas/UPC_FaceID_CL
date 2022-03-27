
import os
from os.path import join
from torch.utils.data import Dataset
from random import sample
from skimage import io
from torchvision import transforms
import cv2


#Clase de custom dataset
class CustomDataset_Supervised(Dataset):
  def __init__(self, path, transform=None):
    self.list_files = [[join(path, x, y) for y in os.listdir(join(path, x))] for x in os.listdir(path)]
    self.transform = transform

  def __len__(self):
    return len(self.list_files)
  #Devolver 2 imágenes y los labels
  def __getitem__(self, idx):
    imgs_paths = sample(self.list_files[idx], 2)
    img_0 = cv2.imread(imgs_paths[0], cv2.COLOR_BGR2RGB)
    img_1 = cv2.imread(imgs_paths[1], cv2.COLOR_BGR2RGB)

    if self.transform:
      img_0 = self.transform(img_0)
      img_1 = self.transform(img_1)

    return {'image1': img_0, 'image2': img_1, 'label': idx}

class CustomDataset_Unsupervised(Dataset):
  def __init__(self, path, transform=None):
    self.path = path
    self.list_files = os.listdir(self.path)
    self.tranform = transform
    self.pair_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation=0.5, hue=0.5),
      transforms.RandomHorizontalFlip(p=0.5),
      #transforms.GaussianBlur(15*15),
      transforms.CenterCrop((120, 120)),
      transforms.ToTensor()
    ])

  def __len__(self):
    return len(self.list_files)
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.path, self.list_files[idx])
    img_0 = cv2.imread(img_path)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
    img_0 = self.tranform(img_0)
    img_1 = self.pair_transform(img_0)

    return {'image1': img_0, 'image2': img_1, 'label': idx}
