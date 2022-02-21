
import os
from os.path import join
from torch.utils.data import Dataset
from random import sample
from skimage import io

#Clase de custom dataset

class CustomDataset(Dataset):
  def __init__(self, path, transform=None):
    self.list_files = [[join(path, x, y) for y in os.listdir(join(path, x))] for x in os.listdir(path)]
    self.transform = transform

  def __len__(self):
    return len(self.list_files)
  #Devolver 2 imágenes y los labels
  def __getitem__(self, idx):
    imgs_paths = sample(self.list_files[idx], 2)
    img_0 = io.imread(imgs_paths[0])
    img_1 = io.imread(imgs_paths[1])

    if self.transform:
      img_0 = self.transform(img_0)
      img_1 = self.transform(img_1)

    return {'image1': img_0, 'image2': img_1, 'label': idx}
