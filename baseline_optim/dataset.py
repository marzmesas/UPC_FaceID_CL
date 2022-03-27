import os
from os.path import join
from torch.utils.data import Dataset
from random import sample
from skimage import io
from torchvision import transforms
import cv2


#Clase de custom dataset
class CustomDataset_Supervised(Dataset):
  def __init__(self, path, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((240,240)), transforms.ToTensor()])):
    self.list_files = [[join(path, x, y) for y in os.listdir(join(path, x))] for x in os.listdir(path)]
    self.transform = transform

  def __len__(self):
    return len(self.list_files)
  
  #Devolver 2 imágenes y los labels
  def __getitem__(self, idx):
    imgs_paths = sample(self.list_files[idx], 2)
    img_0 = cv2.imread(imgs_paths[0])
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
    img_1 = cv2.imread(imgs_paths[1])
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

    if self.transform:
      img_0 = self.transform(img_0)
      img_1 = self.transform(img_1)

    return {'image0': img_0, 'image1': img_0, 'image2': img_1, 'label': idx}

class CustomDataset_Unsupervised(Dataset):
  def __init__(self, path, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])):
    self.path = path
    self.list_files = os.listdir(self.path)
    self.transform = transform
    #self.pair_transform = transforms.Compose([
    #  transforms.ToPILImage(),
    #  transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation=0.5, hue=0.5),
    #  transforms.RandomHorizontalFlip(p=0.5),
      #transforms.GaussianBlur(15*15),
    #  transforms.CenterCrop((120, 120)),
    #  transforms.ToTensor()
    #])
    self.pair_transformations = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation=0.5, hue=0.5),
      #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
      #transforms.RandomGrayscale(p=0.2),
      transforms.RandomHorizontalFlip(p=0.5),
      #transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
      transforms.RandomEqualize(),
      transforms.ToTensor(),
      #transforms.Normalize((0.5188, 0.3797, 0.3145),(0.0712, 0.0653, 0.0722))
      ])
    
  def __len__(self):
    return len(self.list_files)
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.path, self.list_files[idx])
    img_0 = cv2.imread(img_path)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
    if self.transform:
      img_0 = self.transform(img_0)
    img_1 = self.pair_transformations(img_0)
    img_2 = self.pair_transformations(img_0)

    return {'image0':img_0, 'image1': img_1, 'image2': img_2, 'label': idx}
