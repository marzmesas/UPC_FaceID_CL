import os
from os.path import join
from torch.utils.data import Dataset
from random import sample
from torchvision import transforms
import cv2
import torch
import glob

#Clase de custom dataset para contrastive supervised
class CustomDataset_Supervised(Dataset):
  def __init__(self, path, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])):
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

#clase customdataset para contrastive unsupervised
class CustomDataset_Unsupervised(Dataset):
  def __init__(self, path, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])):
    self.path = path
    self.list_files = os.listdir(self.path)
    self.transform = transform
    self.pair_transformations = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ColorJitter(brightness=0.3, contrast = 0.3, saturation=0.3, hue=0.3),
      transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2)),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomCrop(size=(90,90)),
      transforms.ToTensor()])
      
    '''
      Otras transformaciones probadas
      transforms.RandomResizedCrop(size=(160, 160)),
      transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
      transforms.RandomEqualize(),
      transforms.Normalize((0.5188, 0.3797, 0.3145),(0.0712, 0.0653, 0.0722))
      
    '''

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

    '''
    # Código para debugar
    img1=np.ascontiguousarray((img_1.permute(1, 2, 0).numpy()*255), dtype=np.uint8)
    img2=np.ascontiguousarray((img_2.permute(1, 2, 0).numpy()*255), dtype=np.uint8)
    image_stack = np.hstack((img1, img2))
        
    plt.imshow(image_stack)
    plt.show()
    '''

    return {'image0':img_0, 'image1': img_1, 'image2': img_2, 'label': idx}

#clase customdataset para testing
class CustomDataset_Testing(Dataset):
  def __init__(self, path, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])):
    self.list_files = sorted(glob.glob(path+'/*/*.bmp',recursive=True))
    if self.list_files==[]:
      self.list_files = sorted(glob.glob(path+'/*.bmp',recursive=True))
    self.transform = transform
    self.labels = [(x.split('/')[-1])[0:4] for x in self.list_files]

  def __len__(self):
    return len(self.list_files)
  #Devolver 2 imágenes y los labels
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.list_files[idx]
    #image = Image.open(img_name).convert('RGB')
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = int(self.labels[idx][-2:])
    if self.transform:
      img_0 = self.transform(image)
    return {'image0': img_0,'label': label, 'path': img_name}

#clase customadatset para training y testing del supervised con la MLP
class CustomDataset_supervised_Testing(Dataset):
  def __init__(self, filenames,labels, transform=None):
    self.list_files = filenames
    self.transform = transform
    self.labels = labels

  def __len__(self):
    return len(self.list_files)
  #Devolver 2 imágenes y los labels
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.list_files[idx]
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = self.labels[idx]
    if self.transform:
      img_0 = self.transform(image)
    return {'image1': img_0,'label': label}