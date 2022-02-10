import os
import numpy as np
from random import sample
import torch
import datetime
from skimage import io
torch.manual_seed(1)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1)
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as models
from collections import OrderedDict
import copy
import torch.optim as optim
import seaborn as sns
from os.path import join
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import pandas as pd
import seaborn as sn
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()

class CustomDataset(Dataset):
  def __init__(self, path, transform=None):
    self.list_files = [[join(path, x, y) for y in os.listdir(join(path, x))] for x in sorted(os.listdir(path))]
    self.transform = transform
  def __len__(self):
    return len(self.list_files)

  def __getitem__(self, idx):
    imgs_paths = sample(self.list_files[idx], 2)
    img_0 = io.imread(imgs_paths[0])
    img_1 = io.imread(imgs_paths[1])
    if self.transform:
      img_0 = self.transform(img_0)
      img_1 = self.transform(img_1)
    return {'image1': img_0, 'image2': img_1, 'label': idx}


def log_embeddings(model, data_loader, writer):
  # take a few batches from the training loader
  list_latent = []
  list_images = []
  list_labels =[]
  for i in range(20):
    for i,batch in enumerate(data_loader):
      imgs = batch['image1']
      labels = batch['label']
      # forward batch through the encoder
      list_latent.append(model(imgs))
      list_images.append(imgs)
      list_labels.append(labels)

  latent = torch.cat(list_latent)
  images = torch.cat(list_images)
  labels = torch.cat(list_labels)


  writer.add_embedding(latent,metadata=labels,label_img=images)
  return latent,labels,images

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir)
resnetq = models.resnet18(pretrained=False)

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(resnetq.fc.in_features, 100)),
    ('added_relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(100, 50)),
    ('added_relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(50, 25))
]))

directory = '/home/oriol/~project-faceid/GTV-Database-UPC'
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
resnetq.fc = classifier

if len(nn.Sequential(*list(resnetq.fc.children()))) == 5:
    resnetq.fc = nn.Sequential(*list(resnetq.fc.children())[:-3])
resnetq.eval()
PATH = "C:/users/hp/Escritorio/project/modelq.pt"
resnetq.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

dataset_embedding = CustomDataset(directory,transform)
data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=16,shuffle=True)
latents,labels,images=log_embeddings(resnetq, data_loader_embedding, writer)

#Cálculo de los clusters y representación gráfica con T-SNE
latents_cpu = latents.detach().cpu().numpy()
standardized_data = StandardScaler().fit_transform(latents_cpu)
print(standardized_data.shape)
sample_data = standardized_data
# matrix multiplication using numpy
covar_matrix = np.matmul(sample_data.T , sample_data)
values, vectors = eigh(covar_matrix, eigvals=(98,99))
vectors = vectors.T
new_coordinates = np.matmul(vectors, sample_data.T)
new_coordinates = np.vstack((new_coordinates, labels)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "labels"))
print(dataframe.head())
sn.FacetGrid(dataframe, hue="labels", height=10).map(plt.scatter,"1st_principal","2nd_principal" ).add_legend()
plt.savefig('T-SNE2.png')
