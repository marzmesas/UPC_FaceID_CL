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
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import pandas as pd
import seaborn as sn

#Declaramos device para GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()

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

#InfoNCE loss function
def loss_function(q, k, queue):

    # N es el batch size
    N = q.shape[0]
    
    # C es la dimensionalidad de las representaciones
    C = q.shape[1]

    #Batch matrix multiplication (las query con las keys por los batch)
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),τ))
    
    #Matrix multiplication (las query con la queue acumulada (memory bank))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),τ)), dim=1)
   
    #Calculamos el denominador de la función de coste
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))


def log_embeddings(model, data_loader, writer):
  #Cogemos 16 imágenes de cada unos de los directorios (personas distintas)
  list_latent = []
  list_images = []
  list_labels = []
  for i in range(16):
    for i,batch in enumerate(DataLoader):
      imgs = batch['image1']
      labels = batch['label']
      list_latent.append(model(imgs.to(device)))
      list_images.append(imgs)
      list_labels.append(labels)

  latent = torch.cat(list_latent)
  images = torch.cat(list_images)
  labels = torch.cat(list_labels)
  #Guardamos los embeddings para el tensorboard y su representación con PCA,T-SNE
  writer.add_embedding(latent,metadata=labels,label_img=images)
  return latent,labels,images

#Cálculo del coste medio
def get_mean_of_list(L):
    return sum(L) / len(L)


#TAMAÑO dim 0 K, Y TEMPERATURA   
τ = 0.05
K = 2000

#MODELO CON RESNET18
resnetq = models.resnet18(pretrained=False).to(device)
#Projection 
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(resnetq.fc.in_features, 100)),
    ('added_relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(100, 50)),
    ('added_relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(50, 25))
]))

resnetq.fc = classifier.to(device)
resnetk = copy.deepcopy(resnetq)



directory = r"C:/Users/hp/aidl-2022-winter-hands-on/Project/GTV-Database-UPC"
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
epoch_losses_train = []
num_epochs = 0
flag = 0
queue = None
optimizer = optim.SGD(resnetq.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)

dataset_queue = CustomDataset(directory,transform)
data_loader_queue = DataLoader(dataset=dataset_queue,batch_size=16,shuffle=True)

if queue is None:
    while True:
        with torch.no_grad():
            for (_, data) in enumerate(data_loader_queue):            
                
                xq, xk = data['image1'], data['image2']
                xq, xk = xq.to(device),xk.to(device)
                k = resnetk(xk)
                k = k.detach()
                k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                if queue is None:
                    queue = k
                else:
                    if queue.shape[0] < K:
                        queue = torch.cat((queue, k), 0)    
                    else:
                        flag = 1
                
                if flag == 1:
                    break

        if flag == 1:
            break

momentum = 0.999
num_epochs = 3000

#TRAINING
resnetq.train()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir)

dataset = CustomDataset(directory,transform)
data_loader = DataLoader(dataset=dataset,batch_size=16,shuffle=True)

for epoch in range(num_epochs):
  losses_train = []
  for i, data in enumerate(data_loader):
    optimizer.zero_grad()
    xq, xk = data['image1'], data['image2']
    xq, xk = xq.to(device), xk.to(device)
    #Obtención de los outputs
    q = resnetq(xq)
    k = resnetk(xk)
    k = k.detach()
    #Normalización de los vectores (para así calcular el coste, con multiplicación matricial)
    q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
    k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))
    #Obtención del coste
    loss = loss_function(q, k, queue)
    losses_train.append(loss.cpu().data.item())

    #Backpropagation
    loss.backward()

    #Actualización de los weights del resnet de las querys
    optimizer.step()
    queue = torch.cat((queue,k), 0) 
    #Si el tamaño de nuestro memory bank (queue) es mayor a 2000, eliminamos el último batch (batch size=16)
    if queue.shape[0] > K:
        queue = queue[16:,:]

    #Actualizamos el resnet de las keys
    for θ_k, θ_q in zip(resnetk.parameters(), resnetq.parameters()):
        θ_k.data.copy_(momentum*θ_k.data + θ_q.data*(1.0 - momentum))
    #Guardamos las losses y hacemos la media, así como la metemos en el writer para el tensorboard
  epoch_losses_train.append(get_mean_of_list(losses_train))
  print('Epoch'+str(epoch)+":"+" Loss: "+str(epoch_losses_train[epoch]))
  writer.add_scalar("train loss", epoch_losses_train[epoch], epoch)

#Para representar mejor los embeddings, quitamos la linear y nos quedamos con la convolucional
if len(nn.Sequential(*list(resnetq.fc.children()))) == 5:
    resnetq.fc = nn.Sequential(*list(resnetq.fc.children())[:-3])
#Cargamos a la función para obtener los embeddings el modelo sin la linear, y el dataset
latents,labels = log_embeddings(resnetq, data_loader, writer)
latents_cpu = latents.detach().cpu().numpy()
standardized_data = StandardScaler().fit_transform(latents_cpu)
print(standardized_data.shape)
sample_data = standardized_data
#Cálculo de los clusters con T-SNE
covar_matrix = np.matmul(sample_data.T , sample_data)
values, vectors = eigh(covar_matrix, eigvals=(98,99))
vectors = vectors.T
new_coordinates = np.matmul(vectors, sample_data.T)
new_coordinates = np.vstack((new_coordinates, labels)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "labels"))
print(dataframe.head())
sn.FacetGrid(dataframe, hue="labels", height=6).map(plt.scatter,"1st_principal","2nd_principal" ).add_legend()
plt.savefig('T-SNE.png')

fig = plt.figure(figsize=(10, 10))
sns.set_style('darkgrid')
plt.plot(epoch_losses_train)
plt.legend(['Training Losses'])
plt.savefig('losses.png')
plt.close()
torch.save(resnetq.state_dict(), 'C:/Users/hp/modelq.pt')
torch.save(resnetk.state_dict(), 'C:/Users/hp/modelk.pt')
torch.save(optimizer.state_dict(),'C:/Users/hp/optimizer.pt')
np.savez("C:/Users/hp/lossesfile", np.array(losses_train))
torch.save(queue, 'C:/Users/hp/queue.pt')
