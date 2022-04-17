import os
import torch
import torch.nn as nn

torch.manual_seed(1)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1)

import copy
from os.path import join

from model import base_model
from logging_moco import log_embeddings
from run_training import train_model
from evaluation import graph_embeddings, compute_embeddings, prediction, accuracy, silhouette

import wandb
import cv2
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from torch.utils.tensorboard import SummaryWriter
import datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir)

from dataset import CustomDataset_Supervised, CustomDataset_Unsupervised, CustomDataset_Testing
import glob
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#Declaramos device para GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


############# - Entorno a definir ####################

# Tipo entrenamiendo
# supervised --> True
# self-supervised --> False
supervised = False

path_img_training = './Cropped-IMGS-2'
path_img_testing = './Cropped-IMGS-2-supervised'

path_model='./models'
path_checkpoint='./checkpoints'
# Variable para entrenar o para evaluar
training = True
# Variable para optimizar
optim= False
# Variable para testear predicciones
testing = True

# fichero checkpoint para cargar en entrenamiento
pretrained_checkpoint_file =  'None'
# fichero para cargar pesos pre-entrenados
pretrained_model_file =  'None'
# fichero modelo de salida de entrenamiento
output_model_file = 'None'
# fichero modelo de test
test_model_file = output_model_file
# fichero checkpoint de test
test_checkpoint_file = 'None'

# available architectures
# - resnet18
# - resnet50
# - resnet101
# - inception resnet v1
# - vgg16
arch='resnet18'

# Tamaño de la queue
K=2000
batch_size=32
epochs=200
# Intervalo para guardar checkpoints
checkpoint_interval=50

######################################################

# Conjunto de parámetros fijados
config_fixed={
          # Path imágenes training
          "image_path": path_img_training,
          # Path imágenes testing
          "image_path_test": path_img_testing,
          "tau": 0.05,
          "epochs":epochs,
          "momentum_optimizer": 0.9,
          "weight_decay": 1e-6,
          "checkpoint_interval": checkpoint_interval,
          "supervised": supervised
          
      }

# Definimos los modelos
modelq = base_model(pretrained=False, arch=arch).to(device)
# Cargarmos checkpoint si existe
if os.path.exists(os.path.join(path_checkpoint, pretrained_checkpoint_file)):
  checkpoint = torch.load(os.path.join(path_checkpoint, pretrained_checkpoint_file), map_location=device)
  modelq.load_state_dict(checkpoint['model_state_dict'])
  optim_state=checkpoint['optimizer_state_dict']
else:
  optim_state=None
# Cargamos pesos pre-entrenados si existe
if os.path.exists(os.path.join(path_model, pretrained_model_file)):  
  modelq.load_state_dict(torch.load(os.path.join(path_model, pretrained_model_file),map_location=device))
# Copiamos el encoder-q a encoder-k
modelk = copy.deepcopy(modelq)

# Entrenamos el modelo
# Si no optimizamos fijamos el conjunto de parámetros que utilizaremos más adelante para optimizar
if not optim:
  config={}
  config['lr']= 0.001
  config['batch_size']=batch_size
  config['momentum']=0.999
  config['K']=K
  
  if training:

    # Entreno
    trained_modelq, trained_modelk=train_model(config, config_fixed, modelq, modelk, optim_state)

    # Guardamos modelos entrenados
    torch.save(trained_modelq.state_dict(), os.path.join(path_model, output_model_file))

    # Graficamos los embeddings en Tensorboard (opcional)
    latents, labels, images, path, trained_modelq = compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed, writer=writer, testing=False, image_test=None, supervised=supervised, inception=False)
    
  if testing:
    loaded=False
    trained_modelq = base_model(pretrained=False, arch=arch)
    # Cargamos
    if os.path.exists(os.path.join(path_model, test_model_file)):
      trained_modelq.load_state_dict(torch.load(os.path.join(path_model, test_model_file),map_location=device))
      loaded=True
    elif os.path.exists(os.path.join(path_checkpoint, test_checkpoint_file)):  
      checkpoint = torch.load(os.path.join(path_checkpoint, test_checkpoint_file), map_location=device)
      trained_modelq.load_state_dict(checkpoint['model_state_dict'])
      loaded=True
    else:
      print('Model or checkpoint for testing not found')
    
    if loaded:    

      latents, labels, images, path, trained_modelq = compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed, writer=writer, testing=testing, image_test=None, supervised=True, inception=False)
      
      test_names = sorted(glob.glob(config_fixed['image_path_test']+'/*/*.bmp',recursive=True))

      print('K-MEANS METHOD')
      print('TOPK1')
      print(accuracy(latents=latents, images=images, path=path, modelq=trained_modelq,list_files_test=test_names,topk=1,nombres=labels, method='kmeans'))
      print('TOPK3')
      print(accuracy(latents=latents, images = images, path=path, modelq=trained_modelq,list_files_test=test_names,topk=3,nombres=labels, method='kmeans'))
      print('TOPK5')
      print(accuracy(latents=latents, images = images, path=path, modelq=trained_modelq,list_files_test=test_names,topk=5,nombres=labels, method='kmeans'))
      print('K-NEIGHBOORS METHOD')
      print('TOPK1')
      print(accuracy(latents=latents,images = images, path=path, modelq=trained_modelq,list_files_test=test_names,topk=1,nombres=labels, method='kneighboors'))
      print('TOPK3')
      print(accuracy(latents=latents,images = images, path=path, modelq=trained_modelq,list_files_test=test_names,topk=3,nombres=labels, method='kneighboors'))
      print('TOPK5')
      print(accuracy(latents=latents, images = images, path=path, modelq=trained_modelq,list_files_test=test_names,topk=5,nombres=labels, method='kneighboors'))
      silhouette(X=latents)

else:

  # Optimizacion con la librería W&B (para más adelante)
  metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

  sweep_config = {
    'name' : 'moco hyperparam optimizer',
    'method' : 'bayes',
    'metric': metric,
    'parameters' : 
    {
      'K' : {'values' : [2000, 3000, 4000]},
      'lr' :  {
        'distribution': 'uniform',
        'min': 0.0001,
        'max': 0.1
        },
      'momentum':  {
        'distribution': "uniform",
        'min':0.95,
        'max':0.9999
        },
      'batch_size': {'values': [8,16,32,128]}
    }
  }

  sweep_id = wandb.sweep(sweep_config)

  count = 10 # number of runs to execute
  config=None
  wandb.agent(sweep_id, partial(train_model, config, config_fixed, modelq, modelk), count=count)

