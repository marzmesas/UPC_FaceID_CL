import os
import torch

torch.manual_seed(1)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1)

import copy
from os.path import join

from model import base_model
from logging_moco import log_embeddings
from run_training import train_model
from evaluation import graph_embeddings, compute_embeddings, prediction

#from ray import tune
#from ray.tune.schedulers import ASHAScheduler
#from ray.tune import CLIReporter
import wandb
import cv2
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from torch.utils.tensorboard import SummaryWriter
import datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir)

#Declaramos device para GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

path_model='./saved_models'
# Variable para entrenar o para evaluar
training=True
# Variable para optimizar
optim=False
# Variable para testear predicciones
testing = False

# Conjunto de parámetros fijados
config_fixed={
          # He puesto el path en la configuración para pasar rápidamente del supervised al unsupervised
          "image_path":'./Cropped-IMGS',
          "tau": 0.05,
          #"batch_size": 16,
          #"lr": 0.001,
          "epochs":200,
          #"K": 2000,
          "momentum_optimizer": 0.9,
          "weight_decay": 1e-6
      }

# Definimos los modelos
modelq = base_model(pretrained=False).to(device)
modelk = copy.deepcopy(modelq)

# Entrenamos el modelo
# Si no optimizamos fijamos el conjunto de parámetros que utilizaremos más adelante para optimizar
if not optim:
  config={}
  config['lr']= 0.001
  config['batch_size']=16
  config['momentum']=0.999
  config['K']=2000
  
  if training:

    trained_modelq, trainet_modelk=train_model(config, config_fixed, modelq, modelk)

    # Guardamos modelos entrenados
    torch.save(modelq.state_dict(), os.path.join(path_model, 'modelq.pt'))
    torch.save(modelk.state_dict(), os.path.join(path_model, 'modelk.pt'))

  else:
    #load model state_dict
    trained_modelq = base_model(pretrained=False)
    trained_modelq.load_state_dict(torch.load(os.path.join(path_model, 'modelq.pt'),map_location=torch.device('cpu')))
  
  if testing:
    # Cogemos una imagen del dataset
    path_image_test = './Cropped-IMGS/ID39_001.bmp'
    # Cogemos una imagen que no está en el dataset
    #path_image_test = '/home/carles/faceid/carles_musoll.jpeg'

    # Recuperamos los embeddings y las imágenes para testear la predicción
    latents, labels, images = compute_embeddings(trained_modelq, config, config_fixed)
    #graph_embeddings(latents, labels)

    image_test = cv2.imread(path_image_test)
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    dist, idxs = prediction(image_test, trained_modelq, latents, num_neighboors=5)
    print(f'Distance from closest centroid: {dist}, Image from cluster {idxs}')
    
    # Graficamos las k imágenes del dataset que estan más próximas de la imagen de test
    image_stack = cv2.resize(image_test, (120,120))
    image_stack = cv2.rectangle(image_stack, (0,0), (120,120), (255,0,0), 10)
    image_stack = cv2.putText(image_stack, text='INPUT IMAGE', org=(10, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)
    
    for k, idx in enumerate(idxs):
      closer_image = images[idx]
      closer_image = closer_image.squeeze(0)
      closer_image = torch.permute(closer_image, (1,2,0))
      closer_image = closer_image.numpy()
      closer_image = np.ascontiguousarray((closer_image*255), dtype=np.uint8)
      closer_image = cv2.putText(closer_image, text=f'Dist {round(dist, 4)}', org=(10, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(0, 255, 0),thickness=1)
      image_stack = np.hstack((image_stack, closer_image))
      
    plt.imshow(image_stack)
    plt.show()

else:
  # Optimizacion ASHA de la librería tune.ray (la dejo por si la necesitamos más adelante)
  '''
  config={
          "lr": tune.uniform(1e-4, 1e-1),
          "batch_size": tune.choice([8,16,32, 64,128]),
          "tau": tune.uniform(0.1, 0.01),
          "momentum": tune.uniform(0.98, 0.999),
          "epochs": tune.choice([200,400,600, 800,1000])
      }

  scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=500,
        grace_period=1,
        reduction_factor=2)

  reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss"])

  gpus_per_trial = 1
  
  result = tune.run(
      partial(train_model, config_fixed=config_fixed, resnetq=modelq, resnetk=modelk, writer=None, optimization=optim),
      resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
      config=config,
      num_samples=30,
      scheduler=scheduler,
      progress_reporter=reporter,
      checkpoint_at_end=True)

  best_trial = result.get_best_trial("loss", "min", "last")
  print("Best trial config: {}".format(best_trial.config))
  
  config={}
  config['lr']=best_trial.config['lr']
  config['batch_size']=best_trial.config['batch_size']
  config['tau']=best_trial.config['tau']
  config['momentum']=best_trial.config['momentum']
  config['epochs']=best_trial.config['epochs']
  '''

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
      #'tau':  {
      #  'distribution': "uniform",
      #  'min':0.01,
      #  'max':0.1
      #  },
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

  # Una vez hecha la optimizacion, deberemos guardar los modelos optimizados 
  #torch.save(modelq.state_dict(), os.path.join(path_model, 'modelq_opt.pt'))
  #torch.save(modelk.state_dict(), os.path.join(path_model, 'modelk_opt.pt'))