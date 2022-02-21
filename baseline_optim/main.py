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
from evaluation import graph_embeddings

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

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
training=False
# Variable para optimizar
optim=True

config_fixed={
          "image_path":'/home/carles/faceid/UPC_FaceID_CL/GTV-Database-UPC',
          #"tau": 0.05,
          #"batch_size": 16,
          #"lr": 0.001,
          #"epochs":200,
          "K": 2000,
          "momentum_optimizer": 0.9,
          "weight_decay": 1e-6
      }

# Definimos los modelos
modelq = base_model(pretrained=True).to(device)
modelk = copy.deepcopy(modelq)

# Entrenamos el modelo
if not optim:
  config={}
  config['lr']= 0.001
  config['batch_size']=16
  config['tau']=0.05
  config['momentum']=0.99
  config['epochs']=200
  
  if training:

    trained_modelq, trainet_modelk=train_model(config, config_fixed, modelq, modelk, writer=writer, optimization=optim)

    # Guardamos modelos entrenados
    torch.save(modelq.state_dict(), os.path.join(path_model, 'modelq.pt'))
    torch.save(modelk.state_dict(), os.path.join(path_model, 'modelk.pt'))

  else:
    #load model state_dict
    trained_modelq = base_model(pretrained=False).to(device)
    trained_modelq.load_state_dict(torch.load(os.path.join(path_model, 'modelq.pt'),map_location=torch.device('cpu')))

  graph_embeddings(trained_modelq, config, config_fixed, writer)  

else:

  # Parametros a optimizar
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
  
  trained_modelq, trained_modelk=train_model(config, config_fixed, modelq, modelk, writer, False)

  # Guardamos modelos optimizados
  torch.save(modelq.state_dict(), os.path.join(path_model, 'modelq_opt.pt'))
  torch.save(modelk.state_dict(), os.path.join(path_model, 'modelk_opt.pt'))

  graph_embeddings(trained_modelq, config, config_fixed, writer) 