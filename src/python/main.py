import os
import torch
torch.manual_seed(1)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1)
import copy

from model import base_model
from model import supervised_model
from run_training import train_model
from run_training import train_supervised_model
from evaluation import compute_embeddings, accuracy, test_supervised_model,silhouette

import wandb
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import glob
import matplotlib.pyplot as plt

#Declaramos device para GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


############# - Entorno a definir ####################

# Tipo entrenamiendo
# supervised --> True
# self-supervised --> False
# supervised_MLP (con un clasificador --> True, sin clasificador, KMEANS, KNN --> False)
supervised = True
supervised_MLP = False
plot_MLP = True  #Variable booleana para plotear la loss en el supervised contrastive MLP
plot_contrastive=True #Variable booleana para plotear la contrastive loss
#Variable booleana para comprobar la accuracy del dataset de test mientras entrenas (SUPERVISED WITH MLP)
testing_training = True
path_img_training="./Datasets/Cropped-IMGS-3-supervised"
path_img_testing ="./Datasets/Cropped-IMGS-3-supervised"
path_model="./src/python/saved_models"
path_checkpoint="./src/resources/checkpoints"
# Variable para entrenar o para evaluar
training = False
# Variable para optimizar
optim= False
# Variable para testear predicciones
testing = True

# fichero checkpoint para cargar en entrenamiento
pretrained_checkpoint_file =  'None'
# fichero para cargar pesos pre-entrenados
pretrained_model_file =  'None'

# fichero modelo de salida de entrenamiento contrastive 
output_model_file = 'model_Contrastive.pt'
# fichero modelo de test contrastive
test_model_file = output_model_file
# fichero modelo de salida de entrenamiento MLP (para el supervised)
output_model_file_MLP = 'model_MLP.pt'
# fichero modelo de test MLP (para el supervised)
test_model_file_MLP = output_model_file_MLP

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
K=350
batch_size=32
epochs_contrastive=2000
epochs_supervisedMLP=15
# Intervalo para guardar checkpoints
checkpoint_interval=1000

######################################################

# Conjunto de parámetros fijados
config_fixed={
          # Path imágenes training
          "image_path": path_img_training,
          # Path imágenes testing
          "image_path_test": path_img_testing,
          "tau": 0.05,
          "epochs":epochs_contrastive,
          "epochs_supervised":epochs_supervisedMLP,
          "momentum_optimizer": 0.9,
          "weight_decay": 1e-6,
          "checkpoint_interval": checkpoint_interval,
          "supervised": supervised,
          "supervised_MLP":supervised_MLP
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
    if supervised:
      if supervised_MLP:
        print("Entrenamiento basado en supervised contrastive loss con un clasificador MLP:")
      else:
        print("Entrenamiento basado en supervised contrastive loss sin clasificador:")
    else:
      print("Entrenamiento basado en self-supervised contrastive loss:")
    # Entreno
    trained_modelq, trained_modelk,epoch_losses=train_model(config, config_fixed, modelq, modelk, optim_state)

    # Guardamos modelos entrenados
    torch.save(trained_modelq.state_dict(), os.path.join(path_model, output_model_file))

    # Graficamos los embeddings en Tensorboard del dataset de training (opcional)
    compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed,testing=True, image_test=None, supervised=supervised, inception=False,show_latents=True,show_latents_test=False)
    
    #Ploteamos loss contrastive
    if plot_contrastive:
      plt.figure(figsize=(10, 10))
      plt.plot(epoch_losses)
      plt.legend(['Training Losses'])
      plt.savefig('./src/resources/Images/ContrastiveLoss_'+str(config_fixed["epochs"])+'epochs')
      plt.show()

    if config_fixed["supervised_MLP"] == True:
      model_MLP = supervised_model().to(device)
      metrics,trained_modelMLP = train_supervised_model(model_MLP,trained_modelq,config_fixed,testing_training,K=1)
      torch.save(trained_modelMLP.state_dict(), os.path.join(path_model, output_model_file_MLP))
      #Ploteamos loss y accuracy MLP en supervised contrastive
      if plot_MLP:
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.xlabel('Epoch')
        plt.ylabel('NLLLoss')
        plt.plot(metrics["tr_losses"], label='train')
        plt.plot(metrics["te_losses"], label='eval')
        plt.legend()
        plt.subplot(2,1,2)
        plt.xlabel('Epoch')
        plt.ylabel('Eval Accuracy [%]')
        plt.plot(metrics["tr_accs"], label='train')
        plt.plot(metrics["te_accs"], label='eval')
        plt.legend()
        plt.savefig('./src/resources/Images/SupervisedMLPLoss_'+str(config_fixed["epochs_supervised"])+'epochs')
        plt.show()

  if testing:
    loaded1=False
    loaded2=False
    loaded=False
    trained_modelq = base_model(pretrained=False, arch=arch)
    trained_modelMLP = supervised_model()
    # Cargamos 
    if supervised_MLP==False or supervised==False:
      if os.path.exists(os.path.join(path_model, test_model_file)):
        trained_modelq.load_state_dict(torch.load(os.path.join(path_model, test_model_file),map_location=device))
        loaded=True
      elif os.path.exists(os.path.join(path_checkpoint, test_checkpoint_file)):  
        checkpoint = torch.load(os.path.join(path_checkpoint, test_checkpoint_file), map_location=device)
        trained_modelq.load_state_dict(checkpoint['model_state_dict'])
        loaded=True
    else:
      if os.path.exists(os.path.join(path_model, test_model_file)):
        trained_modelq.load_state_dict(torch.load(os.path.join(path_model, test_model_file),map_location=device))
        loaded1=True
      elif os.path.exists(os.path.join(path_checkpoint, test_checkpoint_file)):  
        checkpoint = torch.load(os.path.join(path_checkpoint, test_checkpoint_file), map_location=device)
        trained_modelq.load_state_dict(checkpoint['model_state_dict'])
        loaded1=True
      if os.path.exists(os.path.join(path_model,test_model_file_MLP)):
        trained_modelMLP.load_state_dict(torch.load(os.path.join(path_model, test_model_file_MLP),map_location=device))
        loaded2=True
      loaded = loaded1 and loaded2

    if loaded: 
      #Te extrae los latents del dataset de test para representarlos en el Tensorboard (despues de haber entrenado el modelo contrastive)
      compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed,testing=False, image_test=None, supervised=supervised, inception=False,show_latents=True,show_latents_test=True)   

      if supervised_MLP==False or supervised==False:
        #Te extrae los latents del dataset de training para calcular KMEANS y KNN (después de haber entrenado el modelo de contrastive)
        latents, labels, images, path, trained_modelq = compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed, testing=testing, image_test=None, supervised=True, inception=False,show_latents=False,show_latents_test=False)
        test_names = sorted(glob.glob(config_fixed['image_path_test']+'/*/*.bmp',recursive=True))
        print('K-MEANS METHOD')
        print('TOPK1')
        print(accuracy(latents=latents,modelq=trained_modelq,list_files_test=test_names,topk=1,nombres=labels, method='kmeans'))
        print('TOPK3')
        print(accuracy(latents=latents,modelq=trained_modelq,list_files_test=test_names,topk=3,nombres=labels, method='kmeans'))
        print('TOPK5')
        print(accuracy(latents=latents,modelq=trained_modelq,list_files_test=test_names,topk=5,nombres=labels, method='kmeans'))
        print('K-NEIGHBOORS METHOD')
        print('TOPK1')
        print(accuracy(latents=latents,modelq=trained_modelq,list_files_test=test_names,topk=1,nombres=labels, method='kneighboors'))
        print('TOPK3')
        print(accuracy(latents=latents,modelq=trained_modelq,list_files_test=test_names,topk=3,nombres=labels, method='kneighboors'))
        print('TOPK5')
        print(accuracy(latents=latents,modelq=trained_modelq,list_files_test=test_names,topk=5,nombres=labels, method='kneighboors'))
        silhouette(X=latents)

      else:
        print('Loss based on a Supervised-contrastive method with MLP head classifier')
        print('TOPK1')
        test_supervised_model(trained_modelq,trained_modelMLP,config_fixed,K=1)
        print('TOPK3')
        test_supervised_model(trained_modelq,trained_modelMLP,config_fixed,K=3)
        print('TOPK5')
        test_supervised_model(trained_modelq,trained_modelMLP,config_fixed,K=5)
    
    else:
      print('Model or checkpoint for testing not found')

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

