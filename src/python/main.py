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

#Declaration of the device for the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


############# - Environment ####################

# Type of training
# supervised --> True
# self-supervised --> False
# supervised_MLP (with a classifier --> True, without one (KMEANS, KNN) --> False)
supervised = True
supervised_MLP = False
plot_MLP = True  # Boolean to plot the loss in supervised contrastive MLP
plot_contrastive=True # Boolean to plot the contrastive loss
# Boolean to check the accuracy of the test dataset while training (Only in supervised contrastive MLP)
testing_training = True
path_img_training="../../Datasets/Cropped-IMGS-2-supervised-train"
path_img_testing ="../../Datasets/Cropped-IMGS-2-supervised-test"
path_model="./saved_models"
path_checkpoint="../../src/resources/checkpoints"

path_logs = "./logs"
if not(os.path.exists(path_logs)):
  os.mkdir(path_logs)

# Boolean to train
training = True
# Boolean to optimize
optim= False
# Boolean to test the predictions
testing = True

# Checkpoint file to load for training 
pretrained_checkpoint_file =  'None'
# File to load pre-trained weights
pretrained_model_file =  'None'

# File to save the trained contrastive model 
output_model_file = 'model_Contrastive.pt'
# File to save the test contrastive model 
test_model_file = output_model_file
# File to save the trained MLP model
output_model_file_MLP = 'model_MLP.pt'
# File to save the test MLP model 
test_model_file_MLP = output_model_file_MLP

# Checkpoint file to test
test_checkpoint_file = 'None'

# Available architectures:
# - Resnet18
# - Resnet50
# - Resnet101
# - Inception resnet v1
# - Vgg16

arch='resnet18'

# Size of the queue (for the MOCOV2 contrastive)
K=350
batch_size=32
epochs_contrastive=3
epochs_supervisedMLP=15
# Interval to save the checkpoints
checkpoint_interval=1000

######################################################

# Fixed parameters:
config_fixed={
          # Path of the training dataset
          "image_path": path_img_training,
          # Path of the test dataset
          "image_path_test": path_img_testing,
          "logs_path":path_logs,
          "tau": 0.05,
          "epochs":epochs_contrastive,
          "epochs_supervised":epochs_supervisedMLP,
          "momentum_optimizer": 0.9,
          "weight_decay": 1e-6,
          "checkpoint_interval": checkpoint_interval,
          "supervised": supervised,
          "supervised_MLP":supervised_MLP
      }

config={}
config['lr']= 0.001
config['batch_size']=batch_size
config['momentum']=0.999
config['K']=K

def inference():
  # Model definition
  modelq = base_model(pretrained=False, arch=arch).to(device)
  # Load checkpoints if they exist
  if os.path.exists(os.path.join(path_checkpoint, pretrained_checkpoint_file)):
    checkpoint = torch.load(os.path.join(path_checkpoint, pretrained_checkpoint_file), map_location=device)
    modelq.load_state_dict(checkpoint['model_state_dict'])
    optim_state=checkpoint['optimizer_state_dict']
  else:
    optim_state=None
  # Load pre-trained weights if they exist
  if os.path.exists(os.path.join(path_model, pretrained_model_file)):  
    modelq.load_state_dict(torch.load(os.path.join(path_model, pretrained_model_file),map_location=device))
  # Copy the encoder-q to the encoder-k
  modelk = copy.deepcopy(modelq)

  # Model training:
  # Fixed parameters if optimize is set to False
  if not optim:
    config={}
    config['lr']= 0.001
    config['batch_size']=batch_size
    config['momentum']=0.999
    config['K']=K
    
    if training:
      if supervised:
        if supervised_MLP:
          print("Training based on a supervised contrastive loss with MLP classifier:")
        else:
          print("Training based on a supervised contrastive loss without classifier:")
      else:
        print("Training based on a self-supervised contrastive loss:")
      # Training
      trained_modelq, trained_modelk,epoch_losses=train_model(config, config_fixed, modelq, modelk, optim_state)

      # Save trained contrastive model
      torch.save(trained_modelq.state_dict(), os.path.join(path_model, output_model_file))

      # Compute the embeddings and plot it on Tensorboard for the training dataset (with the trained model)
      compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed,testing=True, image_test=None, supervised=supervised, inception=False,show_latents=True,show_latents_test=False)
      
      #Plot the contrastive loss
      if plot_contrastive:
        plt.figure(figsize=(10, 10))
        plt.plot(epoch_losses)
        plt.legend(['Training Losses'])
        plt.savefig('../resources/Images/ContrastiveLoss_'+str(config_fixed["epochs"])+'epochs')
        plt.show()

      if config_fixed["supervised_MLP"] == True:
        model_MLP = supervised_model().to(device)
        metrics,trained_modelMLP = train_supervised_model(model_MLP,trained_modelq,config_fixed,testing_training,K=1)
        torch.save(trained_modelMLP.state_dict(), os.path.join(path_model, output_model_file_MLP))
        #Plot loss and accuracy in a contrastive supervised MLP model
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
          plt.savefig('../resources/Images/SupervisedMLPLoss_'+str(config_fixed["epochs_supervised"])+'epochs')
          plt.show()

    if testing:
      loaded1=False
      loaded2=False
      loaded=False
      trained_modelq = base_model(pretrained=False, arch=arch)
      trained_modelMLP = supervised_model()
      # LOAD: 
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
        # Compute the latents of the test dataset to represent them on tensorboard (with the trained model)
        compute_embeddings(modelq=trained_modelq, config=config, config_fixed=config_fixed,testing=False, image_test=None, supervised=supervised, inception=False,show_latents=True,show_latents_test=True)   

        if supervised_MLP==False or supervised==False:
          # Compute the latents of the training dataset to compute the KMEANS and the KNearest Neighbors 
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

    count = 10 # Number of runs to execute
    config=None
    wandb.agent(sweep_id, partial(train_model, config, config_fixed, modelq, modelk), count=count)

# Run Main

if __name__ == '__main__':
  inference()

