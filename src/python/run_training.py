import os
import torch

from evaluation import test_supervised_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from dataset import CustomDataset_Supervised, CustomDataset_Unsupervised, CustomDataset_supervised_Testing
from evaluation import topkcorrect_predictions
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import numpy as np
from torchvision import transforms
import random
import glob
import torch.nn.functional as F
import torch.nn as nn


os.environ['WANDB_MODE'] = 'offline'

# InfoNCE loss function
def loss_function(q, k, queue, tau):

    # N is the batch size
    N = q.shape[0]
    
    # C is the dimensionality of the representations 
    C = q.shape[1]

    # Batch matrix multiplication (querys x keys for all the batches)
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),tau))
    
    # Matrix multiplication (querys with the accumulated queue (memory bank))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),tau)), dim=1)
   
    # Computation of the denominator of the loss function
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))

def train_epoch(data_loader, modelq, modelk, optimizer, queue, config, config_fixed):
    cumu_loss=0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        xq, xk = data['image1'], data['image2']
        xq, xk = xq.to(device), xk.to(device)
        # Outputs
        q = modelq(xq)
        k = modelk(xk)
        k = k.detach()
        # Vector normalization
        q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
        k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))
        # Loss
        loss = loss_function(q, k, queue, config_fixed["tau"])
        cumu_loss+=loss.item()
        # Backpropagation
        loss.backward()
        # Update of the weights for the query
        optimizer.step()
        queue = torch.cat((queue,k), 0) 
        # If the size of our memory bank (queue) is bigger than the K value, the last batch is removed
        if queue.shape[0] > config["K"]:
            queue = queue[config["batch_size"]:,:]

        # Updating the resnet of the keys
        for θ_k, θ_q in zip(modelk.parameters(), modelq.parameters()):
            θ_k.data.copy_(config["momentum"]*θ_k.data + θ_q.data*(1.0 - config["momentum"]))    
    return cumu_loss/len(data_loader), queue

def train_model(config, config_fixed, modelq, modelk, optim_state=None):
    
    # Initialize a new wandb run
    with wandb.init(config=config):
         # If called by wandb.agent, as below,
        # This config will be set by Sweep Controller
        config = wandb.config

        optimizer = optim.SGD(modelq.parameters(), lr=config["lr"], momentum=config_fixed["momentum_optimizer"], weight_decay=config_fixed["weight_decay"])
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)

        if config_fixed["supervised"]:
            dataset_queue = CustomDataset_Supervised(config_fixed['image_path'])
            dataset = CustomDataset_Supervised(config_fixed['image_path'])
        else:
            dataset_queue = CustomDataset_Unsupervised(config_fixed['image_path']) 
            dataset = CustomDataset_Unsupervised(config_fixed['image_path'])
          
        data_loader_queue = DataLoader(dataset=dataset_queue,batch_size=config["batch_size"],shuffle=True)
        data_loader = DataLoader(dataset=dataset,batch_size=config["batch_size"],shuffle=True)

        num_epochs = 0
        flag = 0
        queue = None

        if queue is None:
            while True:
                with torch.no_grad():
                    for (_, data) in enumerate(data_loader_queue):            
                
                        xq, xk = data['image1'], data['image2']
                        xq, xk = xq.to(device),xk.to(device)
                        k = modelk(xk)
                        k = k.detach()
                        k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                        if queue is None:
                            queue = k
                        else:
                            if queue.shape[0] < config["K"]:
                                queue = torch.cat((queue, k), 0)    
                            else:
                                flag = 1
                
                        if flag == 1:
                            break

                if flag == 1:
                    break

        # TRAINING
        modelq.train()
        epoch_losses_train=[]
        for epoch in range(config_fixed["epochs"]):
            loss_train,queue=train_epoch(data_loader, modelq, modelk, optimizer, queue, config, config_fixed)
            print(f'Epoch {epoch} - Loss: {loss_train}')
            if epoch % config_fixed["checkpoint_interval"] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': modelq.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                    },  './src/resources/checkpoints/checkpoint_'+str(epoch)+'.pt')
            wandb.log({"loss": loss_train})
            epoch_losses_train.append(loss_train)

    return modelq, modelk,epoch_losses_train

def train_epoch_supervised(train_loader,network_contrastive,Linear_model,optimizer,criterion,epoch,K,Classes_Map):
  Linear_model.train()
  losses_train = []
  accs_train=[]
  for i, data in enumerate(train_loader,1):
    optimizer.zero_grad()
    image = data["image1"]
    image= image.to(device)
    y_actual = data["label"]
    y_actual = torch.tensor([Classes_Map[i] for i in y_actual])
    y_actual =y_actual.to(device)
    # Outputs
    with torch.no_grad():
      y_resnetq = network_contrastive(image)
    y_predicted = Linear_model(y_resnetq)
    loss = criterion(y_predicted,y_actual)
    # Backpropagation
    loss.backward()
    losses_train.append(loss.data.item())
    acc_topk=topkcorrect_predictions(y_predicted,y_actual,(K,))
    acc = (acc_topk[0].item()*100)/data["image1"].shape[0]
    accs_train.append(acc)
    # Updating of the weights in the resnet of the querys
    optimizer.step()
    if i % 8 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.1f}'.format(
          epoch, i*len(data["image1"]), len(train_loader.dataset),
          100. * i / len(train_loader), loss.item(),
          acc))
  return np.mean(losses_train),np.mean(accs_train)


def train_supervised_model(model_MLP,network_contrastive,config_fixed,testing_training,K):

    train_names = sorted(glob.glob(config_fixed["image_path"]+'/*/*.bmp',recursive=True))
    names_train = random.sample(train_names, len(train_names))
    labels_train = [(x.split('/')[-1])[0:4] for x in names_train]
    transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((160,160)), transforms.ToTensor()])
    dataset_train=CustomDataset_supervised_Testing(names_train,labels_train,transform)
    train_loader = DataLoader(dataset_train,batch_size=16,shuffle=True)
    optimizer = optim.SGD(model_MLP.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)
    criterion = F.nll_loss

    tr_loss_epoch=[]
    tr_accs_epoch=[]
    te_loss_epoch=[]
    te_accs_epoch=[]

    Classes_List = sorted(os.listdir(config_fixed["image_path"]))
    Classes_Map = dict()
    for i in range(len(Classes_List)):
        Classes_Map[Classes_List[i]] = i
    
    if len(nn.Sequential(*list(network_contrastive.fc.children()))) == 5:
        network_contrastive.fc = nn.Sequential(*list(network_contrastive.fc.children())[:-5])

    for epoch in range(0,config_fixed["epochs_supervised"]):
        tr_loss,tr_acc=train_epoch_supervised(train_loader,network_contrastive,model_MLP,optimizer,criterion,epoch,K,Classes_Map)
        tr_loss_epoch.append(tr_loss)
        tr_accs_epoch.append(tr_acc)
        if testing_training==True:
            te_loss,te_acc = test_supervised_model(network_contrastive,model_MLP,config_fixed,K)
            te_loss_epoch.append(te_loss)
            te_accs_epoch.append(te_acc)
        rets = {'tr_losses':tr_loss_epoch, 'te_losses':te_loss_epoch,'tr_accs':tr_accs_epoch, 'te_accs':te_accs_epoch}
    return rets,model_MLP