import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import datetime
from statistics import mean

from dataset import CustomDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from ray import tune

#InfoNCE loss function
def loss_function(q, k, queue, tau):

    # N es el batch size
    N = q.shape[0]
    
    # C es la dimensionalidad de las representaciones
    C = q.shape[1]

    #Batch matrix multiplication (las query con las keys por los batch)
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),tau))
    
    #Matrix multiplication (las query con la queue acumulada (memory bank))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),tau)), dim=1)
   
    #Calculamos el denominador de la función de coste
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))

def train_model(config, config_fixed, resnetq, resnetk, writer, optimization):
    
    optimizer = optim.SGD(resnetq.parameters(), lr=config["lr"], momentum=config_fixed["momentum_optimizer"], weight_decay=config_fixed["weight_decay"])
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
    dataset_queue = CustomDataset(config_fixed['image_path'],transform)
    data_loader_queue = DataLoader(dataset=dataset_queue,batch_size=config["batch_size"],shuffle=True)
    dataset = CustomDataset(config_fixed['image_path'],transform)
    data_loader = DataLoader(dataset=dataset,batch_size=config["batch_size"],shuffle=True)

    epoch_losses_train = []
    num_epochs = 0
    flag = 0
    queue = None

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
                        if queue.shape[0] < config_fixed["K"]:
                            queue = torch.cat((queue, k), 0)    
                        else:
                            flag = 1
                
                    if flag == 1:
                        break

            if flag == 1:
                break

    #TRAINING
    resnetq.train()

    for epoch in range(config["epochs"]):
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
            loss = loss_function(q, k, queue, config["tau"])
            losses_train.append(loss.cpu().data.item())

            #Backpropagation
            loss.backward()

            #Actualización de los weights del resnet de las querys
            optimizer.step()
            queue = torch.cat((queue,k), 0) 
            #Si el tamaño de nuestro memory bank (queue) es mayor a 2000, eliminamos el último batch (batch size=16)
            if queue.shape[0] > config_fixed["K"]:
                queue = queue[config["batch_size"]:,:]

            #Actualizamos el resnet de las keys
            for θ_k, θ_q in zip(resnetk.parameters(), resnetq.parameters()):
                θ_k.data.copy_(config["momentum"]*θ_k.data + θ_q.data*(1.0 - config["momentum"]))
        
        #Guardamos las losses y hacemos la media, así como la metemos en el writer para el tensorboard
        epoch_losses_train.append(mean(losses_train))
        print(f'Epoch {epoch} - Loss: {epoch_losses_train[epoch]}')
            #'Epoch'+str(epoch)+":"+" Loss: "+str(epoch_losses_train[epoch]))
        if not optimization:
            writer.add_scalar("train loss", epoch_losses_train[epoch], epoch)
        else:
            tune.report(loss=mean(losses_train))

    return resnetq, resnetk
