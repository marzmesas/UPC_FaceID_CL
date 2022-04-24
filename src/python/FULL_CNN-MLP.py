import os
import torch
import datetime
torch.manual_seed(1)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1)
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import OrderedDict
import glob
from PIL import Image
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Declaration of the device for the GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()

# Class of the custom dataset
class CustomDataset(Dataset):
  def __init__(self, filenames,labels, transform=None):
    self.list_files = filenames
    self.transform = transform
    self.labels = labels

  def __len__(self):
    return len(self.list_files)
  # Return 2 images and labels
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.list_files[idx]
    image = Image.open(img_name).convert('RGB')
    label = self.labels[idx]
    if self.transform:
      img_0 = self.transform(image)
    return {'image1': img_0,'label': label}
# Get a mean of a list
def get_mean_of_list(L):
    return sum(L) / len(L)
# Topk predictions for the accuracy 
def topkcorrect_predictions (predicted_batch,label_batch,topk=(1,)):
  maxk = max(topk)
  batch_size = label_batch.size(0)
  _, pred = predicted_batch.topk(k=maxk, dim=1)
  pred = pred.t() 
  target_reshaped = label_batch.view(1, -1).expand_as(pred)  # [B] -> [B, 1] -> [maxk, B]
  # Compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
  correct = (pred == target_reshaped)
  list_topk_accs = []
  for k in topk:
    ind_which_topk_matched_truth = correct[:k] 
    flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float() 
    tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
    topk_acc = tot_correct_topk  
    list_topk_accs.append(topk_acc)
  return list_topk_accs

def train_epoch(train_loader,model,optimizer,criterion,epoch):
  model.train()
  losses_train = []
  accs_train=[]
  for i, data in enumerate(train_loader,1):
    optimizer.zero_grad()
    image = data["image1"]
    image= image.to(device)
    y_actual = data["label"]
    y_actual = torch.tensor([Classes_Map[i] for i in y_actual])
    y_actual =y_actual.to(device)
    y_predicted = model(image)
    loss = criterion(y_predicted,y_actual)
    # Backpropagation
    loss.backward()
    losses_train.append(loss.data.item())
    acc_topk=topkcorrect_predictions(y_predicted,y_actual,(K,))
    acc = (acc_topk[0].item()*100)/data["image1"].shape[0]
    accs_train.append(acc)
    # Update the weights of the resnet
    optimizer.step()
    if i % 8 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.1f}'.format(
          epoch, i*len(data["image1"]), len(train_loader.dataset),
          100. * i / len(train_loader), loss.item(),
          acc))
  return np.mean(losses_train),np.mean(accs_train)

def test_epoch(test_loader,model,criterion):
  model.eval()
  test_loss=0
  acc_test=0
  with torch.no_grad():
    for i,data in enumerate(test_loader):
      image = data["image1"]
      image= image.to(device)
      y_actual = data["label"]
      y_actual = torch.tensor([Classes_Map[i] for i in y_actual])
      y_actual =y_actual.to(device)
      y_predicted = model(image)
      accuracy_topk = topkcorrect_predictions(y_predicted,y_actual,(K,))
      acc_test += accuracy_topk[0].item()
      test_loss += criterion(y_predicted, y_actual, reduction='sum').item() # sum up batch loss
  test_loss /= len(test_loader.dataset)
  test_acc = (acc_test/len(test_loader.dataset))*100
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, acc_test, len(test_loader.dataset), test_acc,))
  return test_loss,test_acc

# MODEL WITH RESNET18
resnet = models.resnet18(pretrained=False).to(device)

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(resnet.fc.in_features, 256)),
    ('added_relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(256, 128)),
    ('added_relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(128, 44)),
    ('logsoftmax', nn.LogSoftmax())
]))

resnet.fc = classifier.to(device)
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])

# TRAINING

root_folder_train = r'./Datasets/Cropped-IMGS-2-supervised-train'
train_names = sorted(glob.glob(root_folder_train+'/*/*.bmp',recursive=True))
names_train = random.sample(train_names, len(train_names))
labels_train = [(x.split('/')[-1])[0:4] for x in names_train]
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
dataset_train=CustomDataset(names_train,labels_train,transform)
dataloader_train = DataLoader(dataset_train,batch_size=16,shuffle=True)

Classes_List = sorted(os.listdir(root_folder_train))
Classes_Map = dict()
for i in range(len(Classes_List)):
  Classes_Map[Classes_List[i]] = i

tr_loss_epoch = []
tr_accs_epoch = []
te_loss_epoch = []
te_acc_epoch = []
num_epochs = 50
criterion = F.nll_loss
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir) 


# TESTING
root_folder_test = r'./Datasets/Cropped-IMGS-2-supervised-test'
test_names = sorted(glob.glob(root_folder_test+'/*/*.bmp',recursive=True))
names_test = random.sample(test_names, len(test_names))
labels_test = [(x.split('/')[-1])[0:4] for x in names_test]
dataset_test = CustomDataset(names_test,labels_test,transform)
dataloader_test = DataLoader(dataset_test,batch_size=16,shuffle=True)


# TOPK PARAMETER
K=5
plot=True
#---------------------


for epoch in range(0,num_epochs):
  tr_loss,tr_acc=train_epoch(dataloader_train,resnet,optimizer,criterion,epoch)
  te_loss,te_acc=test_epoch(dataloader_test,resnet,criterion)
  tr_loss_epoch.append(tr_loss)
  tr_accs_epoch.append(tr_acc)
  te_loss_epoch.append(te_loss)
  te_acc_epoch.append(te_acc)
  writer.add_scalar("train loss", tr_loss_epoch[epoch], epoch)
  writer.add_scalar("train acc",tr_accs_epoch[epoch],epoch)
  writer.add_scalar("test loss",te_loss_epoch[epoch],epoch)
  writer.add_scalar("test acc",te_acc_epoch[epoch],epoch)
  rets = {'tr_losses':tr_loss_epoch, 'te_losses':te_loss_epoch,
          'tr_accs':tr_accs_epoch, 'te_accs':te_acc_epoch}

if plot:
  plt.figure(figsize=(10, 8))
  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('NLLLoss')
  plt.plot(tr_loss_epoch, label='train')
  plt.plot(te_loss_epoch, label='eval')
  plt.legend()
  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Eval Accuracy [%]')
  plt.plot(tr_accs_epoch, label='train')
  plt.plot(te_acc_epoch, label='eval')
  plt.legend()
  plt.show()

torch.save(resnet.state_dict(), './src/python/saved_models/modelFullCNNLinear.pt')
