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

from os.path import join
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter


class CustomDataset(Dataset):
  def __init__(self, path, transform=None):
    self.list_files = [[join(path, x, y) for y in os.listdir(join(path, x))]
                       for x in os.listdir(path)]
    self.transform = transform

  def __len__(self):
    return len(self.list_files)

  def __getitem__(self, idx):
    imgs_paths = sample(self.list_files[idx], 2)

    img_0 = io.imread(imgs_paths[0])
    img_1 = io.imread(imgs_paths[1])

    if self.transform:
      img_0 = self.transform(img_0).unsqueeze(0)
      img_1 = self.transform(img_1).unsqueeze(0)

    return torch.cat([img_0, img_1])

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
    self.cnn1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
    self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
    self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=3)

    self.fc_0 = nn.Linear(in_features=1568, out_features=128)
    self.fc_1 = nn.Linear(in_features=128, out_features=32)

  def forward(self, x):
    out = self.relu(self.pool(self.cnn0(x)))
    out = self.relu(self.pool(self.cnn1(out)))
    out = self.relu(self.pool(self.cnn2(out)))
    batch, channels, width, height = out.shape
    out = out.reshape(batch, -1)
    out = self.relu(self.fc_0(out))
    return self.fc_1(out)

def log_embeddings(model, data_loader, writer):
  # take a few batches from the training loader
  list_latent = []
  list_images = []
  for i in range(16):
    for batch in data_loader:
      imgs = batch[:, 0]

      # forward batch through the encoder
      list_latent.append(model(imgs.to(device)))
      list_images.append(imgs)

  latent = torch.cat(list_latent)
  images = torch.cat(list_images)

  writer.add_embedding(latent, label_img=images)

def loss_function(preds):
  labels = torch.arange(preds.shape[0], device=device).long()
  return criterion(preds, labels)

def forward_images(m0, m1, data):
  imgs_0, imgs_1 = data[:, 0], data[:, 1]
  imgs_0, imgs_1 = imgs_0.to(device), imgs_1.to(device)
  preds_0 = m0(imgs_0)
  preds_1 = m1(imgs_1)
  result = torch.matmul(preds_0, preds_1.T)
  output = result - torch.max(result, 1)[0].unsqueeze(dim=1)
  return output


def forward_step(m0, m1, loader, optimizer):
  loss_list = []
  for i, data in enumerate(loader):
    output = forward_images(m0, m1, data)
    loss = loss_function(output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

  return loss_list


path = './GTV-Database-UPC'
transform = transforms.Compose(
  [transforms.ToTensor(), transforms.CenterCrop((240, 240))])

dataset = CustomDataset(path, transform=transform)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir)


m0 = CNN().to(device)
m1 = CNN().to(device)

m1.load_state_dict(m0.state_dict())

criterion = nn.CrossEntropyLoss()

lr=1e-3
epochs=40
batch_size = 16

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(params=m0.parameters(), lr=lr, weight_decay=1e-5)

for epoch in range(epochs):
    train_loss = forward_step(m0, m1, data_loader, optimizer)
    train_loss_avg = np.mean(train_loss)
    if epoch % 2 == 0:
        m1.load_state_dict(m0.state_dict())
    print(f"Epoch #{epoch} loss:{round(train_loss_avg,2)}")
    writer.add_scalar("train loss", train_loss_avg.item(), epoch)

log_embeddings(m0, data_loader, writer)
