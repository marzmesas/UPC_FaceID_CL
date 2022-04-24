import torch
from torchvision import transforms
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# Function that logs latents of the images (training and test dataset)
def log_embeddings(model, data_loader,path_logs,testing=False, image_test = None,show_latents=False):
    
  list_latent = []
  list_images = []
  list_labels = []
  list_path = []
  if show_latents:
    if testing:
      logdir = os.path.join(path_logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_EmbeddingsTraining")
    else:
      logdir = os.path.join(path_logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_EmbeddingsTest")
    writer = SummaryWriter(log_dir=logdir)
  model=model.to(device)

  transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])
  for batch in data_loader:
    imgs = batch['image0'].to(device)
    labels = batch['label']
    latent=model(imgs)
    list_latent.append(latent.detach().cpu())
    list_images.append(imgs.to('cpu'))
    list_labels.append(labels)
    if testing:
      path = batch['path']
      list_path.append(path)

  if image_test is not None:
    image_test = transform(image_test).to(device)
    image_test = image_test.to(device)
    image_test = image_test.unsqueeze(0)
    latent = model(image_test)
    list_latent.append(latent.detach().cpu())
    list_images.append(image_test.to('cpu'))
    list_labels.append(labels)
    list_path.append('')

  latent = torch.cat(list_latent)
  images = torch.cat(list_images)
  labels = torch.cat(list_labels)
  if testing:
    path = [item for sublist in list_path for item in sublist]
  else:
    path=[]
  # Embeddings are saved for tensoboard representation
  if show_latents:
    writer.add_embedding(latent,metadata=labels, label_img=images)
  print("latents obtained")

  return latent,labels, images, path
 