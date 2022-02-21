
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def log_embeddings(model, data_loader, writer):
  model=model.to(device)
  list_latent = []
  list_images = []
  list_labels = []

  for i in range(16):
    for i,batch in enumerate(data_loader):
      imgs = batch['image1'].to(device)
      labels = batch['label']
      latent=model(imgs)
      list_latent.append(latent.detach().cpu())
      list_images.append(imgs.to('cpu'))
      list_labels.append(labels)

  latent = torch.cat(list_latent)
  images = torch.cat(list_images)
  labels = torch.cat(list_labels)
  #Guardamos los embeddings para el tensorboard y su representación con PCA,T-SNE
  writer.add_embedding(latent,metadata=labels,label_img=images)
  return latent,labels,images