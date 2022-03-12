import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def log_embeddings(model, data_loader, writer):

  model=model.to(device)
  list_latent = []
  list_images = []
  list_labels = []

  # recorremos 40 veces el dataset con el dataloader (40x3) para coger más imágenes de cada id, esto se debe quitar en el unsupervised
  # he puesto 40 porque tenemos aproximadamente 40 imágenes por id, aunque habrá seguramente imágenes repetidas por el dataset coge aleatoriamente
  # imágenes dentro de cada carpeta
  for i in range(40):
    # recorremos el dataset, como sólo tenemos 44 id, lo recorreremos 3 veces con batch_size de 16, 16 y 12
    for batch in data_loader:
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
  #writer.add_embedding(latent,metadata=labels,label_img=images)
  return latent,labels,images