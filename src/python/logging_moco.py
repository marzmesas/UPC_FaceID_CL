import torch
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def log_embeddings(model, data_loader, writer, testing=False, image_test = None):
    
  list_latent = []
  list_images = []
  list_labels = []
  list_path = []

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
  #Guardamos los embeddings para el tensorboard y su representación con PCA,T-SNE
  writer.add_embedding(latent,metadata=labels, label_img=images)

  return latent,labels, images, path
 