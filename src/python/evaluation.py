import os
import datetime
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from dataset import CustomDataset_Supervised, CustomDataset_Unsupervised, CustomDataset_Testing
from logging_moco import log_embeddings
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import umap
from sklearn.cluster import OPTICS, KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
from scipy.spatial import distance
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_embeddings(modelq, config, config_fixed, writer, testing=False, image_test=None, supervised=False, inception = False):
    
    modelq.eval()

    #if inception:
    #    modelq = nn.Sequential(*list(modelq.children())[:-5])
    #else:
    #    modelq.fc = nn.Sequential(*list(modelq.fc.children())[:-5])
    
    if testing:
        dataset_embedding = CustomDataset_Testing(config_fixed['image_path_test'])
        data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
    else:
        if supervised:
            dataset_embedding = CustomDataset_Supervised(config_fixed['image_path_test'])
            data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
        else:
            dataset_embedding = CustomDataset_Unsupervised(config_fixed['image_path'])
            data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
    
    #Calculamos los embeddings
    # Sacamos tambień los labels y las imágenes para graficar
    latents, labels, images, path=log_embeddings(modelq, data_loader_embedding, writer, testing, image_test)

    return latents, labels, images, path, modelq

def graph_embeddings(latents, labels):

    # Sacamos gráficas UMAP y PCA. Utilizamos los labels para pintar de distintos colores y ver
    # que estamos haciendo bien los clústers

    #UMAP
    UMAP_fig=plt.figure(1, figsize=(8, 6))
    reducer = umap.UMAP()
    standardized_data = StandardScaler().fit_transform(latents.numpy())
    embedding = reducer.fit_transform(standardized_data)
    custom_palette = sns.color_palette("hls", 44)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[custom_palette[x] for x in labels.tolist()])
    plt.gca().set_aspect('equal', 'datalim')

    # Compute OPTICS
    clustering = OPTICS(min_samples=20).fit(embedding)
    n_clusters_=len(set(clustering.labels_))
    plt.title('UMAP projection of the latents dataset - Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    #PCA 
    PCA_fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(PCA_fig, elev=-150, azim=110)
    pca=PCA(n_components=3)
    X_reduced = pca.fit_transform(latents.numpy())
    print(f'Explained variance with 3 components: {sum(pca.explained_variance_ratio_)*100}%')
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=labels,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )
    # Compute OPTICS
    clustering = OPTICS(min_samples=20).fit(X_reduced)
    n_clusters_=len(set(clustering.labels_))
    print("Estimated number of clusters PCA: %d" % n_clusters_)
    ax.set_title('First three PCA directions - Estimated number of clusters PCA: %d' % n_clusters_)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.pause(0)

def prediction(image, modelq, latents, num_neighboors=1):
        
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])
    modelq.eval()
    modelq.to(device)

    #modelq = nn.Sequential(*list(modelq.children())[:-5])

    if len(nn.Sequential(*list(modelq.fc.children()))) == 5:
        modelq.fc = nn.Sequential(*list(modelq.fc.children())[:-5])

    tensor_sample = transform(image).to(device)
    tensor_sample = tensor_sample.unsqueeze(0)
    latent_sample = modelq(tensor_sample)
    latent_sample = latent_sample.to('cpu').detach().numpy()
    
    # Utilizamos OPTICS para sacar el número de clusters para utilizar posteriormente con el kmeans
    # Utilizamos OPTICS frente al DBSCAN porque sólo hay el parámetro min_samples
    #clustering = OPTICS(min_samples=20).fit(latents)
    #n_clusters_=len(set(clustering.labels_))
    kmeans = KMeans(n_clusters=44, random_state=0).fit(latents)
    # Predecimos el cluster más próximo
    label_closest_cluster = kmeans.predict(latent_sample.astype(float))
    label_closest_cluster=label_closest_cluster[0]
    # Sacamos los índices de los elementos que forman el cluster
    idx = np.where(kmeans.labels_==label_closest_cluster)
    # Tomamos num_neighboors de los elementos que forman el cluster
    idx=idx[0][0:num_neighboors]
    # Recuperamos los centroids de los clusters
    centroids = kmeans.cluster_centers_
    # Calculamos la distancia entre la muestra y el centroid más cercano
    distance_from_closest_centroid = distance.euclidean(centroids[label_closest_cluster], latent_sample)

    return distance_from_closest_centroid, idx

def accuracy(latents,images,path, modelq,list_files_test,topk,nombres, method):
  topk = topk
  accuracy=0

  kmeans = KMeans(n_clusters=44, random_state=0).fit(latents)
  neigh = NearestNeighbors(n_neighbors=topk+1)
  neigh.fit(latents)

  transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])

  for i in range(len(list_files_test)):
    #print(i)

    img = cv2.imread(list_files_test[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = Image.open(list_files_test[i]).convert('RGB')
    nombre_groundtruth = list_files_test[i]
    #print(nombre_groundtruth)
    nombre_groundtruth=nombre_groundtruth.split('/')[-2][-2:]
    #print(nombre_groundtruth)
    #print("el numero de la persona es: "+nombre_groundtruth)
    res = transform(img).to(device)
    tensor_sample = res.unsqueeze(0)
    latent_sample = modelq(tensor_sample)
    latent_sample = latent_sample.to('cpu').detach().numpy()
    if method =='kmeans':
    # Predecimos el cluster más próximo
        label_closest_cluster = kmeans.predict(latent_sample.astype(float))
        label_closest_cluster=label_closest_cluster[0]
        # Sacamos los índices de los elementos que forman el cluster
        idx = np.where(kmeans.labels_==label_closest_cluster)
        # Tomamos num_neighboors de los elementos que forman el cluster
        idx=idx[0][0:topk]
        list_labels = (nombres[idx]).tolist()
    elif method == 'kneighboors':
        dist, idx = neigh.kneighbors(latent_sample)
        idxs = idx[0]
        idxs=idxs[1:]
        list_labels = (nombres[idxs]).tolist()
        
        '''
        # Código para debugar
        imgs_path=[]
        
        for i,idx in enumerate(idxs):
            imgs_path.append(path[idx])
        image_stack = cv2.resize(img, (160,160))
        image_stack = cv2.rectangle(image_stack, (0,0), (160,160), (160,0,0), 10)
        image_stack = cv2.putText(image_stack, text='INPUT IMAGE', org=(10, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)

        for path_ in imgs_path:
        closer_image = cv2.imread(path_)
        closer_image = cv2.cvtColor(closer_image, cv2.COLOR_BGR2RGB)
        image_stack = np.hstack((image_stack, closer_image))
        
        plt.imshow(image_stack)
        plt.show()
        '''

    for i,_ in enumerate(list_labels):  
      nombre_prediccion = list_labels[i]
      #print(f'el numero de la persona predicha es {nombre_prediccion} y el groundtruth es {int(nombre_groundtruth)}')
      if nombre_prediccion == int((nombre_groundtruth)):
        accuracy+=1
        break
  
  accuracy = (accuracy/len(list_files_test))*100    
  return accuracy


    
