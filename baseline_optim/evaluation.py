import os
import datetime
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from dataset import CustomDataset
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#writer = SummaryWriter(log_dir=logdir)

def compute_embeddings(modelq, config, config_fixed, writer=None):
    modelq.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
    dataset_embedding = CustomDataset(config_fixed["image_path"],transform)
    data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
    
    #Calculamos los embeddings
    # Sacamos tambień los labels y las imágenes para graficar
    latents,labels, images=log_embeddings(modelq, data_loader_embedding, writer)
    torch.save(latents, './saved_models/latents.pt')

    return latents, labels, images

def graph_embeddings(latents, labels):

    #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #writer = SummaryWriter(log_dir=logdir)

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
        
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
    modelq.to(device)
    modelq.eval()
    tensor_sample = transform(image).to(device)
    tensor_sample = tensor_sample.unsqueeze(0)
    latent_sample = modelq(tensor_sample)
    latent_sample = latent_sample.to('cpu').detach().numpy()
    
    # Algoritmo que encuentra los k-vecinos más próximos y devuelve la distancia a ellos (no a los centroides)
    # Si utilizamos un Kmeans debemos indicar el número de clústers y en principio es una cosa que no sabemos
    
        #neigh = NearestNeighbors(n_neighbors=num_neighboors)
        #neigh.fit(latents)
        #dist, idx = neigh.kneighbors(latent_sample)
        #dist = dist[0]
        #idx = idx[0]

    # Utilizamos OPTICS para sacar el número de clusters para utilizar posteriormente con el kmeans
    # Utilizamos OPTICS frente al DBSCAN porque sólo hay el parámetro min_samples
    clustering = OPTICS(min_samples=20).fit(latents)
    n_clusters_=len(set(clustering.labels_))
    kmeans = KMeans(n_clusters=n_clusters_, random_state=0).fit(latents)
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
