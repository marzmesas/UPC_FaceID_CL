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

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_embeddings(modelq, config, config_fixed, writer, testing=False, image_test=None, supervised=False, inception = False):
    
    modelq.eval()
    
    if inception:
        modelq = nn.Sequential(*list(modelq.children())[:-5])
    else:
        modelq.fc = nn.Sequential(*list(modelq.fc.children())[:-5])
    

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

def silhouette(X):

    X=X.numpy()
    range_n_clusters = list(np.arange(20,80))

    silhouette_avg=[]

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=44)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg.append(silhouette_score(X, cluster_labels))
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg[-1],
        )
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

    silhouette_max = np.argmax(silhouette_avg)
    n_clusters = range_n_clusters[silhouette_max]
        
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=44)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

    plt.show()


    
