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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#writer = SummaryWriter(log_dir=logdir)

def graph_embeddings(modelq, config, config_fixed, writer):

    #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #writer = SummaryWriter(log_dir=logdir)

    modelq.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((240, 240))])
    dataset_embedding = CustomDataset(config_fixed["image_path"],transform)
    data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
    
    if len(nn.Sequential(*list(modelq.fc.children()))) == 5:
        modelq.fc = nn.Sequential(*list(modelq.fc.children())[:-3])
    
    latents,labels, images=log_embeddings(modelq, data_loader_embedding, writer)

    #UMAP
    UMAP_fig=plt.figure(1, figsize=(8, 6))
    reducer = umap.UMAP()
    standardized_data = StandardScaler().fit_transform(latents.numpy())
    embedding = reducer.fit_transform(standardized_data)
    embedding.shape
    custom_palette = sns.color_palette("hls", 44)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[custom_palette[x] for x in labels.tolist()])
    plt.gca().set_aspect('equal', 'datalim')

    # Compute DBSCAN UMAP
    db = DBSCAN(eps=0.3, min_samples=3).fit(embedding)
    labels_ = db.labels_
    n_clusters_ = len(set(labels_)) - (1 if -1 in labels_ else 0)
    #print("Estimated number of clusters UMAP: %d" % n_clusters_)
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
    # Compute DBSCAN PCA
    db = DBSCAN(eps=0.5, min_samples=3).fit(X_reduced)
    labels_ = db.labels_
    n_clusters_ = len(set(labels_)) - (1 if -1 in labels_ else 0)
    print("Estimated number of clusters PCA: %d" % n_clusters_)
    ax.set_title('First three PCA directions - Estimated number of clusters PCA: %d' % n_clusters_)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.pause(0)

    