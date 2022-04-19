import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset_Supervised, CustomDataset_Unsupervised, CustomDataset_Testing
from logging_moco import log_embeddings
from torchvision import transforms
import umap
from sklearn.cluster import OPTICS, KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
from scipy.spatial import distance
import glob
import random
import torch.nn.functional as F
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


from dataset import CustomDataset_supervised_Testing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_embeddings(modelq, config, config_fixed, testing=False, image_test=None, supervised=False, inception = False,show_latents=False,show_latents_test=False):
    
    modelq.eval()

    if inception:
        modelq = nn.Sequential(*list(modelq.children())[:-5])
    else:
        modelq.fc = nn.Sequential(*list(modelq.fc.children())[:-5])
    
    if testing:
        dataset_embedding = CustomDataset_Testing(config_fixed['image_path'])
        data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
    else:
        if supervised:
            dataset_embedding = CustomDataset_Supervised(config_fixed['image_path'])
            data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
        else:
            dataset_embedding = CustomDataset_Unsupervised(config_fixed['image_path'])
            data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=True)
    if show_latents_test:
        dataset_embedding = CustomDataset_Testing(config_fixed['image_path_test'])
        data_loader_embedding = DataLoader(dataset=dataset_embedding,batch_size=config["batch_size"],shuffle=False)
    # Get embeddings
    # We get the labels, latents and images for graphic purposes
    latents, labels, images, path=log_embeddings(modelq, data_loader_embedding, testing, image_test,show_latents)

    return latents, labels, images, path, modelq

def graph_embeddings(latents, labels):

    # UMAP and PCA graphics.

    # UMAP
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

    # PCA 
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

    if len(nn.Sequential(*list(modelq.fc.children()))) == 5:
        modelq.fc = nn.Sequential(*list(modelq.fc.children())[:-5])

    tensor_sample = transform(image).to(device)
    tensor_sample = tensor_sample.unsqueeze(0)
    latent_sample = modelq(tensor_sample)
    latent_sample = latent_sample.to('cpu').detach().numpy()
    
    # OPTICS is used to obtain the Nº of clusters to use them with KMEANS method 
    # OPTICS is used and not DBSCAN because it only needs 1 parameter to optimize min_samples 
    # Clustering = OPTICS(min_samples=20).fit(latents)
    # n_clusters_=len(set(clustering.labels_))
    kmeans = KMeans(n_clusters=44, random_state=0).fit(latents)
    # We get the prediction of the closest cluster
    label_closest_cluster = kmeans.predict(latent_sample.astype(float))
    label_closest_cluster=label_closest_cluster[0]
    # Indexs of the elements that forms the cluster 
    idx = np.where(kmeans.labels_==label_closest_cluster)
    # Take num_neighboors of the elements that forms the cluster
    idx=idx[0][0:num_neighboors]
    # Restore the centroids of the clusters
    centroids = kmeans.cluster_centers_
    # Obtain the distance between the sample and the closest cluster
    distance_from_closest_centroid = distance.euclidean(centroids[label_closest_cluster], latent_sample)

    return distance_from_closest_centroid, idx

def accuracy(latents,modelq,list_files_test,topk,nombres,method):
  topk = topk
  accuracy=0

  kmeans = KMeans(n_clusters=44, random_state=0).fit(latents)
  neigh = NearestNeighbors(n_neighbors=topk+1)
  neigh.fit(latents)

  transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])

  for i in range(len(list_files_test)):
    img = cv2.imread(list_files_test[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nombre_groundtruth = list_files_test[i]
    nombre_groundtruth=nombre_groundtruth.split('/')[-2][-2:]
    res = transform(img).to(device)
    tensor_sample = res.unsqueeze(0)
    latent_sample = modelq(tensor_sample)
    latent_sample = latent_sample.to('cpu').detach().numpy()
    if method =='kmeans':
    # Predict the closest cluster
        label_closest_cluster = kmeans.predict(latent_sample.astype(float))
        label_closest_cluster=label_closest_cluster[0]
        # Obtain the index of the elements that forms the cluster
        idx = np.where(kmeans.labels_==label_closest_cluster)
        # Take num_neighboors of the elements that forms the cluster
        idx=idx[0][0:topk]
        list_labels = (nombres[idx]).tolist()
    elif method == 'kneighboors':
        dist, idx = neigh.kneighbors(latent_sample)
        idxs = idx[0]
        idxs=idxs[1:]
        list_labels = (nombres[idxs]).tolist()
        
    for i,_ in enumerate(list_labels):  
      nombre_prediccion = list_labels[i]
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
    plt.savefig('./src/resources/Images/Silhouette_Analysis')
    plt.show()




def topkcorrect_predictions (predicted_batch,label_batch,topk=(1,)):
  maxk = max(topk)
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


def test_supervised_model(network_contrastive,Linear_model,config_fixed,K):
    Linear_model.eval().to(device)
    network_contrastive.eval().to(device)
    test_loss=0
    acc_test=0
    criterion = F.nll_loss
    Classes_List = sorted(os.listdir(config_fixed["image_path_test"]))
    Classes_Map = dict()
    for i in range(len(Classes_List)):
        Classes_Map[Classes_List[i]] = i
    test_names = sorted(glob.glob(config_fixed["image_path_test"]+'/*/*.bmp',recursive=True))
    names_test = random.sample(test_names, len(test_names))
    labels_test = [(x.split('/')[-1])[0:4] for x in names_test]
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor()])
    dataset_test = CustomDataset_supervised_Testing(names_test,labels_test,transform)
    dataloader_test = DataLoader(dataset_test,batch_size=16,shuffle=False)

    if len(nn.Sequential(*list(network_contrastive.fc.children()))) == 5:
        network_contrastive.fc = nn.Sequential(*list(network_contrastive.fc.children())[:-5])
        
    with torch.no_grad():
        for i,data in enumerate(dataloader_test):
            image = data["image1"]
            image= image.to(device)
            y_actual = data["label"]
            y_actual = torch.tensor([Classes_Map[i] for i in y_actual])
            y_actual =y_actual.to(device)
            y_resnetq = network_contrastive(image)
            y_predicted = Linear_model(y_resnetq)
            accuracy_topk = topkcorrect_predictions(y_predicted,y_actual,(K,))
            acc_test += accuracy_topk[0].item()
            test_loss += criterion(y_predicted, y_actual, reduction='sum').item() # sum up batch loss
    test_loss /= len(dataloader_test.dataset)
    test_acc = (acc_test/len(dataloader_test.dataset))*100
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, acc_test, len(dataloader_test.dataset), test_acc,))
    return test_loss,test_acc
    
