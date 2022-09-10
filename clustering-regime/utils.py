import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def directory_create(dirName):
  try:
    # "model", "model/clt5", etc...
    os.mkdir(dirName)
    print("Directory " , dirName ,  ": Created ") 
  except FileExistsError:
    print("Directory " , dirName ,  ": Already exists")

def kmeans_clustering(dataloader, encoder, lsize):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_features = []
    
    with torch.no_grad():
        for images in dataloader: 
            inputs = images.to(device)
            latent_var = encoder(inputs)
            latent_features += latent_var.cpu().tolist()
    
    latent_features = np.array(latent_features)
    latent_features = np.unique(latent_features, axis=1)
    # Find an optimal elbow point
    dist = []
    k = [5, 10, 20, 30, 40, 50]
    for i in k:
      km = KMeans(n_clusters=i, random_state=0).fit(latent_features)
      dist.append(km.inertia_)

    directory_create('results')
    plt.plot(k, dist, '*-')
    plt.xlabel("number of clusters")
    plt.ylabel("inertia")
    plt.savefig("./results/elbow_"+str(lsize)+".png")

def kmean_predict(x, centroids):
  y_assign = []
  num_clusters = centroids.shape[0]
  latent_size = centroids.shape[1]
  for m in range(x.shape[0]):
      h = np.broadcast_to(x[m],(num_clusters, latent_size))
      assign = np.argmin(np.sum((h-centroids)**2,axis=1),axis=0)
      y_assign.append(assign.item())    
  
  return y_assign  
