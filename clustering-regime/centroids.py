import model as m
import utils
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genarate the optimal centroids')
    parser.add_argument('--datapath', default='./data/0-10_400_47x63_re40.npy', type=str, help='datasets')
    args = parser.parse_args()
    print(args)
    
    utils.directory_create('models/clt5')
    utils.directory_create('models/clt30')
    utils.directory_create('models/clt100')

    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')
    
    # Dataset
    traindata = torch.FloatTensor(np.load(args.datapath)).to(device)
    for num_clt in tqdm([5, 30, 100]):
        for lsize in [2,3,5,8,12]:
            # Load a pretrained model
            encoder = m.Encoder(lsize).to(device)
            encoder.load_state_dict(torch.load('./models/en'+'_'+str(lsize)+'.pth'))
            
            # Reduced scheduling parameters
            latent_features = []
            with torch.no_grad():
                inputs = traindata.to(device)
                latent_var = encoder(inputs)
                latent_features = latent_var.cpu().numpy()
            
            # k-means clustering
            kmeans = KMeans(n_clusters=num_clt, random_state=0).fit(latent_features)
            predictions = kmeans.predict(latent_features)
            centroids = kmeans.cluster_centers_
            np.save('./models/clt'+str(num_clt)+'/centroids'+str(num_clt)+'_'+str(lsize)+'.npy', centroids)
            
    
