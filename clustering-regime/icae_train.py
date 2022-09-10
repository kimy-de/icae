import data
import model as m
import utils
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time

class TensorData(Dataset):
    def __init__(self, x_data):
        self.x_data = torch.FloatTensor(x_data) 
        self.len = self.x_data.size(0) 
    def __getitem__(self, index):
        return self.x_data[index]
    def __len__(self):
        return self.len 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Individual Convolutional Autoencoder')
    parser.add_argument('--datapath', default='./data/0-10_400_47x63_re40.npy', type=str, help='datasets')
    parser.add_argument('--num_epochs', default=15001, type=int, help='number of epochs')  
    parser.add_argument('--latent_size', default=12, type=int, help='size of latent vector') 
    parser.add_argument('--num_clusters', default=5, type=int, help='number of clusters') 
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()
    print(args)
    
    utils.directory_create('models/de'+str(args.latent_size)+'_clt'+str(args.num_clusters))

    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')
    
    # Freeze the pretrained encoder
    encoder = m.Encoder(args.latent_size).to(device)
    encoder.load_state_dict(torch.load('./models/en'+'_'+str(args.latent_size)+'.pth'))
    for i, (name, param) in enumerate(encoder.named_parameters()):
        param.requires_grad = False

    # Centroids
    centroids = np.load('./models/clt'+str(args.num_clusters)+'/centroids'+str(args.num_clusters)+'_'+str(args.latent_size)+'.npy')
    
    # Dataset
    trainloader = data.datasets(args.datapath, 200, True)
    velocity = []
    labels = []
    
    with torch.no_grad():
        for images in trainloader: 
            latent_var = encoder(images.to(device))
            predictions = utils.kmean_predict(latent_var.detach().cpu().numpy(), centroids)
            labels += predictions
            b = images.size(0) 
            velocity += images.tolist()
    
    velocity = np.array(velocity)
    state_vec_in_cluster = []
    labels = np.array(labels)
    print(f"Snapshots: {velocity.shape}, Labels: {len(labels)}")
    
    for l in range(args.num_clusters):
        idx = np.where(labels == l)[0]
        vec_group = velocity[idx]
        vec_group_tensor = TensorData(vec_group) 
        vec_group_dataloader = DataLoader(vec_group_tensor, batch_size=16, shuffle=True) 
        state_vec_in_cluster.append(vec_group_dataloader)
 
    # Train iCAEs
    print("Training..")
    start = time.time()
    criterion = torch.nn.MSELoss()
    for l in range(args.num_clusters):
        sw = 0
        print(f'############# Train a model in Cluster {l} #############')
        decoder = m.Decoder(args.latent_size).to(device)
        decoder.load_state_dict(torch.load('./models/de'+'_'+str(args.latent_size)+'.pth'))
        
        optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        ls = 1 # loss threshold to save model parameters

        for ep in range(args.num_epochs):
            running_loss = 0
            for images in state_vec_in_cluster[l]:
                inputs = images.to(device)
                optimizer.zero_grad()
                latent_var = encoder(inputs)
                outputs = decoder(latent_var)
                loss = criterion(inputs, outputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
            avg_loss = running_loss / len(state_vec_in_cluster[l])
            
            if avg_loss < ls:
                ls = avg_loss
                bep = ep
                torch.save(decoder.state_dict(),'./models/de'+str(args.latent_size)+'_clt'+str(args.num_clusters)+'/de'+str(args.latent_size)+'_'+str(l)+'.pth')
                sw = 1
                
            if (ep % 5000 == 0) and (ep > 0):
                if sw == 1:
                    print('[BEST][%d / %d] Train loss: %.5f' %(bep, ep, ls)) 
                else:
                    print('[Epoch][%d] Train loss: %.5f' %(ep, avg_loss)) 
            
        print('[BEST][%d] Train loss: %.5f' %(bep, ls)) 
        end = time.time()
        print('Training runtime: %.5f' %(end-start)) 
