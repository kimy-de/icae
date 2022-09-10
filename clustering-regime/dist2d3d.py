import model as m
import utils
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample test')
    parser.add_argument('--datapath', default='./data/0-10_800_47x63_re40.npy', type=str, help='datasets')
    args = parser.parse_args()
    print(args)
    utils.directory_create('results')
 
    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')
    
    # Datasets
    testdata = torch.FloatTensor(np.load(args.datapath)).to(device)
    len_tdata = len(testdata)
    
    # CAEs
    encoder2 = m.Encoder(2).to(device)
    encoder3 = m.Encoder(3).to(device)
    encoder2.load_state_dict(torch.load('./models/en_2.pth',map_location=device))
    encoder3.load_state_dict(torch.load('./models/en_3.pth',map_location=device))

    # Centroids, k=5
    cent2 = np.load('./models/clt5/centroids5_2.npy')
    cent3 = np.load('./models/clt5/centroids5_3.npy')
    
    # 2d-, 3d- rho
    with torch.no_grad():
        inputs = testdata.to(device)
        rho2 = encoder2(inputs).cpu().numpy()
        rho3 = encoder3(inputs).cpu().numpy()
        pred2 = utils.kmean_predict(rho2, cent2)
        pred3 = utils.kmean_predict(rho3, cent3)
        
    optimal_k = 5
    reduced_param = np.concatenate([rho2, cent2], axis=0)
    pred_t = np.concatenate([pred2, np.array([optimal_k]*optimal_k)], axis=0)
    colors = ['orange','red','blue','green','gray']
    plt.figure(figsize=(10, 10))
    mm = range(optimal_k+1)
    for i, label in zip(range(optimal_k+1), mm):
        idx = np.where(pred_t == i)
        if i == optimal_k:
            pass
            plt.scatter(reduced_param[idx, 0], reduced_param[idx, 1], 
            marker='*', label="centroids", linewidths=5, color='black')
        else:
            plt.scatter(reduced_param[idx, 0], reduced_param[idx, 1], 
            marker='.', label=str(label),color=colors[i])

    plt.savefig('./results/rho2d_dist.png')
    plt.close()

    reduced_param = np.concatenate([rho3, cent3], axis=0)
    pred_t = np.concatenate([pred3, np.array([optimal_k]*optimal_k)], axis=0)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    mm = range(optimal_k+1)
    colors = ['orange','red','blue','green','gray']
    for i, label in zip(range(optimal_k+1), mm):
        idx = np.where(pred_t == i)
        if i == optimal_k:
            pass
            ax.scatter(reduced_param[idx, 0], reduced_param[idx, 1], reduced_param[idx, 2], 
            marker='*', label="centroids", linewidths=5, color='black')
        else:
            ax.scatter(reduced_param[idx, 0], reduced_param[idx, 1], reduced_param[idx, 2], 
            marker='o', label=str(label), color=colors[i])

    ax.view_init(-20, 20)
    ax.grid(False)
    plt.savefig('./results/rho3_dist.png')
    plt.close()
