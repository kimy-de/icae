import model as m
import utils
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample test')
    parser.add_argument('--datapath', default='./data/0-10_800_47x63_re40.npy', type=str, help='datasets')
    parser.add_argument('--latent_size', default=2, type=int, help='size of latent vector') 
    args = parser.parse_args()
    print(args)
    utils.directory_create('results')
    
    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')
    
    # Datasets
    testdata = torch.FloatTensor(np.load(args.datapath)).to(device)
    len_tdata = len(testdata)
    
    # POD
    lsize = args.latent_size
    V = torch.FloatTensor(np.load('./data/V_pod'+str(lsize)+'.npy')).to(device)
    testvec = testdata.reshape(len_tdata,-1)
    reduced_param  = (V@ testvec[:, :, None]).squeeze(-1)
    pod_rec = (V.T@ reduced_param[:, :, None]).squeeze(-1).cpu().numpy()
    
    # Convolutional Autoencoder (CAE)
    encoder = m.Encoder(lsize).to(device)
    decoder = m.Decoder(lsize).to(device)
    encoder.load_state_dict(torch.load('./models/en'+'_'+str(lsize)+'.pth',map_location=device))
    decoder.load_state_dict(torch.load('./models/de'+'_'+str(lsize)+'.pth',map_location=device))
    
    # Individual Convolutional Autoencoder (iCAE)
    #de5 = []
    de30 = []
    #for i in range(5):
    #    decoder5 = m.Decoder(lsize).to(device)
    #    decoder5.load_state_dict(torch.load('./models/de'+str(lsize)+'_clt5/de'+str(lsize)+'_'+str(i)+'.pth'))
    #    de5.append(decoder5)
        
    for i in range(30):
        decoder30 = m.Decoder(lsize).to(device)
        decoder30.load_state_dict(torch.load('./models/de'+str(lsize)+'_clt30/de'+str(lsize)+'_'+str(i)+'.pth',map_location=device))
        de30.append(decoder30) 

    #centr5 = np.load('./models/clt5/centroids5_'+str(lsize)+'.npy')
    centr30 = np.load('./models/clt30/centroids30_'+str(lsize)+'.npy')

    with torch.no_grad():
        inputs = testdata.to(device)
        latent_var = encoder(inputs)
        
        # CAE
        cae_rec = decoder(latent_var).cpu().numpy().reshape(len_tdata,-1)
            
        # iCAE5, 30
        #icae5_rec = []
        icae30_rec = []
        #pred5 = utils.kmean_predict(latent_var.cpu().numpy(), centr5)
        pred30 = utils.kmean_predict(latent_var.cpu().numpy(), centr30)
        for i in range(inputs.size(0)):
            #m5 = de5[int(pred5[i])]
            m30 = de30[int(pred30[i])]
            #icae5_rec.append(m5(latent_var[i:i+1]).cpu().reshape(-1).tolist())
            icae30_rec.append(m30(latent_var[i:i+1]).cpu().reshape(-1).tolist())
            
    #icae5_rec = np.array(icae5_rec)
    icae30_rec = np.array(icae30_rec)
    testdata = testdata.detach().cpu().numpy().reshape(len_tdata,-1)
    
    k1 = 1292 # x-velocity point
    k2 = 2552 # x-velocity point
    k3 = k1 + 2961 # y-velocity point
    k4 = k2 + 2961 # y-velocity point
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(15,10))
    fig.suptitle('Comparison between the baseline and reconstructions (2d)')
    
    ax1.plot(testdata[:,k1],'--', linewidth=2.0)
    ax1.plot(pod_rec[:,k1], color='orange',linewidth=1.0)
    ax1.plot(cae_rec[:,k1], color='blue',linewidth=1.0)
    ax1.plot(icae30_rec[:,k1], color='red',linewidth=1.0)
    ax1.legend(['baseline','POD','CAE','iCAE30'])
    ax1.set_ylabel("$v_{%d}$" %k1, fontsize=15)
    
    ax2.plot(testdata[:,k2],'--',linewidth=2.0)
    ax2.plot(pod_rec[:,k2], color='orange',linewidth=1.0)
    ax2.plot(cae_rec[:,k2], color='blue',linewidth=1.0)
    ax2.plot(icae30_rec[:,k2], color='red',linewidth=1.0)
    ax2.legend(['baseline','POD','CAE','iCAE30'])
    ax2.set_ylabel("$v_{%d}$" %k2, fontsize=15)
    
    ax3.plot(testdata[:,k3],'--',linewidth=2.0)
    ax3.plot(pod_rec[:,k3], color='orange',linewidth=1.0)
    ax3.plot(cae_rec[:,k3], color='blue',linewidth=1.0)
    ax3.plot(icae30_rec[:,k3], color='red',linewidth=1.0)
    ax3.legend(['baseline','POD','CAE','iCAE30'])
    ax3.set_ylabel("$v_{%d}$" %k3, fontsize=15)
    
    ax4.plot(testdata[:,k4],'--',linewidth=2.0)
    ax4.plot(pod_rec[:,k4], color='orange',linewidth=1.0)
    ax4.plot(cae_rec[:,k4], color='blue',linewidth=1.0)
    ax4.plot(icae30_rec[:,k4], color='red',linewidth=1.0)
    ax4.legend(['baseline','POD','CAE','iCAE30'])
    ax4.set_xlabel("time step $T$", fontsize=10)
    ax4.set_ylabel("$v_{%d}$" %k4, fontsize=15)
    plt.savefig('./results/trajectory_'+str(lsize)+'.png')
