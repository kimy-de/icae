import argparse
import model as m
import torch
import numpy as np
from matplotlib import pyplot as plt
import utils
plt.style.use('seaborn-dark-palette')
plt.style.use('seaborn-bright')
plt.style.use('seaborn-deep')
        
def l2distance(a, b):    
    return torch.argmin(torch.sum((a-b)**2,dim=1),dim=0)

def argminl2(centroids, latent_var):
  y_assign = []
  for m in range(latent_var.size(0)):
      h = latent_var[m].expand(len(centroids),-1)
      assign = l2distance(h, torch.FloatTensor(centroids).to(device))
      y_assign.append(centroids[assign.item()])
  y_assign = np.array(y_assign)
  return torch.FloatTensor(y_assign).to(device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of CAE and CAE100')
    parser.add_argument('--datapath', default='./data/0-10_800_47x63_re40.npy', type=str, help='datasets')

    args = parser.parse_args()
    print(args)
    utils.directory_create('results')
    
    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')
    
    # Datasets
    print("Loading evaluation data..")
    testdata = torch.FloatTensor(np.load(args.datapath)).to(device)
    len_tdata = len(testdata)
    
    lsize_list = [2, 3, 5, 8, 12] 
    pod_errors = []
    cae_errors = []
    iclt5_errors = []
    iclt30_errors = []
    clt100_errors = []
    
    print("Evaluating..")
    for lsize in lsize_list:
        # POD
        V = torch.FloatTensor(np.load('./data/V_pod'+str(lsize)+'.npy')).to(device)
        testvec = testdata.reshape(len_tdata,-1)
        reduced_param  = (V@ testvec[:, :, None]).squeeze(-1)
        recon_param = (V.T@ reduced_param[:, :, None]).squeeze(-1)
        pod_errors.append(torch.mean((recon_param-testvec)**2).item())
        
        # Convolutional Autoencoder (CAE)
        encoder = m.Encoder(lsize).to(device)
        decoder = m.Decoder(lsize).to(device)
        encoder.load_state_dict(torch.load('./models/en'+'_'+str(lsize)+'.pth',map_location=device))
        decoder.load_state_dict(torch.load('./models/de'+'_'+str(lsize)+'.pth',map_location=device))
        
        
        # Individual Convolutional Autoencoder (iCAE)
        de5 = []
        de30 = []
        for i in range(5):
            decoder5 = m.Decoder(lsize).to(device)
            mdp = './models/de'+str(lsize)+'_clt5/de'+str(lsize)+'_'+str(i)+'.pth'
            decoder5.load_state_dict(torch.load(mdp,map_location=device))
            de5.append(decoder5)
            
        for i in range(30):
            decoder30 = m.Decoder(lsize).to(device)
            mdp = './models/de'+str(lsize)+'_clt30/de'+str(lsize)+'_'+str(i)+'.pth'
            decoder30.load_state_dict(torch.load(mdp,map_location=device))
            de30.append(decoder30)

        centr5 = np.load('./models/clt5/centroids5_'+str(lsize)+'.npy')
        centr30 = np.load('./models/clt30/centroids30_'+str(lsize)+'.npy')
        centr100 = np.load('./models/clt100/centroids100_'+str(lsize)+'.npy')
         
        avg_error5 = 0
        avg_error30 = 0
        with torch.no_grad():
            inputs = testdata.to(device)
            latent_var = encoder(inputs)
            
            # CAE
            outputs = decoder(latent_var)
            cae_errors.append(torch.mean((inputs-outputs)**2).item())

            # CAE100
            predictions = argminl2(centr100, latent_var)
            cltoutputs = decoder(predictions)
            clt100_errors.append(torch.mean((inputs-cltoutputs)**2).item())
            
            # iCAE5, 30
            pred5 = utils.kmean_predict(latent_var.cpu().numpy(), centr5)
            pred30 = utils.kmean_predict(latent_var.cpu().numpy(), centr30)
            for i in range(inputs.size(0)):
                m5 = de5[int(pred5[i])]
                m30 = de30[int(pred30[i])]
                outputs5 = m5(latent_var[i:i+1])
                outputs30 = m30(latent_var[i:i+1])
                avg_error5 += torch.mean((inputs[i:i+1]-outputs5)**2)
                avg_error30 += torch.mean((inputs[i:i+1]-outputs30)**2)

        iclt5_errors.append((avg_error5/len_tdata).item())
        iclt30_errors.append((avg_error30/len_tdata).item())
        
    print("CAE:", cae_errors)
    print("CAE100:", clt100_errors)
    print("iCAE5:", iclt5_errors)
    print("iCAE30:", iclt30_errors)
    
    fig= plt.figure(figsize=(12,6))
    l = [2,3,5,8,12]
    plt.plot(l,pod_errors,'-^')
    plt.plot(l,cae_errors,'-v')
    plt.plot(l,clt100_errors,'-o')
    plt.plot(l,iclt5_errors,'-s')
    plt.plot(l,iclt30_errors,'-s')
    plt.yticks(fontsize=20)
    plt.ylim(-1e-4, 0.011)
    plt.xticks(l, fontsize=20)
    plt.legend(['POD','CAE','CAE100','iCAE5','iCAE30'],prop={'size': 20})
    plt.xlabel("reduced dimension", fontsize='xx-large')
    plt.title(r"Averaged $l_2$ error $\frac{1}{T}\sum\parallel v_i-\tilde{v}_i\parallel^2_2$", fontsize='xx-large')
    plt.savefig('./results/recon_errors.png')
    plt.show()
