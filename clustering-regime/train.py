import data
import model as m
import utils
import argparse
import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)    
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight)    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Convolutional Autoencoder')
    parser.add_argument('--datapath', default='./data/0-10_400_47x63_re40.npy', type=str, help='datasets')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')   
    parser.add_argument('--num_epochs', default=4001, type=int, help='number of epochs')  
    parser.add_argument('--latent_size', default=12, type=int, help='size of latent vector') 
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization penalty')
    parser.add_argument('--pret', default=None, type=int, help='pretrained model')
    args = parser.parse_args()
    print(args)
    
    utils.directory_create('models')

    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')
    
    # Convolutional Autoencoder (CAE)
    encoder = m.Encoder(args.latent_size).to(device)
    decoder = m.Decoder(args.latent_size).to(device)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    
    # Pretrained model
    if args.pret != None:
        encoder.load_state_dict(torch.load('./models/en'+'_'+str(args.latent_size)+'.pth'))
        decoder.load_state_dict(torch.load('./models/de'+'_'+str(args.latent_size)+'.pth'))
        
    # Dataset
    trainloader = data.datasets(args.datapath, args.batch_size, True)
 
    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.l2)
    
    # Training a CAE
    sw = 0
    ls = 100 # loss threshold to save model parameters 0.006
    bep = 0
    print("Training..")
    for ep in range(args.num_epochs):
        running_loss = 0.0

        for images in trainloader:
            inputs = images.to(device)
            optimizer.zero_grad()
            latent_var = encoder(inputs)
            outputs = decoder(latent_var)
            loss = criterion(inputs, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)       

        if avg_loss < ls:
            ls = avg_loss
            bep = ep	
            sw = 1
            torch.save(encoder.state_dict(),'./models/en'+'_'+str(args.latent_size)+'.pth')
            torch.save(decoder.state_dict(),'./models/de'+'_'+str(args.latent_size)+'.pth')
        	
        if (ep % 1000 == 0) and (ep > 0):
            if sw == 1:
                print('[BEST][%d / %d] Train loss: %.5f' %(bep, ep, ls)) 
            else:
                print('[Epoch][%d] Train loss: %.5f' %(ep, avg_loss)) 
    
    print('[BEST][%d] Train loss: %.5f' %(bep, ls))            
    # Elbow method for optimal value of k in k-means
    encoder.load_state_dict(torch.load('./models/en'+'_'+str(args.latent_size)+'.pth'))
    utils.kmeans_clustering(trainloader, encoder, args.latent_size)
