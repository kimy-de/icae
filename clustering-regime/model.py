import torch
import torch.nn as nn

class Flatten(torch.nn.Module): 
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1) 
    
class Deflatten(nn.Module): 
    def __init__(self, k, a, b):
        super(Deflatten, self).__init__()
        self.k = k
        self.a = a
        self.b = b
        
    def forward(self, x):
        s = x.size()    
        return x.view(s[0],self.k, self.a, self.b)

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(2, 8, 7, stride=2, bias=False), 
                        nn.ELU(),
                        nn.Conv2d(8, 36, 7, stride=2, bias=False),
                        nn.ELU(),
                        nn.Conv2d(36, 51, 3, stride=1, bias=False),
                        nn.ELU(),
                        nn.Conv2d(51, 77, 3, stride=1, bias=False),
                        nn.ELU(),
                        nn.Conv2d(77, 132, 3, stride=1, bias=False),
                        nn.ELU(),
                        Flatten(),
                        nn.Linear(132*6*2, latent_size, bias=False),
                        nn.ELU()
                        )
    def forward(self, x): 
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
                        nn.Linear(latent_size, 132*6*2),
                        Deflatten(132, 6, 2),
                        nn.ConvTranspose2d(132, 77, 3, stride=1),
                        nn.ConvTranspose2d(77, 51, 3, stride=1),
                        nn.ConvTranspose2d(51, 36, 3, stride=1),
                        nn.ConvTranspose2d(36, 8, 7, stride=2),
                        nn.ConvTranspose2d(8, 2, 7, stride=2)
                        )
    def forward(self, x):      
        x = self.decoder(x)
        return x
