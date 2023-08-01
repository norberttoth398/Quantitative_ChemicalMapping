# PyTorch Autoencoder
# Lot of this code is based on/inspired by code given to me by Po-Yen - greatly appreciated.
# Concepts/network here are very similar to those used in the Keras version.


import torch
import time
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
from utils import FeatureDataset, weights_init, element


class SemiSuperAutoencoder(nn.Module):
    def __init__(self,input_dim = 6, latent_dim = 2,hidden_layer_sizes=(64, 32), activation = nn.LeakyReLU(0.02)):
        super(SemiSuperAutoencoder, self).__init__()
        #make sure attributes are assigned
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        #create encoder architecture
        encoder = element(self.input_dim, self.hls[0], activation)
        for i in range(len(self.hls) - 1):
            encoder += element(self.hls[i], self.hls[i + 1], activation)
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        #create decoder architecture
        decoder = element(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += element(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] 

        #build encoder with bottleneck inside
        self.encode = nn.Sequential(*encoder)

        #build decoder
        self.decode = nn.Sequential(*decoder)
        
        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to layer before latent space
        return self.encode(x)
    
    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)
    
    def forward(self, x):
        en = self.encoded(x)
        return en
    
    
    def inference_numpy(self, x, batch_size = 200, n = 50):
        #transform real data to latent space using the trained model
        self.eval()
        latents=[]
        self.to(device)

        dataset_ = FeatureDataset(x)
        loader = DataLoader(dataset_,batch_size=batch_size,shuffle=False)
        
        with torch.no_grad():
            self.eval()
            for i, data in enumerate(loader):
                x = data.to(device)
                z = self.encoded(x)
                latents.append(z.detach().cpu().numpy())

        return np.concatenate(latents, axis=0)
    

    def train_model(self, optimizer, train_loader, test_loader, n_epoch, criterion):

        for epoch in range(n_epoch):
            
            # Training
            self.train()
            t = time.time()
            total_loss = []
            for i, data in enumerate(train_loader):
                x = data.to(device)
                x_recon = self(x)
                loss = criterion(x_recon, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss.append(loss.detach().item())
            
            # Testing
            self.eval()
            test_loss = []
            for i, test in enumerate(test_loader):
                x = test.to(device)
                x_recon = self(x)
                loss = criterion(x_recon, x)
                test_loss.append(loss.item())
            
            # Logging
            avg_loss = sum(total_loss) / len(total_loss)
            avg_test = sum(test_loss) / len(test_loss)
            training_time = time.time() - t
            
            print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')

    def save_model(self, optimizer, path):
        check_point = {'params': self.state_dict(),                            
                    'optimizer': optimizer.state_dict()}
        torch.save(check_point, path)

    def load_model(self, optimizer=None, path=''):
        check_point = torch.load(path)
        self.load_state_dict(check_point['params'])
        if optimizer is not None:
            optimizer.load_state_dict(check_point['potimizer'])


