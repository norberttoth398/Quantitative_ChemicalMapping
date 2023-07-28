# PyTorch Autoencoder
# Lot of this code is based on/inspired by code given to me by Po-Yen - greatly appreciated.
# Concepts/network here are very similar to those used in the Keras version.


import torch
import time
from torch import nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class FeatureDataset(Dataset):
    def __init__(self, x):
        if len(x.shape)==2:
            self.x = x
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training

    def __len__(self):
        return self.x.shape[0] 
    
    def __getitem__(self, n): 
        return torch.Tensor(self.x[n])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Autoencoder(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(self.input_dim,512),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(512,256),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(256,128),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(128,self.latent_dim)
                                    )

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim,128),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(128,256),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(256,512),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(512,self.input_dim)
                                    )
        

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encoder(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decoder(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de

class Shallow_Autoencoder(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2):
        super(Shallow_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(self.input_dim,64),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(64,32),
                                     nn.BatchNorm1d(32),
                                     nn.LeakyReLU(0.02),
                                     
                                     
                                     nn.Linear(32,self.latent_dim)
                                    )

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim,32),
                                     nn.BatchNorm1d(32),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(32,64),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(64,self.input_dim)
                                    )
        

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encoder(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decoder(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de


class Tanh_Autoencoder(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2):
        super(Tanh_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(self.input_dim,64),
                                     #nn.LayerNorm(64),
                                     nn.Tanh(),
                                     #nn.BatchNorm1d(64),
                                     nn.Linear(64,32),
                                     #nn.LayerNorm(32),
                                     nn.Tanh(),
                                     #nn.BatchNorm1d(32),
                                     
                                     nn.Linear(32,self.latent_dim)
                                     )

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim,32),
                                     #nn.LayerNorm(32),
                                     nn.Tanh(),
                                     #nn.BatchNorm1d(32),
                                     nn.Linear(32,64),
                                     #nn.LayerNorm(64),
                                     nn.Tanh(),
                                     #nn.BatchNorm1d(64),
                                     nn.Linear(64,self.input_dim)
                                    )

    def encoded(self, x):
        #encodes data to latent space
        return self.encoder(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decoder(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de
    
class Tanh_Autoencoder_bn(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2):
        super(Tanh_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(self.input_dim,64),
                                     #nn.LayerNorm(64),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64,32),
                                     #nn.LayerNorm(32),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(32),
                                     
                                     nn.Linear(32,self.latent_dim)
                                     )

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim,32),
                                     #nn.LayerNorm(32),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(32),
                                     nn.Linear(32,64),
                                     #nn.LayerNorm(64),
                                     nn.Tanh(),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64,self.input_dim)
                                    )

    def encoded(self, x):
        #encodes data to latent space
        return self.encoder(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decoder(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de


class Autoencoder_dropOut(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2, dropout = 0.25):
        super(Autoencoder_dropOut, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.encoder = nn.Sequential(nn.Linear(self.input_dim,86),
                                     nn.LayerNorm(86),
                                     nn.LeakyReLU(0.02),
                                     nn.Dropout(self.dropout),
                                     
                                     nn.Linear(86,43),
                                     nn.LayerNorm(43),
                                     nn.LeakyReLU(0.02),
                                     nn.Dropout(self.dropout),
                                     
                                     
                                     nn.Linear(43,self.latent_dim),
                                     nn.Dropout(self.dropout),
                                    )

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim,43),
                                     nn.LayerNorm(43),
                                     nn.LeakyReLU(0.02),
                                     nn.Dropout(self.dropout),
                                     
                                     nn.Linear(43,86),
                                     nn.LayerNorm(86),
                                     nn.LeakyReLU(0.02),
                                     nn.Dropout(self.dropout),
                                     
                                     nn.Linear(86,self.input_dim),
                                     nn.Dropout(self.dropout),
                                    )

    def encoded(self, x):
        #encodes data to latent space
        return self.encoder(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decoder(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de





def train(model, optimizer, train_loader, test_loader, n_epoch, criterion):

    for epoch in range(n_epoch):
        
        # Training
        model.train()
        t = time.time()
        total_loss = []
        for i, data in enumerate(train_loader):
            x = data.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.detach().item())
        
        # Testing
        model.eval()
        test_loss = []
        for i, test in enumerate(test_loader):
            x = test.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            test_loss.append(loss.item())
        
        # Logging
        avg_loss = sum(total_loss) / len(total_loss)
        avg_test = sum(test_loss) / len(test_loss)
        training_time = time.time() - t
        
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')


def save_model(model, optimizer, path):
    check_point = {'params': model.state_dict(),                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

def load_model(model, optimizer=None, path=''):
    check_point = torch.load(path)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['potimizer'])

def enable_dropout(m):
  for m in m.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()

def getLatent(model, dataset:np):
    #transform real data to latent space using the trained model
    model.eval()

    enable_dropout(model)

    latents=[]
    model.to(device)

    dataset_ = FeatureDataset(dataset)
    loader = DataLoader(dataset_,batch_size=20,shuffle=False)
    
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(loader):
            x = data.to(device)
            z = model.encoded(x)
            latents.append(z.detach().cpu().numpy())

    model.eval()
    
    return np.concatenate(latents, axis=0)

def getMCLatent(model, dataset:np, MC_iter = 30):
    #transform real data to latent space using the trained model
    model.eval()

    enable_dropout(model)

    
    latent_stack = np.asarray([])
    model.to(device)

    dataset_ = FeatureDataset(dataset)
    loader = DataLoader(dataset_,batch_size=20,shuffle=False)
    
    for j in range(MC_iter):
        latents=[]
        with torch.no_grad():
            for i, data in enumerate(loader):
                x = data.to(device)
                z = model.encoded(x)
                latents.append(z.detach().cpu().numpy())

            lat = np.concatenate(latents, axis=0)
        #print(latent_stack)
        if j == 0:
            latent_stack = lat.copy()
        else:
            latent_stack = np.dstack((latent_stack, lat))

    model.eval()
    print(latent_stack.shape)
    mean = np.mean(latent_stack, axis = 2)
    std = np.std(latent_stack, axis = 2)
    
    return mean, std



#code from https://github.com/AlexPasqua/Autoencoders
import torch.nn.modules.loss as loss

def contractive_loss(input, target, lambd, ae, reduction: str):
    """
    Actual function computing the loss of a contractive autoencoder
    :param input: (Tensor)
    :param target: (Tensor)
    :param lambd: (float) regularization parameter
    :param ae: (DeepAutoencoder) the model itself, used to get it's weights
    :param reduction: (str) type of reduction {'mean' | 'sum'}
    :raises: ValueError
    :return: the loss
    """
    term1 = (input - target) ** 2
    enc_weights = [ae.encoder[i].weight for i in reversed(range(0, len(ae.encoder), 2))]
    term2 = lambd * torch.norm(torch.chain_matmul(*enc_weights))
    contr_loss = torch.mean(term1 + term2, 0)
    if reduction == 'mean':
        return torch.mean(contr_loss)
    elif reduction == 'sum':
        return torch.sum(contr_loss)
    else:
        raise ValueError(f"value for 'reduction' must be 'mean' or 'sum', got {reduction}")

class ContractiveLoss(loss.MSELoss):
    """
    Custom loss for contractive autoencoders.
    note: the superclass is MSELoss, simply because the base class _Loss is protected and it's not a best practice.
          there isn't a real reason between the choice of MSELoss, since the forward method is overridden completely.
    Overridden for elasticity -> it's possible to use a function as a custom loss, but having a wrapper class
    allows to do:
        criterion = ClassOfWhateverLoss()
        loss = criterion(output, target)    # this line always the same regardless of the type on loss
    """
    def __init__(self, ae, lambd: float, size_average=None, reduce=None, reduction: str = 'sum') -> None:
        super(ContractiveLoss, self).__init__(size_average, reduce, reduction)
        self.ae = ae
        self.lambd = lambd

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return contractive_loss(input, target, self.lambd, self.ae, self.reduction)
