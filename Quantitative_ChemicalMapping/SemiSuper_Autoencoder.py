import torch
import time
from torch import nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def element(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

class FeatureDataset(Dataset):
    def __init__(self, x, labels):
        #self.one_hot_labels = nn.functional.one_hot(torch.Tensor(labels).to(torch.int64))
        self.final_labs = labels
        #self.final_labs[self.final_labs>0] = 1
        if len(x.shape)==2:
            self.x = x
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training

    def __len__(self):
        return self.x.shape[0] 
    
    def __getitem__(self, n): 
        return torch.Tensor(self.x[n]),  self.final_labs[n]

class InferenceDataset(Dataset):
    def __init__(self, x):
        #self.one_hot_labels = nn.functional.one_hot(torch.Tensor(labels).to(torch.int64))
        #self.final_labs = labels
        #self.final_labs[self.final_labs>0] = 1
        if len(x.shape)==2:
            self.x = x
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training

    def __len__(self):
        return self.x.shape[0] 
    
    def __getitem__(self, n): 
        return torch.Tensor(self.x[n])
    

class VariationalLinear(nn.Module):
    def __init__(self, input_size, output_size, prior_mean=0.0, prior_std=1.0):
        super(VariationalLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # Initialize variational parameters
        self.weight_mean = nn.Parameter(torch.randn(output_size, input_size))
        self.weight_logvar = nn.Parameter(torch.zeros(output_size, input_size))
        self.bias_mean = nn.Parameter(torch.randn(output_size))
        self.bias_logvar = nn.Parameter(torch.zeros(output_size))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # Sample weights and biases from the variational distribution
        weight = self.reparameterize(self.weight_mean, self.weight_logvar)
        bias = self.reparameterize(self.bias_mean, self.bias_logvar)

        # Compute the output of the linear layer
        output = torch.matmul(x, weight.t()) + bias.unsqueeze(0)

        return output

    def kl_divergence(self):
        # Compute the KL divergence between the variational distribution
        # and the prior distribution (assuming both are Gaussian)
        kl_weight = 0.5 * torch.sum(
            self.weight_logvar.exp() - self.weight_logvar + self.weight_mean.pow(2) - 1
        )
        kl_bias = 0.5 * torch.sum(
            self.bias_logvar.exp() - self.bias_logvar + self.bias_mean.pow(2) - 1
        )
        return kl_weight + kl_bias
    
################################################################################################
################################################################################################
################################################################################################

class SemiSuperAutoencoder(nn.Module):
    def __init__(self,input_dim = 6, latent_dim = 2, class_dim = 4,hidden_layer_sizes=(64, 32), type = "svi"):
        super(SemiSuperAutoencoder, self).__init__()
        #make sure attributes are assigned
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.type = type
        self.hls = hidden_layer_sizes

        #create encoder architecture
        encoder = element(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += element(self.hls[i], self.hls[i + 1])

        #create decoder architecture
        decoder = element(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += element(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] 

        #build encoder
        self.encode = nn.Sequential(*encoder)

        #bottleneck layer is standard                        
        self.bottleneck = nn.Sequential(nn.Linear(self.hls[-1],self.latent_dim))

        #build decoder
        self.decode = nn.Sequential(*decoder)
        
        #build classifier that is desired
        if self.type == "svi":
            self.classifier = nn.Sequential(VariationalLinear(32, self.class_dim),
                                        nn.Softmax())
        elif self.type == "mc_dropout":
            self.classifier = nn.Sequential(nn.Linear(32, self.class_dim),
                                        nn.Dropout(),
                                        nn.Softmax())

        elif self.type == "deterministic":
            self.classifier = nn.Sequential(nn.Linear(32, self.class_dim),
                                        nn.Softmax())

        else:
            raise ValueError("Type is not right, has to be one of svi, mc_dopout or deterministic.")
        

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to layer before latent space
        return self.encode(x)
    
    def bottleneck_calc(self, x):
        #encodes data to latent space from input
        en = self.encoded(x)
        return self.bottleneck(en)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)
    
    def classified(self, x):
        return self.classifier(x)

    def forward(self, x):
        en = self.encoded(x)
        b = self.bottleneck(en)
        de = self.decoded(b)
        cl = self.classified(en)
        return de, cl
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def inference_numpy(self, x, batch_size = 200):
        #transform real data to latent space using the trained model
        self.eval()

        if self.type == "mc_dropout":
            self.enable_dropout()
        else:
            pass

        latents=[]
        classes = []
        classes_var = []
        self.to(device)

        dataset_ = InferenceDataset(x)
        loader = DataLoader(dataset_,batch_size=batch_size,shuffle=False)
        
        with torch.no_grad():
            self.eval()
            for i, data in enumerate(loader):
                x = data.to(device)
                en = self.encoded(x)
                z = self.bottleneck(en)
                logits = []
                for i in range(50):
                    logits.append(self.classifier(en).detach().cpu().numpy())
                cl = np.mean(logits, axis = 0)
                cl_var = np.var(logits, axis = 0)
                latents.append(z.detach().cpu().numpy())
                classes.append(cl)
                classes_var.append(cl_var)

        self.eval()
        
        return np.concatenate(latents, axis=0), np.concatenate(classes, axis=0), np.concatenate(classes_var, axis = 0)
    

    def train_model(self, optimizers, train_loader, test_loader, n_epoch):

        opt_ae = optimizers[0]
        opt_class = optimizers[1]
        opt_final = optimizers[2]


        for epoch in range(n_epoch):
            
            # Training
            self.train()
            t = time.time()
            total_loss = []
            for i, data in enumerate(train_loader):
                #print(i)
                class_d = data[1][torch.where(data[1] > 0)]
                l = class_d.to(device)
                x = data[0].to(device)

                x_recon, _ = self.forward(x)
                loss_ae = nn.functional.mse_loss(x_recon, x)
                #backprop
                opt_ae.zero_grad()
                loss_ae.backward()
                opt_ae.step()

                if len(l) == 0:
                    pass
                else:
                    #print("class")
                    _, x_cl = self.forward(x)
                    x_class = x_cl[torch.where(data[1] > 0)]
                    loss_class = nn.CrossEntropyLoss()(x_class, l)
                    
                    # Add KL divergence
                    kl_div = 0.
                    for module in self.modules():
                        if isinstance(module, VariationalLinear):
                            kl_div += module.kl_divergence()
                    loss_class += kl_div / (len(train_loader.dataset)/len(x))#modified to take batch size into account
                    
                    #backprop
                    opt_class.zero_grad()
                    loss_class.backward()
                    opt_class.step()

                total_loss.append(loss_ae.detach().item())
            
            ## Testing
            #model.eval()
            #test_loss = []
            #for i, test in enumerate(test_loader):
            #    x = test.to(device)
            #    x_recon = model(x)
            #    loss_ae = crit_ae(x_recon, x)
            #    test_loss.append(loss_ae.item())
            
            # Logging
            avg_loss = sum(total_loss) / len(total_loss)
            #avg_test = sum(test_loss) / len(test_loss)
            training_time = time.time() - t
            
            print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, time: {training_time:.2f} s')

    
    def train_classifier(self, optimizer, train_loader, n_epoch):

        opt_class = optimizer
        for epoch in range(n_epoch):
            
            # Training
            self.train()
            t = time.time()
            total_loss = []
            for i, data in enumerate(train_loader):
                #print(i)
                class_d = data[1][torch.where(data[1] > 0)]
                l = class_d.to(device)
                x = data[0].to(device)

                if len(l) == 0:
                    pass
                else:
                    #print("class")
                    _, x_cl = self.foward(x)
                    x_class = x_cl[torch.where(data[1] > 0)]
                    loss_class = nn.CrossEntropyLoss()(x_class, l)
                    # Add KL divergence
                    kl_div = 0.
                    for module in self.modules():
                        if isinstance(module, VariationalLinear):
                            kl_div += module.kl_divergence()
                    loss_class += kl_div / (len(train_loader.dataset)/len(x))#modified to take batch size into account
                    #loss_class = loss_class*10
                    #backprop
                    opt_class.zero_grad()
                    loss_class.backward()
                    opt_class.step()

                total_loss.append(loss_class.detach().item())

            avg_loss = sum(total_loss) / len(total_loss)
            training_time = time.time() - t
            
            print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, time: {training_time:.2f} s')