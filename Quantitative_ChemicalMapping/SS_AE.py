import torch
import time
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

from utils import weights_init, element, SSInferenceDataset
from VariationalLayer import VariationalLinear


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
    
    def inference_numpy(self, x, batch_size = 200, n = 50):
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

        dataset_ = SSInferenceDataset(x)
        loader = DataLoader(dataset_,batch_size=batch_size,shuffle=False)
        
        with torch.no_grad():
            self.eval()
            for i, data in enumerate(loader):
                x = data.to(device)
                en = self.encoded(x)
                z = self.bottleneck(en)
                logits = []
                for i in range(n):
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

    def save_model(self, optimizer, path):
        check_point = {'params': self.state_dict(),                            
                    'optimizer': optimizer.state_dict()}
        torch.save(check_point, path)

    def load_model(self, optimizer=None, path=''):
        check_point = torch.load(path)
        self.load_state_dict(check_point['params'])
        if optimizer is not None:
            optimizer.load_state_dict(check_point['potimizer'])