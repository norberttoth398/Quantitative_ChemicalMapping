import torch
from torch import nn
from torch.utils.data import Dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def element(in_channel, out_channel, activation = nn.LeakyReLU(0.02)):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                activation,
            ]

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

class SSFeatureDataset(Dataset):
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

class SSInferenceDataset(Dataset):
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
    