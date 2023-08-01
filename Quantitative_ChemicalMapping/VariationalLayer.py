import torch
from torch import nn


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
    