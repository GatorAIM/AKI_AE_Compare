import torch
from torch import nn

class AE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, activation):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function '{activation}'. \
            Choose 'relu' or 'sigmoid'.")
 
                      
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            self.activation,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction