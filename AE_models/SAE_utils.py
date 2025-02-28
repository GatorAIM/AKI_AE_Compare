import torch
import numpy as np

def reconstruction_loss(reconstruction, data):
    loss_fnc = torch.nn.MSELoss()
    return loss_fnc(reconstruction, data)

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(rho_hat, dim=0)
    rho_hat = torch.clamp(rho_hat, min=1e-10, max=1-1e-10)
    rho = torch.full_like(rho_hat, rho)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

def criterion_SAE(reconstruction, data, embedding, rho = 0.05, beta = 3):
    return reconstruction_loss(reconstruction, data) + beta * kl_divergence(rho, embedding)

def train_step_SAE(model, train_loader, optimizer, device, **kwargs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        embedding, reconstruction = model(data)
        loss = criterion_SAE(reconstruction, data, embedding)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    running_loss = running_loss / len(train_loader)
    return running_loss

def val_step_SAE(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device)
            embedding, reconstruction = model(data)
            loss = reconstruction_loss(reconstruction, data)
            running_loss += loss.item()
        
    running_loss = running_loss / len(val_loader)
    return running_loss