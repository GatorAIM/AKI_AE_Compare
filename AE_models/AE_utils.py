import torch

def reconstruction_loss(reconstruction, data):
    loss_fnc = torch.nn.MSELoss()
    return loss_fnc(reconstruction, data)

def train_step_AE(model, train_loader, optimizer, device, **kwargs):
    model.train()
    running_loss = 0.0
    # AE model is unsupervised training so we do not need labels here
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        
        _, reconstruction = model(data)
        loss = reconstruction_loss(reconstruction, data)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    running_loss = running_loss / len(train_loader)
    return running_loss

def val_step_AE(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device)
            _, reconstruction = model(data)
            loss = reconstruction_loss(reconstruction, data)
            running_loss += loss.item()
        
    running_loss = running_loss / len(val_loader)
    return running_loss