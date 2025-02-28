import torch
from torch.autograd import Variable

def reconstruction_loss(reconstruction, data):
    loss_fnc = torch.nn.MSELoss()
    return loss_fnc(reconstruction, data)

def contrastive_loss_sigmoid(W, h):
    # Compute the derivative of the hidden layer output
    dh = h * (1 - h)  # Assuming h is the output of a sigmoid activation

    # Calculate the L2 norm of the weights, summing over input dimensions
    w_sum = torch.sum(W**2, dim=1)
    w_sum = w_sum.unsqueeze(1)  # Reshape to (N_hidden, 1) to match dh

    # Compute the contractive loss component
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return contractive_loss

def contrastive_loss_relu(W, X, b):
    z = torch.matmul(X, W.T) + b
    dh_dz = (z > 0).float()
    batch_size = X.size(0)
    W_expand = W.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, hidden_dim, input_dim)
    dh_dz_expand = dh_dz.unsqueeze(2)  # (batch_size, hidden_dim, 1)
    jacobian = W_expand * dh_dz_expand
    frobenius_norm_sq = torch.sum(jacobian ** 2)
    return frobenius_norm_sq

def criterion(reconstruction, data, W, b, embedding, activation, lambd = 1e-4):
    rl = reconstruction_loss(reconstruction, data)
    if activation == 'sigmoid':
        cl = contrastive_loss_sigmoid(W, embedding)
    elif activation == 'relu':
        cl = contrastive_loss_relu(W, data, b)
    return rl + lambd * cl

def train_step_CAE(model, train_loader, optimizer, device, activation, **kwargs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        
        embedding, reconstruction = model(data)
        
        W = model.encoder[0].weight
        b = model.encoder[0].bias
        
        loss = criterion(reconstruction, data, W, b, embedding, activation)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    running_loss = running_loss / len(train_loader)
    return running_loss

def val_step_CAE(model, val_loader, device):
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