import torch
import numpy as np

def reconstruction_loss(reconstruction, data):
    loss_fnc = torch.nn.MSELoss()
    return loss_fnc(reconstruction, data)

def DAE_swap_noise(batch, noise_ratio):
    noisy_batch = batch.clone()
    num_rows, num_cols = noisy_batch.shape
    num_noisy_cells = int(num_rows * num_cols * noise_ratio)
    
    row_indices = np.random.randint(0, num_rows, num_noisy_cells)
    col_indices = np.random.randint(0, num_cols, num_noisy_cells)
    new_row_indices = np.random.randint(0, num_rows, num_noisy_cells)
    
    temp_values = noisy_batch[row_indices, col_indices].clone()
    noisy_batch[row_indices, col_indices] = noisy_batch[new_row_indices, col_indices]
    noisy_batch[new_row_indices, col_indices] = temp_values
    return noisy_batch

def train_step_DAE(model, train_loader, optimizer, device, **kwargs):
    model.train()
    running_loss = 0.0
    for batch_idx, (clean_data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        noise_data = DAE_swap_noise(clean_data, 0.10)
        clean_data = clean_data.to(device)
        noise_data = noise_data.to(device)
        
        _, reconstruction = model(noise_data)
        loss = reconstruction_loss(reconstruction, clean_data)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    running_loss = running_loss / len(train_loader)
    return running_loss

def val_step_DAE(model, val_loader, device):
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