import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import glob
import os
from model import NFBaseModel # Imports the architecture from model.py

# --- SETTINGS ---
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001

monkeys = {'beignet': 89, 'affi': 239}

for monkey, n_channels in monkeys.items():
    print(f"\n=== Training {monkey.upper()} ===")
    
    # 1. Load Data
    files = glob.glob(f"train_data_{monkey}*.npz")
    if not files: 
        print(f"❌ No files found for {monkey}")
        continue
        
    all_data = []
    for f in files:
        with np.load(f) as data:
            key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
            all_data.append(data[key])
    full_data = np.concatenate(all_data, axis=0)
    
    # 2. Stats (Z-Score)
    avg = np.mean(full_data, axis=(0, 1), keepdims=True)
    std = np.std(full_data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0 # Safety
    
    np.savez(f"stats_{monkey}.npz", average=avg, std=std)
    
    # 3. Normalize
    X_norm = (full_data - avg) / std
    
    # 4. Prepare Batches
    # Input: (Batch, 10, C*F) - Flattened features
    X_input = X_norm[:, :10, :, :]
    B, T, C, F = X_input.shape
    X_flat = X_input.reshape(B, T, C*F)
    
    # Target: (Batch, 10, C) - Only Feature 0
    Y_target = X_norm[:, 10:, :, 0]
    
    dataset = TensorDataset(torch.FloatTensor(X_flat), torch.FloatTensor(Y_target))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Train
    model = NFBaseModel(num_channels=n_channels, num_features=9, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx) # bx is already flattened
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"model_{monkey}.pth")
            
        if (epoch+1) % 10 == 0:
            print(f" Epoch {epoch+1} | Loss: {avg_loss:.6f}")
            
    print(f"✅ Saved Best Model for {monkey} (Loss: {best_loss:.6f})")