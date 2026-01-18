import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import glob
import os
from model import NFBaseModel

# --- SETTINGS ---
HIDDEN_DIM = 512  # Back to the robust size
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001

monkeys = {'beignet': 89, 'affi': 239}

for monkey, n_channels in monkeys.items():
    print(f"\n=== Training {monkey.upper()} (V9: Safe Norm + Smart Train) ===")
    
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
    
    # 2. Stats (Matches V3: Avg +/- 4*Std)
    avg = np.mean(full_data, axis=(0, 1), keepdims=True)
    std = np.std(full_data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0 
    np.savez(f"stats_{monkey}.npz", average=avg, std=std)
    
    # 3. Normalize (The Safe V3 Math)
    # Range is [-1, 1] approximately
    X_norm = (full_data - avg) / (4 * std)
    
    # 4. Prepare Batches
    X_input = X_norm[:, :10, :, :]
    B, T, C, F = X_input.shape
    X_flat = X_input.reshape(B, T, C*F)
    Y_target = X_norm[:, 10:, :, 0]
    
    full_dataset = TensorDataset(torch.FloatTensor(X_flat), torch.FloatTensor(Y_target))
    
    # --- 90/10 Split (Keep this! It helps Affi!) ---
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Setup Model
    model = NFBaseModel(num_channels=n_channels, num_features=9, hidden_dim=HIDDEN_DIM)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Huber Loss (Critical for noise)
    criterion = nn.HuberLoss(delta=1.0)
    
    best_val_loss = float('inf')
    
    print(f"Training on {len(train_dataset)} samples | Validating on {len(val_dataset)}")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        # Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"model_{monkey}.pth")
            
        if (epoch+1) % 10 == 0:
            print(f" Epoch {epoch+1} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
            
    print(f"✅ Finished {monkey}. Best Val Loss: {best_val_loss:.6f}")