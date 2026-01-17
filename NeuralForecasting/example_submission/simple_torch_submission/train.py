import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os

HIDDEN_DIM = 512

class NFBaseModel(nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=HIDDEN_DIM):
        super(NFBaseModel, self).__init__()
        # Input is now Channels * Features (e.g. 89 * 9)
        self.input_size = num_channels * num_features
        self.output_size = num_channels # We still only predict the raw signal (1 feature)
        
        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_size)

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Features)
        batch, time, channels, features = x.shape
        x_flat = x.reshape(batch, time, -1) # Flatten channels*features
        
        out, _ = self.encoder(x_flat)
        out = self.fc(out)
        return out

monkeys = {'beignet': 89, 'affi': 239}

for monkey, n_channels in monkeys.items():
    print(f"\n=== Training {monkey.upper()} (V3 - Full Features) ===")
    
    files = glob.glob(f"train_data_{monkey}*.npz")
    if not files: continue
        
    all_data = []
    for f in files:
        with np.load(f) as data:
            key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
            all_data.append(data[key])
    full_data = np.concatenate(all_data, axis=0)
    
    # 1. Calculate Stats for ALL features
    # Shape: (1, 1, Channels, 9)
    avg = np.mean(full_data, axis=(0, 1), keepdims=True)
    std = np.std(full_data, axis=(0, 1), keepdims=True)
    
    # Save 4D stats
    np.savez(f"stats_{monkey}.npz", average=avg, std=std)
    
    # 2. Normalize Everything
    X_norm = (full_data - avg) / (4 * std)
    
    # 3. Prepare Tensors
    # Input: All 9 features
    X_train = torch.FloatTensor(X_norm[:, :10, :, :])
    # Target: Only Feature 0 (Raw Signal) because that's what we predict
    Y_train = torch.FloatTensor(X_norm[:, 10:, :, 0])
    
    model = NFBaseModel(num_channels=n_channels, num_features=9)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    print("Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step(loss)
        
        # Save only if best
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"model_{monkey}.pth")
        
        if (epoch+1) % 10 == 0:
            print(f" Epoch {epoch+1} | Loss: {loss.item():.6f}")

    print(f"✅ Finished. Best Loss: {best_loss:.6f}")
