import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
from model import Model  # Imports your actual class

def visualize(monkey_name):
    print(f"\nGeneratng graphs for {monkey_name.upper()}...")
    
    # 1. Load Data
    files = glob.glob(f"train_data_{monkey_name}*.npz")
    if not files:
        print(f"No data found for {monkey_name}")
        return
        
    # Load first file
    data = np.load(files[0])
    key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
    # Shape: (Samples, 20, Channels, Features)
    # We only care about Feature 0 (Raw Signal)
    full_data = data[key][:, :, :, 0]
    
    # 2. Load Your Model
    model = Model(monkey_name)
    model.load() # Loads weights + stats
    
    # 3. Pick a random sample (e.g., Sample #0)
    sample_idx = 0
    
    # Get inputs (Batch=1, Time=20, Channels, Features=9)
    # Note: We pass the full 20 steps, but your predict() function 
    # intelligently slices just the first 10, as written in model.py.
    sample_input = data[key][sample_idx:sample_idx+1] 
    
    # 4. Predict
    # Returns (1, 20, Channels)
    prediction = model.predict(sample_input)
    
    # 5. Plot the first 3 Channels
    for channel_idx in range(3):
        plt.figure(figsize=(10, 5))
        
        # A. Plot Ground Truth (The actual recorded brain wave)
        # We plot all 20 steps (0-19)
        truth = full_data[sample_idx, :, channel_idx]
        plt.plot(range(20), truth, color='black', linestyle='--', label='Target (Actual)')
        
        # B. Plot Prediction (What your model thought would happen)
        # We only plot the FUTURE (Steps 10-19)
        pred_future = prediction[0, 10:, channel_idx]
        plt.plot(range(10, 20), pred_future, color='red', linewidth=2, label='Model Prediction')
        
        # C. Formatting (Vertical Line at "Now")
        plt.axvline(x=10, color='green', linestyle=':', label='Start of Prediction')
        plt.title(f'Monkey: {monkey_name} | Channel {channel_idx}')
        plt.xlabel('Time Steps (ms)')
        plt.ylabel('Neural Activity')
        plt.legend()
        
        # D. Save to file
        filename = f"plot_{monkey_name}_ch{channel_idx}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

# Run for both
visualize('beignet')
visualize('affi')
