import torch
import numpy as np
import glob
from model import Model # Uses your actual submission file

# --- Metrics ---
def r2_score_np(target, output):
    # Avoid division by zero
    target_sum_sq = np.sum(target ** 2)
    if target_sum_sq == 0: return 0.0
    return 1 - (np.sum((target - output) ** 2) / target_sum_sq)

def mse_score_np(target, output):
    return np.mean((target - output) ** 2)

# --- Evaluation Loop ---
def evaluate(monkey_name):
    print(f"\n--- Evaluating {monkey_name.upper()} ---")
    
    # 1. Load Data
    files = glob.glob(f"train_data_{monkey_name}*.npz")
    if not files:
        print("No data found.")
        return

    # Just load the first file for testing
    data = np.load(files[0])
    key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
    
    # The data is (Samples, 20, Channels, Features)
    # We treat this as "Test Data" for now
    # Ideally, you'd split your data, but for a quick check, this works.
    full_data = data[key]
    
    # 2. Load Model
    model = Model(monkey_name)
    try:
        model.load()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    # 3. Predict Batch-by-Batch (to save memory)
    batch_size = 32
    num_samples = len(full_data)
    all_preds = []
    
    print(f"Running inference on {num_samples} samples...")
    
    for i in range(0, num_samples, batch_size):
        batch = full_data[i : i + batch_size]
        
        # Predict using your submission logic
        # Input: (Batch, 20, Channels, Features)
        # Output: (Batch, 20, Channels) -> History + Future
        pred_batch = model.predict(batch)
        all_preds.append(pred_batch)
        
    # Combine
    predictions = np.concatenate(all_preds, axis=0)
    
    # 4. Extract Ground Truth
    # Your model returns: [History (0-9) | Prediction (10-19)]
    # We compare the "Prediction" part against the real data
    
    # Target (Raw Signal, Feature 0)
    targets = full_data[:, :, :, 0]
    
    # Slice: Only look at the future (steps 10 to 19)
    # R2 is calculated on the *prediction*, not the history we already knew.
    future_pred = predictions[:, 10:, :]
    future_target = targets[:, 10:, :]
    
    # 5. Calculate Scores
    r2 = r2_score_np(future_target, future_pred)
    mse = mse_score_np(future_target, future_pred)
    
    print(f"✅ MSE Score (Lower is better): {mse:.6f}")
    print(f"✅ R2 Score  (Higher is better): {r2:.4f}")
    
    if r2 > 0.5:
        print("🌟 result: GREAT MODEL")
    elif r2 > 0.2:
        print("👍 result: DECENT MODEL")
    else:
        print("⚠️ result: NEEDS IMPROVEMENT")

# Run Evaluation
evaluate('beignet')
evaluate('affi')
