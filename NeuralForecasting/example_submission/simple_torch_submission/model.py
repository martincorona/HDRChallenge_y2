import torch
import numpy as np
import os

# --- Helper: Standard Z-Score Normalization ---
def normalize(data, average, std):
    # Standard Z-Score: (X - Mean) / Std
    # Safe division: Replace 0 std with 1
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (data - average) / std_safe

# --- The Brain: Encoder-Decoder Architecture ---
class NFBaseModel(torch.nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=512):
        super(NFBaseModel, self).__init__()
        self.input_size = num_channels * num_features
        self.output_size = num_channels * 10  # Predicting 10 steps at once
        
        # 1. Encoder: Reads the past
        self.encoder = torch.nn.GRU(
            input_size=self.input_size, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2 # Helps with noisy Beignet data
        )
        
        # 2. Decoder: Predicts the future
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.output_size)
        )

    def forward(self, x):
        # x shape: (Batch, 10, C*F)
        batch_size = x.shape[0]
        
        # Run GRU over history
        out, _ = self.encoder(x)
        
        # Take the FINAL hidden state (summarizing all 10 steps)
        last_hidden = out[:, -1, :] 
        
        # Predict all 10 future steps
        prediction_flat = self.head(last_hidden)
        
        # Reshape to (Batch, 10, Channels)
        return prediction_flat.reshape(batch_size, 10, -1)

# --- The Wrapper: Matches Competition Format ---
class Model(torch.nn.Module):
    def __init__(self, monkey_name='beignet'):
        super(Model, self).__init__()
        self.monkey_name = monkey_name
        self.channels = 89 if monkey_name == 'beignet' else 239
        
        # Initialize the architecture
        self.model = NFBaseModel(self.channels)
        
        self.average = None
        self.std = None
        
        # Load Stats
        base = os.path.dirname(__file__)
        try:
            stats_path = os.path.join(base, f'stats_{self.monkey_name}.npz')
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.average = stats['average']
                self.std = stats['std']
        except:
            pass

    def load(self):
        # Matches the example strictly
        base = os.path.dirname(__file__)
        path = os.path.join(base, f"model_{self.monkey_name}.pth")
        
        if os.path.exists(path):
            # Load weights into the internal model
            # Added weights_only=True as per guidelines
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, x):
        # x shape: (Batch, 20, Channels, 9)
        # We only use the first 10 steps (0-9)
        
        # 1. Normalize
        if self.average is not None:
            x_norm = normalize(x, self.average, self.std)
        else:
            x_norm = x
            
        # 2. Prepare Input
        # Flatten the features for the encoder
        # Input shape becomes (Batch, 10, Channels * 9)
        input_data = x_norm[:, :10, :, :]
        batch, time, chan, feat = input_data.shape
        input_flat = input_data.reshape(batch, time, -1)
        
        input_tensor = torch.FloatTensor(input_flat)
        
        # 3. Predict
        with torch.no_grad():
            output_norm = self.model(input_tensor).numpy()

        # 4. Denormalize
        if self.average is not None:
            # Revert Z-Score: X = Z * Std + Mean
            avg_0 = self.average[:, :, :, 0] # Feature 0 only
            std_0 = self.std[:, :, :, 0]     # Feature 0 only
            output_raw = output_norm * std_0 + avg_0
        else:
            output_raw = output_norm

        # 5. Return (History + Prediction)
        return np.concatenate([x[:, :10, :, 0], output_raw], axis=1)