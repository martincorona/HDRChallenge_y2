import torch
import numpy as np
import os

# --- Helper: Robust Normalization (The V3 Logic) ---
def normalize(data, average, std):
    # This maps [mean-4std, mean+4std] to [-1, 1]
    # It prevents the massive number explosions you saw in V8
    combine_max = average + 4 * std
    combine_min = average - 4 * std
    
    denominator = combine_max - combine_min
    denominator[denominator == 0] = 1.0
    
    # Formula: 2 * (x - min) / (max - min) - 1
    norm_data = 2 * (data - combine_min) / denominator - 1
    return norm_data

# --- The Architecture: Robust GRU (512 Units) ---
class NFBaseModel(torch.nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=512):
        super(NFBaseModel, self).__init__()
        self.input_size = num_channels * num_features
        self.output_size = num_channels * 10
        
        # We use 512 units (V3 size) because V8 (256) was too weak for Beignet
        self.encoder = torch.nn.GRU(
            input_size=self.input_size, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3 # Balanced dropout
        )
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.output_size)
        )

    def forward(self, x):
        # x shape: (Batch, 10, C*F)
        batch_size = x.shape[0]
        
        out, _ = self.encoder(x)
        last_hidden = out[:, -1, :] 
        
        prediction_flat = self.head(last_hidden)
        
        return prediction_flat.reshape(batch_size, 10, -1)

# --- Wrapper matches competition format ---
class Model(torch.nn.Module):
    def __init__(self, monkey_name='beignet'):
        super(Model, self).__init__()
        self.monkey_name = monkey_name
        self.channels = 89 if monkey_name == 'beignet' else 239
        self.model = NFBaseModel(self.channels)
        
        self.average = None
        self.std = None
        
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
        base = os.path.dirname(__file__)
        path = os.path.join(base, f"model_{self.monkey_name}.pth")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, x):
        # 1. Normalize (Using the Safe V3 Logic)
        if self.average is not None:
            x_norm = normalize(x, self.average, self.std)
        else:
            x_norm = x
            
        # 2. Predict
        input_data = x_norm[:, :10, :, :]
        batch, time, chan, feat = input_data.shape
        input_flat = input_data.reshape(batch, time, -1)
        input_tensor = torch.FloatTensor(input_flat)
        
        with torch.no_grad():
            output_norm = self.model(input_tensor).numpy()

        # 3. Denormalize (Reversing V3 Logic)
        if self.average is not None:
            avg_0 = self.average[:, :, :, 0]
            std_0 = self.std[:, :, :, 0]
            
            combine_max = avg_0 + 4 * std_0
            combine_min = avg_0 - 4 * std_0
            rng = combine_max - combine_min
            rng[rng==0] = 1.0
            
            output_raw = (output_norm + 1) / 2 * rng + combine_min
        else:
            output_raw = output_norm

        return np.concatenate([x[:, :10, :, 0], output_raw], axis=1)