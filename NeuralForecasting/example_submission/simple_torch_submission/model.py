import torch
import numpy as np
import os

def normalize(data, average, std):
    combine_max = average + 4 * std
    combine_min = average - 4 * std
    denom = combine_max - combine_min
    denom[denom == 0] = 1.0
    return 2 * (data - combine_min) / denom - 1

class NFBaseModel(torch.nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=512):
        super(NFBaseModel, self).__init__()
        self.input_size = num_channels * num_features
        self.output_size = num_channels
        self.encoder = torch.nn.GRU(input_size=self.input_size, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, self.output_size)

    def forward(self, x):
        batch, time, _, _ = x.shape
        x_flat = x.reshape(batch, time, -1)
        out, _ = self.encoder(x_flat)
        out = self.fc(out)
        return out

class Model(torch.nn.Module):
    def __init__(self, monkey_name='beignet'):
        super(Model, self).__init__()
        self.monkey_name = monkey_name
        self.channels = 89 if monkey_name == 'beignet' else 239
        self.model = NFBaseModel(self.channels)
        
        self.average = None
        self.std = None
        
        # Load Stats
        base = os.path.dirname(__file__)
        try:
            stats = np.load(os.path.join(base, f'stats_{self.monkey_name}.npz'))
            self.average = stats['average']
            self.std = stats['std']
        except:
            pass

    def load(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, f"model_{self.monkey_name}.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()

    def predict(self, x):
        # x shape: (Batch, 20, Channels, 9)
        
        # 1. Normalize ALL features
        if self.average is not None:
            x_norm = normalize(x, self.average, self.std)
        else:
            x_norm = x
            
        # 2. Predict
        # Input: First 10 steps, ALL features
        input_tensor = torch.FloatTensor(x_norm[:, :10, :, :])
        
        with torch.no_grad():
            output_norm = self.model(input_tensor).numpy()

        # 3. Denormalize
        # We only need stats for Feature 0 to denormalize the output
        if self.average is not None:
            avg_0 = self.average[:, :, :, 0]
            std_0 = self.std[:, :, :, 0]
            
            c_max = avg_0 + 4 * std_0
            c_min = avg_0 - 4 * std_0
            rng = c_max - c_min
            rng[rng==0] = 1.0
            
            output_raw = (output_norm + 1) / 2 * rng + c_min
        else:
            output_raw = output_norm

        return np.concatenate([x[:, :10, :, 0], output_raw], axis=1)
