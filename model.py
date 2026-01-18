import torch
import numpy as np
import os

# -------------------------------
# Robust V3 Normalization
# -------------------------------
def normalize(data, average, std):
    combine_max = average + 4 * std
    combine_min = average - 4 * std

    denom = combine_max - combine_min
    denom[denom == 0] = 1.0

    data = np.clip(data, combine_min, combine_max)
    return 2 * (data - combine_min) / denom - 1


def denormalize(data, average, std):
    combine_max = average + 4 * std
    combine_min = average - 4 * std

    rng = combine_max - combine_min
    rng[rng == 0] = 1.0

    return (data + 1) / 2 * rng + combine_min


# -------------------------------
# GRU Delta Forecast Model
# -------------------------------
class NFBaseModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout=0.2):
        super().__init__()

        self.encoder = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.delta_head = torch.nn.Linear(hidden_size, input_size)
        self.ln = torch.nn.LayerNorm(hidden_size)

        self.delta_scale = torch.nn.Parameter(torch.ones(input_size) * 0.1)

    def forward(self, x):
        # x: (B, T, D)
        h, _ = self.encoder(x)
        h = self.ln(h)
        h = self.dropout(h)
        delta = self.delta_head(h)
        return x + self.delta_scale * delta


# -------------------------------
# Competition Wrapper
# -------------------------------
class Model(torch.nn.Module):
    def __init__(self, monkey_name='beignet'):
        super().__init__()
        self.monkey_name = monkey_name

        if monkey_name == 'beignet':
            self.channels = 89
        elif monkey_name == 'affi':
            self.channels = 239
        else:
            raise ValueError(f'Unknown monkey {monkey_name}')

        # We only use feature 0 (same as notebook)
        self.features = 1
        self.input_size = self.channels * self.features

        self.model = NFBaseModel(self.input_size)

        self.average = None
        self.std = None

        base = os.path.dirname(__file__)
        stats_path = os.path.join(base, f'stats_{monkey_name}.npz')
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.average = stats['average']
            self.std = stats['std']

    def load(self):
        base = os.path.dirname(__file__)
        model_path = os.path.join(base, f'model_{self.monkey_name}.pth')

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location='cpu',weights_only=True)
            self.model.load_state_dict(state)

        self.model.eval()

    def predict(self, X):
        """
        X: (B, T, C, F)
        Returns:
            (B, 20, C)  -> first 10 observed + 10 predicted
        """

        # -----------------
        # Normalize
        # -----------------
        if self.average is not None:
            Xn = normalize(X, self.average, self.std)
        else:
            Xn = X.copy()

        # Use only feature 0
        Xn = Xn[:, :, :, 0]          # (B, T, C)

        # Initial context
        context = Xn[:, :10, :]      # (B, 10, C)
        preds = []

        inp = context
        inp = inp.reshape(inp.shape[0], inp.shape[1], -1)
        inp = torch.tensor(inp, dtype=torch.float32)

        # -----------------
        # Autoregressive rollout (10 steps)
        # -----------------
        with torch.no_grad():
            for _ in range(10):
                out = self.model(inp)
                next_step = out[:, -1:, :]
                preds.append(next_step)
                inp = torch.cat([inp[:, 1:], next_step], dim=1)

        preds = torch.cat(preds, dim=1).numpy()
        preds = preds.reshape(preds.shape[0], 10, self.channels, 1)

        # -----------------
        # Denormalize
        # -----------------
        if self.average is not None:
            avg0 = self.average[:, :, :, 0]
            std0 = self.std[:, :, :, 0]
            preds = denormalize(preds, avg0, std0)

        # -----------------
        # Output format
        # -----------------
        observed = X[:, :10, :, 0]
        future = preds[:, :, :, 0]

        return np.concatenate([observed, future], axis=1)
