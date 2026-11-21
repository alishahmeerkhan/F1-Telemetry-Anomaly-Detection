import os
import sys
import torch
import joblib
import pandas as pd
import torch.nn as nn
from rich import print
from preprocess import load_and_merge_training_data

AUTOENCODER_PATH = 'models/autoencoder.pth'
SCALAR_PATH = 'models/minmax_scalar.pkl'
DATA_PATH = 'data/raw'

class TelemetryAutoencoder(nn.Module):
    def __init__(self):
        super(TelemetryAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5,16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

def detect():
    print(f'> Loading Autoencoder from {AUTOENCODER_PATH}.')
    autoencoder = load_autoencoder()
    print(f'> Loading telemetry data from {DATA_PATH}.')
    df = prepare_data()

    print(f'\n\n[bold green]Models and Data successfully loaded.[/bold green]')

def prepare_data():
    df_merged = load_and_merge_training_data(['R'])
    print(f'\n\nData loaded and merged into dataframe -> {df_merged.shape}')

    scalar = load_scalar()
    df_scaled = scalar.transform(df_merged)
    df = torch.FloatTensor(df_scaled)
    
    return df

def load_autoencoder():
    if os.path.exists(AUTOENCODER_PATH):
        model = TelemetryAutoencoder()
        model.load_state_dict(torch.load(AUTOENCODER_PATH, weights_only=True))
        model.eval()
        return model

    print(f'[bold red]ERROR:[/bold red] Autoencoder model does not exist at {AUTOENCODER_PATH}. Please verify file path.')
    sys.exit()

def load_scalar():
    if os.path.exists(SCALAR_PATH):
        scalar = joblib.load(SCALAR_PATH)
        return scalar

    print(f'[bold red]ERROR:[/bold red] Min-Max Scalar model does not exist at {SCALAR_PATH}. Please verify file path.')
    sys.exit()

if __name__ == '__main__':
    print('Starting detection for anomaly...')
    detect()