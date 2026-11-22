import os
import sys
import torch
import joblib
import pandas as pd
import torch.nn as nn
from rich import print
import matplotlib.pyplot as plt
from preprocess import load_and_merge_training_data

AUTOENCODER_PATH = 'models/autoencoder.pth'
SCALAR_PATH = 'models/minmax_scalar.pkl'
DATA_PATH = 'data/raw'
THRESHOLD = 0.12

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
    df_raw, race_tensor = prepare_data()

    print(f'\n\n[bold green]Models and Data successfully loaded.[/bold green]')

    with torch.no_grad():
        reconstructed = autoencoder(race_tensor)

    mse_per_row = torch.mean((race_tensor - reconstructed) ** 2, dim=1).numpy()

    df_raw['MSE'] = mse_per_row
    df_raw['isAnomaly'] = df_raw['MSE'] > THRESHOLD

    anomalies = df_raw[df_raw['isAnomaly'] == True]
    generate_report(mse_per_row.max(), len(anomalies), len(df_raw))
    generate_graph(df_raw)

def generate_report(max_mse, size_anomalies, size_df):
    print('\n\n' + '-' * 75)
    print(f'[bold cyan]Detection Complete.[/bold cyan]')
    print(f'Total rows scanned: {size_df}')
    print(f'Anomalies flagged (MSE > {THRESHOLD}): [bold red]{size_anomalies}[/bold red]')
    print(f'Max MSE recorded: {max_mse:.6f}')
    print('-' * 75)

def generate_graph(df_raw):
    plt.figure(figsize=(14, 5))
    plt.plot(df_raw.index, df_raw['MSE'], color='blue', label='Reconstruction Error (MSE)')
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
    plt.title('2026 Monza Race — Autoencoder Anomaly Detection (LEC)')
    plt.xlabel('Telemetry Row Index')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.show()

def prepare_data():
    df_merged = load_and_merge_training_data(['R'])
    df_merged['nGear'] = df_merged['nGear'].clip(0, 8)
    print(f'\n\nData loaded and merged into dataframe -> {df_merged.shape}')

    scalar = load_scalar()
    df_scaled = scalar.transform(df_merged.values)
    tensor_data = torch.FloatTensor(df_scaled)
    
    return df_merged, tensor_data

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