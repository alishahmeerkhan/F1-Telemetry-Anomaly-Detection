import os
import torch
import joblib
import argparse
import pandas as pd
import torch.nn as nn
from rich import print
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_and_merge_training_data

training_sessions = ['FP1', 'FP2', 'FP3', 'Q']

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

def prepare_data():
    df = load_and_merge_training_data(training_sessions)
    print(f'\n\nData loaded and merged into dataframe -> {df.shape}')

    X_train, X_val = train_test_split(df.astype('float32').values, test_size=0.2, random_state=42)

    train_tensor = torch.FloatTensor(X_train)
    val_tesnor = torch.FloatTensor(X_val)

    return train_tensor, val_tesnor

def train_autoencoder():
    print('> Loading and preparing data...')
    train_tensor, val_tensor = prepare_data()

    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=64, shuffle=False)

    model = TelemetryAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    print('\n\nStarting Training...')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)

            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f'Epoch [{epoch + 1} / {epochs}] | Train MSE {avg_train:.6f} | Val MSE: {avg_val:.6f}')

    os.makedirs('models/baseline', exist_ok=True)
    torch.save(model.state_dict(), 'models/baseline/baseline_autoencoder.pth')

    print('Training complete. Model and Scalar saved to ../models/baseline directory.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose between two baseline models for training.')

    parser.add_argument('--autoencoder', action='store_true', help='Train the baseline autoencoder without Min-Max Scaling')
    parser.add_argument('--pca', action='store_true', help='Train a baseline PCA model.')

    args = parser.parse_args()

    if not args.autoencoder and not args.pca:
        print('Please enter a valid argument.')
        print('You can run [bold italic cyan]python baseline_model.py --help[/bold italic cyan] to see valid arguments.')

    if args.autoencoder:
        print('MODE: [bold cyan]Training Autoencoder without Min-Max Scaling...[/bold cyan]')
        train_autoencoder()

    if args.pca:
        print('MODE: [bold cyan]Training PCA as baseline model...[/bold cyan]')
        print('[bold yellow]Sorry, its implementation is not complete yet...[/bold yellow]')