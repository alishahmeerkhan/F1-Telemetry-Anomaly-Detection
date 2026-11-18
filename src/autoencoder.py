import os
import torch
import joblib
import pandas as pd
import torch.nn as nn
from rich import print
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from preprocess import preprocess_for_autoencoder, load_and_merge_training_data

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

    X_train, X_val = train_test_split(df.values, test_size=0.2, random_state=42)
    X_train_scaled, X_val_scaled, scalar = preprocess_for_autoencoder(X_train, X_val)

    train_tensor = torch.FloatTensor(X_train_scaled)
    val_tesnor = torch.FloatTensor(X_val_scaled)

    return train_tensor, val_tesnor, scalar

def train_model():
    print('> Loading and preparing data...')
    train_tensor, val_tensor, scalar = prepare_data()

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

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/autoencoder.pth')
    joblib.dump(scalar, 'models/minmax_scalar.pkl')

    print('Training complete. Model and Scalar saved to ../models/ directory.')

if __name__ == '__main__':
    train_model()