import torch
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from preprocess import preprocess_for_autoencoder, load_and_merge_training_data

training_sessions = ['FP1', 'FP2', 'FP3', 'Q']

def prepare_data():
    df = load_and_merge_training_data(training_sessions)
    print(f'Data loaded and merged into dataframe -> {df.shape}')

    X_train, X_val = train_test_split(df.values, test_size=0.2, random_state=42)
    X_train_scaled, X_val_scaled, scalar = preprocess_for_autoencoder(X_train, X_val)

    train_tensor = torch.FloatTensor(X_train_scaled)
    val_tesnor = torch.FloatTensor(X_val_scaled)

    return train_tensor, val_tesnor, scalar

def train_model():
    pass

if __name__ == '__main__':
    print('Welcome to Autoencoder training pipeline...')