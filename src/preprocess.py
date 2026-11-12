import pandas as pd
from pathlib import Path
from os.path import exists
from sklearn.preprocessing import MinMaxScaler

def load_and_merge_training_data(sessions):
    final_df = []
    features = ['Speed', 'RPM', 'nGear', 'Throttle', 'Brake']

    for session in sessions:
        filepath = base_path = Path(__file__).parent.parent / f'data/raw/2026_Monza_LEC_{session}.csv'
        if exists(filepath):
            df = pd.read_csv(filepath)
            df = df[features].dropna()
            final_df.append(df)

    return pd.concat(final_df, ignore_index=True)

def preprocess_for_autoencoder(df):
    scalar = MinMaxScaler()
    return scalar.fit_transform(df), scalar