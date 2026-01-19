#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_concat(csv_paths):
    dfs = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=df.columns[0])  # Tri par timestamp
    df = df.reset_index(drop=True)
    return df


def preprocess(df):
    # Suppression des colonnes inutiles (timestamp, etc.)
    # On garde open, high, low, close, volume, vwap
    cols = [
        'Open', 'High', 'Low', 'Close',
        'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price'
    ]
    # Correction noms si besoin
    df.columns = [c.strip() for c in df.columns]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante: {c}")
    # Nettoyage : suppression des lignes avec NaN ou inf
    before = len(df)
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    after = len(df)
    print(f"Lignes supprimées lors du nettoyage : {before - after}")
    data = df.values.astype(np.float32)
    # Normalisation
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


def create_sequences_stream(
    data, seq_len=24*60, pred_offset=60, max_sequences=None
):
    """Génère les séquences une par une pour éviter l'explosion mémoire."""
    count = 0
    for i in range(len(data) - seq_len - pred_offset + 1):
        X = data[i:i+seq_len]
        y = data[i+seq_len+pred_offset-1][3]  # 'Close' price
        yield X, y
        count += 1
        if max_sequences is not None and count >= max_sequences:
            break


def save_sequences_stream(
    data, split_ratios=(0.7, 0.15, 0.15),
    seq_len=24*60, pred_offset=60, max_sequences=None
):
    total = len(data) - seq_len - pred_offset + 1
    if max_sequences is not None:
        total = min(total, max_sequences)
    n_train = int(total * split_ratios[0])
    n_val = int(total * split_ratios[1])
    n_test = total - n_train - n_val

    files = [
        ("btc_preprocessed_train.npz", n_train),
        ("btc_preprocessed_val.npz", n_val),
        ("btc_preprocessed_test.npz", n_test)
    ]
    seq_gen = create_sequences_stream(
        data, seq_len, pred_offset, max_sequences
    )
    for fname, n in files:
        X_list, y_list = [], []
        for _ in range(n):
            try:
                X, y = next(seq_gen)
                X_list.append(X)
                y_list.append(y)
            except StopIteration:
                break
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        np.savez_compressed(fname, X=X_arr, y=y_arr)
        print(f"{fname}: {len(X_arr)} séquences sauvegardées.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess_data.py coinbase.csv bitstamp.csv")
        sys.exit(1)
    df = load_and_concat(sys.argv[1:])
    data_scaled, scaler = preprocess(df)
    # Limitez le nombre de séquences pour éviter l'explosion mémoire
    save_sequences_stream(data_scaled, max_sequences=100_000)
    # Sauvegarde du scaler pour l'inférence
    import joblib
    joblib.dump(scaler, "btc_scaler.save")
