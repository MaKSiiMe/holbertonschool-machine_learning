#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import math

# --- PARAMÈTRES ---
SEQ_LEN = 24       # Fenêtre de 24h
PRED_HORIZON = 1   # Prédiction à T+1h
BATCH_SIZE = 64    # Optimisé pour GPU
EPOCHS = 15        # Avec EarlyStopping


def create_dataset_windows(data, seq_len):
    """Crée les fenêtres glissantes X et la cible y"""
    X, y = [], []
    # On assume que la colonne 'Close' est à l'index 0
    target_col_index = 0

    # data est un numpy array (n_samples, n_features)
    for i in range(len(data) - seq_len - PRED_HORIZON):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len + PRED_HORIZON - 1, target_col_index])

    return np.array(X), np.array(y)


def main():
    # 1. Chargement et feature engineering (on refait le preprocessing
    #    ici pour pouvoir fitter le scaler sur le train uniquement)
    print("--- Chargement des données brutes et feature engineering ---")
    try:
        df1 = pd.read_csv('coinbase.csv')
        df2 = pd.read_csv('bitstamp.csv')
    except FileNotFoundError:
        # fallback to preprocessed file if raw csv not available
        if not os.path.exists('preprocessed_btc.csv'):
            print("Erreur: données brutes manquantes et preprocessed_btc.csv absent")
            return
        df = pd.read_csv('preprocessed_btc.csv', index_col=0, parse_dates=True)
        data = df.values.astype('float32')
        print("Utilisation du fichier preprocessed_btc.csv existant")
        X, y = create_dataset_windows(data, SEQ_LEN)
        # load scaler for inverse transform only
        if os.path.exists('btc_scaler.save'):
            scaler = joblib.load('btc_scaler.save')
        else:
            scaler = None
        print("--- Création des séquences ---")
        # continue to evaluation/training below
    else:
        df = pd.concat([df1, df2])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.set_index('Timestamp')
        df = df.sort_index()
        keep_cols = ['Close', 'Volume_(BTC)']
        df = df[keep_cols]
        df_hourly = df.resample('h').mean()
        df_hourly = df_hourly.ffill()
        df_hourly['MA24'] = df_hourly['Close'].rolling(window=24).mean()
        df_hourly = df_hourly.dropna()

        # split by time so scaler is fit on train only
        n_rows = len(df_hourly)
        train_end = int(n_rows * 0.8)
        val_end = int(n_rows * 0.9)

        df_train = df_hourly.iloc[:train_end]
        df_val = df_hourly.iloc[train_end:val_end]
        df_test = df_hourly.iloc[val_end:]

        scaler = MinMaxScaler()
        scaler.fit(df_train.values)

        data_scaled = scaler.transform(df_hourly.values).astype('float32')
        # save scaler
        joblib.dump(scaler, 'btc_scaler.save')

        # create windows from scaled data
        print("--- Création des séquences ---")
        X, y = create_dataset_windows(data_scaled, SEQ_LEN)

    # 3. Split Chronologique (Train 80% / Val 10% / Test 10%)
    n = len(X)
    train_split = int(n * 0.8)
    val_split = int(n * 0.9)

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]

    # Création tf.data.Dataset
    def make_tf_ds(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(10000)
        # .repeat() permet de gérer les epochs manuellement via steps_per_epoch
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()

    train_ds = make_tf_ds(X_train, y_train, shuffle=True)
    val_ds = make_tf_ds(X_val, y_val, shuffle=False)
    test_ds = make_tf_ds(X_test, y_test, shuffle=False)

    steps_per_epoch = len(X_train) // BATCH_SIZE
    val_steps = len(X_val) // BATCH_SIZE

    # 4. Modèle
    print("--- Construction du Modèle ---")
    model = Sequential([
        # Couche 1 : Capture les patterns complexes
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.3),
        # Couche 2 : Synthétise l'info
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        # Sortie
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True
    )

    print("--- Entraînement ---")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[early_stop]
    )

    # 5. Évaluation et Graphique
    print("--- Génération du Graphique Final et diagnostics ---")

    last_n = 200
    X_sample = X_test[-last_n:]
    y_sample_true = y_test[-last_n:]

    preds = model.predict(X_sample, verbose=0)

    def inverse_prices(target_array, scaler):
        dummy = np.zeros((len(target_array), scaler.n_features_in_))
        dummy[:, 0] = target_array.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    real_prices = inverse_prices(y_sample_true, scaler)
    predicted_prices = inverse_prices(preds, scaler)

    # Diagnostics
    mse = float(np.mean((predicted_prices - real_prices) ** 2))
    mae = float(np.mean(np.abs(predicted_prices - real_prices)))
    # R^2
    ss_res = float(np.sum((real_prices - predicted_prices) ** 2))
    ss_tot = float(np.sum((real_prices - np.mean(real_prices)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    # Directional accuracy
    true_delta = np.diff(real_prices)
    pred_delta = np.diff(predicted_prices)
    dir_acc = float(np.mean((np.sign(true_delta) == np.sign(pred_delta)).astype(float)))

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Direction accuracy: {dir_acc:.4f}")

    # Plot without smoothing
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label='Prix Réel (USD)', color='blue', linewidth=1.0)
    plt.plot(predicted_prices, label='Prédiction (USD)', color='orange', linestyle='--', linewidth=1.0)
    plt.title(f"Prédiction BTC vs Réalité (dernier {last_n} points) — sans lissage")
    plt.xlabel("Heures")
    plt.ylabel("Prix ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('btc_prediction_graph_nosmooth.png')

    # also save model
    model.save('btc_model.keras')

    print("✅ Graphique sauvegardé : btc_prediction_graph_nosmooth.png")
    print("✅ Modèle sauvegardé : btc_model.keras")


if __name__ == '__main__':
    main()
